import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from contextlib import nullcontext
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate.utils import is_xpu_available, is_ccl_available
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
import time
from llama_recipes.utils.memory_utils import MemoryTrace
from llama_recipes.utils.train_utils import profile

def train_custom_model(model, train_dataloader, eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, fsdp_config=None, local_rank=None, rank=None, wandb_run=None, model_mode=None):
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()

    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss = []

    if train_config.save_metrics:
        if not os.path.exists(train_config.output_dir):
            os.makedirs(train_config.output_dir, exist_ok=True)
        metrics_filename = f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False

    for epoch in range(train_config.num_epochs):
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader) // gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            with profile(train_config, local_rank) as profile_context:
                for step, batch in enumerate(train_dataloader):
                    total_train_steps += 1
                    if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                        max_steps_reached = True
                        if not train_config.enable_fsdp or local_rank == 0:
                            print("max training steps reached, stopping training, total train steps finished: ", total_train_steps-1)
                        break

                    input_ids = batch["input_ids"].to(local_rank if train_config.enable_fsdp else 'cuda:2')
                    extra_input_ids = batch["extra_input_ids"].to(local_rank if train_config.enable_fsdp else 'cuda:2')
                    attention_mask = batch["attention_mask"].to(local_rank if train_config.enable_fsdp else 'cuda:2')
                    labels = batch["labels"].to(local_rank if train_config.enable_fsdp else 'cuda:2')
                    if model_mode=="rank":
                        rank_id_labels = batch["rank_id_labels"].to(local_rank if train_config.enable_fsdp else 'cuda:2')
                    else:
                        rank_id_labels = batch["rank_id_labels"].to(local_rank if train_config.enable_fsdp else 'cuda:2')
                        reason_labels = batch["reason_labels"].to(local_rank if train_config.enable_fsdp else 'cuda:2')
                    # print("batch[extra_input_ids]", batch["extra_input_ids"].shape)
                    # print("batch[input_ids]", batch["input_ids"].shape)
                    loss = None
                    with autocast():
                        if model_mode=="rank":
                            outputs = model(input_ids=input_ids, 
                                            extra_input_ids=extra_input_ids, 
                                            attention_mask=attention_mask, 
                                            labels=labels,
                                            output_hidden_states=True, 
                                            rank_id_labels=rank_id_labels,
                                            output_attentions=True)
                        elif model_mode=="reason":
                            outputs = model(input_ids=input_ids, 
                                            extra_input_ids=extra_input_ids,
                                            attention_mask=attention_mask,
                                            labels=labels, 
                                            rank_id_labels=rank_id_labels,
                                            reason_labels=reason_labels,
                                            output_hidden_states=True,
                                            output_attentions=True)
                        else:
                            outputs = model(input_ids=input_ids, 
                                            extra_input_ids=extra_input_ids,
                                            attention_mask=attention_mask,
                                            labels=labels, 
                      
                                            output_hidden_states=True,
                                            output_attentions=True)
                        # print("output", outputs)
                        try:
                            loss = outputs["loss"]
                        except Exception as e:
                            loss =outputs.loss
                        # loss = outputs["loss"]
                    # print("loss",loss)
                    loss = loss / gradient_accumulation_steps
                    if train_config.save_metrics:
                        train_step_loss.append(loss.detach().float().item())
                        train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                    total_loss += loss.detach().float()

                    if train_config.use_fp16:
                        scaler.scale(loss).backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                scaler.unscale_(optimizer)
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            pbar.update(1)
                    else:
                        loss.backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            optimizer.step()
                            optimizer.zero_grad()
                            pbar.update(1)

                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()
                    if train_config.flop_counter and profile_context.is_done():
                        TFlops = profile_context.get_flops_per_sec() / 1e12
                    if wandb_run:
                        if not train_config.enable_fsdp or rank == 0:
                            wandb_run.log({
                                'train/epoch': epoch + 1,
                                'train/step': epoch * len(train_dataloader) + step,
                                'train/loss': loss.detach().float(),
                            })

                    pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

                    if train_config.save_metrics:
                        save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
                pbar.close()

        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)

        if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        elif torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss / world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if not train_config.enable_fsdp or rank == 0:
            memtrace.print_stats()

        lr_scheduler.step()
        if train_config.run_validation:
            
            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = custom_evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run, model_mode=model_mode)

            if train_config.save_metrics:
                val_step_loss.extend(temp_val_loss)
                val_step_perplexity.extend(temp_step_perplexity)

            checkpoint_start_time = time.perf_counter()
            # if train_config.save_model and eval_epoch_loss < best_val_loss:
            if train_config.save_model:
                if train_config.enable_fsdp:
                    dist.barrier()
                if train_config.use_peft:
                    if train_config.enable_fsdp:
                        if rank == 0:
                            print(f"we are about to save the PEFT modules")
                    else:
                        print(f"we are about to save the PEFT modules")
                    model.save_pretrained(train_config.output_dir)
                    if train_config.enable_fsdp:
                        if rank == 0:
                            print(f"PEFT modules are saved in {train_config.output_dir} directory")
                    else:
                        print(f"PEFT modules are saved in {train_config.output_dir} directory")

                else:
                    if not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                        save_model_checkpoint(model, optimizer, rank, train_config, epoch=epoch)
                    elif not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                        print("Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                        save_model_and_optimizer_sharded(model, rank, train_config)
                        if train_config.save_optimizer:
                            save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
                            print("Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")

                    if not train_config.use_peft and train_config.save_optimizer:
                        save_optimizer_checkpoint(model, optimizer, rank, train_config, epoch=epoch)
                        print("Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                if train_config.enable_fsdp:
                    dist.barrier()
            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
            checkpoint_times.append(checkpoint_end_time)
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                if train_config.enable_fsdp:
                    if rank == 0:
                        print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
                else:
                    print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
            val_loss.append(float(best_val_loss))
            val_prep.append(float(eval_ppl))
        if train_config.enable_fsdp:
            if rank == 0:
                print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        else:
            print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")

        if train_config.save_metrics:
            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times) / len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep) / len(train_prep)
    avg_train_loss = sum(train_loss) / len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep) / len(val_prep)
        avg_eval_loss = sum(val_loss) / len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename
    if train_config.flop_counter:
        results["model_tflops"] = TFlops
    if train_config.enable_fsdp and not train_config.use_peft and rank == 0:
        save_train_params(train_config, fsdp_config, rank)

    return results
