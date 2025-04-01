
from dataclasses import dataclass 
@dataclass
class train_config:
    model_name: str = "LLama path"
    tokenizer_name: str = None
    enable_fsdp: bool = True
    low_cpu_fsdp: bool = True
    run_validation: bool = True
    batch_size_training: int = 1
    batching_strategy: str = "packing"
    context_length: int = 512
    gradient_accumulation_steps: int = 1
    gradient_clipping: bool = True
    gradient_clipping_threshold: float = 1.0
    num_epochs: int = 1
    max_train_step: int = 0
    max_eval_step: int = 0
    num_workers_dataloader: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    gamma: float = 0.85
    seed: int = 42
    use_fp16: bool = True
    mixed_precision: bool = True
    val_batch_size: int = 1
    dataset: str = ""
    peft_method: str = "lora"
    use_peft: bool = True
    from_peft_checkpoint: str = ""
    output_dir: str = "PEFT"
    freeze_layers: bool = True
    num_freeze_layers: int = 12
    quantization: str = None
    save_model: bool = True