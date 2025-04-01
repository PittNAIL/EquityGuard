# Check if GPU is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
import torch.nn as nn
import torch.nn.functional as F
# from transformers import LlamaModel
from tqdm import tqdm 
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import dataclasses
from collections import Counter
import random
import torch.optim as optim
# from peft import get_peft_model, PeftModel
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy
)

from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaConfig,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from configs.fsdp import fsdp_config as FSDP_CONFIG
from configs.training import train_config as TRAIN_CONFIG
from configs.quantization import quantization_config  as QUANTIZATION_CONFIG
# from llama_recipes.data.concatenator import ConcatDataset
from custom.custom_concatdataset import ConcatDataset
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    # get_dataloader_kwargs,
)
from custom.custom_dataloder_kwargs import get_dataloader_kwargs
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset

from llama_recipes.utils.fsdp_utils import hsdp_device_mesh
from llama_recipes.utils.train_utils import (
    # train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies,
)
from custom.custom_train_utils import train_custom_model

from accelerate.utils import is_xpu_available
from warnings import warn
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
path = "../trialgpt_discover/fairness_results"
corpus = ["sigir", "trec_2022", "trec_2021"]

def process_matching_label(trial):
    included = 0
    not_inc = 0
    na_inc = 0
    no_info_inc = 0
    excluded = 0
    not_exc = 0
    na_exc = 0
    no_info_exc = 0
    res = {"included": 0, "not included": 0, "not applicable": 0, "not enough information": 0, "excluede": 0, "not excluded": 0, "not applicable": 0}
    for key, info in trial["inclusion"].items():
        if info[2] == "included":
            included += 1    
        elif info[2] == "not included":
            not_inc += 1
        elif info[2] == "not applicable":
            na_inc += 1
        elif info[2] == "not enough information":
            no_info_inc += 1
    for info in trial["exclusion"]:
        if len(info) != 3:
            continue
        if info[2] == "excluded":
            excluded += 1    
        elif info[2] == "not excluded":
            not_exc += 1
        elif info[2] == "not applicable":
            na_exc += 1
        elif info[2] == "not enough information":
            no_info_exc += 1
    res = {"included": included, "not included": not_inc, "not applicable": na_inc + na_exc, "not enough information": no_info_exc + no_info_inc, "excluede": excluded, "not excluded": not_exc, "not applicable": na_exc + na_inc}
    return res

def load_data(corpus):
    sample = []
    retrieval_trials = f"../{corpus}/retrieved_trials.json"
    senstive_patient = f"../trialgpt_discover/sensitive_changed_patient_note/{corpus}_all_sensitive.json"
    match_res = f"../trialgpt_results[67]/matching_results_{corpus}_gpt-4-turbo.json"
    agg_res = f"../trialgpt_results[67]/aggregation_results_{corpus}_gpt-4-turbo.json"
    with open(retrieval_trials, "r") as f:
        retrieval_trials = json.load(f)
    with open(senstive_patient, "r") as f:
        senstive_patient = json.load(f)
    assert len(senstive_patient) == len(retrieval_trials)
    with open(agg_res, "r") as f:
        agg = json.load(f)
    with open(match_res, "r") as f:
        match = json.load(f)
    for trials in retrieval_trials[:10]:
        patient_id = trials["patient_id"]
        senstive_queries = senstive_patient[patient_id]
        labels = ["0", "1", "2"]
        for label in labels:
            label_trials = trials[label]
            for trial in label_trials:
                res = {}
                res["query"] = senstive_queries
                res["inclusion_criteria"] = trial["inclusion_criteria"]
                res["exclusion_criteria"] = trial["exclusion_criteria"]
                res["brief_summary"] = trial["brief_summary"]
                trial_id = trial["NCTID"]
                trial_label = label
                res["trial_id"] = trial_id
                res["trial_label"] = trial_label
                relevance_score_R = agg[patient_id][trial_id]["relevance_score_R"]
                eligibility_score_E = agg[patient_id][trial_id]["eligibility_score_E"]
                res["relevance_score_R"] = relevance_score_R
                res["eligibility_score_E"] = eligibility_score_E
                trial_all = match[patient_id][trial_label][trial_id]
                new_label = process_matching_label(trial_all)
                res["new_label"] = new_label
                sample.append(res)
    return sample

samples = load_data("sigir")
tokenizer = AutoTokenizer.from_pretrained("/home/yuj49/llama3_8B")
tokenizer.add_special_tokens({'pad_token': '[PAD]', 'unk_token': '[UNK]'})
unk_token_id = tokenizer.vocab_size - 1  

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        original_queries = list(sample['query'].values())
        original_queries = sample['query']["patient"]
        original_input_text = f"{sample['query']['patient']} {sample['brief_summary']} {sample['inclusion_criteria']} {sample['exclusion_criteria']}"
        
        labels = sample["new_label"]
        labels_str = ' '.join([f"{key}: {value}" for key, value in labels.items()])

        bias_queries = []
        for key, value in sample["query"].items():
            if key not in ["patient", "all"]:
                if value is not None:
                    bias_queries.append(value)
        if original_queries != []:
            original_text_inputs = self.tokenizer(original_input_text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
            original_inputs = self.tokenizer(original_queries, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)

            bias_inputs = self.tokenizer(bias_queries, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
            labels_inputs = self.tokenizer([labels_str], return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)

            original_input_text_ids = original_text_inputs['input_ids'].squeeze(0)
            original_input_ids = original_inputs['input_ids'].squeeze(0)
            
            bias_input_ids = bias_inputs['input_ids']
            # Replace out-of-vocabulary tokens with unk_token_id
            original_input_ids[original_input_ids >= tokenizer.vocab_size] = unk_token_id
            original_input_text_ids[original_input_text_ids >= tokenizer.vocab_size] = unk_token_id
            
            bias_input_ids[bias_input_ids >= tokenizer.vocab_size] = unk_token_id

            return {
                'original_input_ids':original_input_ids, 
                'original_attention_mask':original_inputs['attention_mask'].squeeze(0),
                'input_text_ids': original_input_text_ids,
                'text_attention_mask': original_text_inputs['attention_mask'].squeeze(0),
                'bias_input_ids': bias_input_ids,
                'bias_attention_mask': bias_inputs['attention_mask'],
                'labels': labels_inputs['input_ids'].squeeze(0)
            }
        return None

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:  # Skip if batch is empty
        return None
    
    # Find the maximum number of queries in the batch
    max_num_queries = max([item['bias_input_ids'].size(0) for item in batch])
    max_seq_len = max([item['bias_input_ids'].size(1) for item in batch])
    
    bias_input_ids = []
    bias_attention_mask = []
    labels_list = []
    
    for item in batch:
        num_queries = item['bias_input_ids'].size(0)
        seq_len = item['bias_input_ids'].size(1)
        
        # Pad bias_input_ids and bias_attention_mask to have max_num_queries
        pad_queries = max_num_queries - num_queries
        pad_seq_len = max_seq_len - seq_len

        padded_bias_input_ids = F.pad(item['bias_input_ids'], (0, pad_seq_len, 0, pad_queries), value=tokenizer.pad_token_id)
        padded_bias_attention_mask = F.pad(item['bias_attention_mask'], (0, pad_seq_len, 0, pad_queries), value=0)

        bias_input_ids.append(padded_bias_input_ids)
        bias_attention_mask.append(padded_bias_attention_mask)
        
        labels_list.append(item["labels"])
    
    try:
        return {
            'input_text_ids': torch.stack([item['input_text_ids'] for item in batch]),
            'text_attention_mask': torch.stack([item['text_attention_mask'] for item in batch]),
            
            'original_input_ids': torch.stack([item['original_input_ids'] for item in batch]),
            'original_attention_mask': torch.stack([item['original_attention_mask'] for item in batch]),
            
            'bias_input_ids': torch.stack(bias_input_ids),
            'bias_attention_mask': torch.stack(bias_attention_mask),
            'labels': torch.stack(labels_list)  # Corrected to stack the list of labels tensors
        }
    except RuntimeError as e:
        print("Error stacking tensors:", e)
        for i, item in enumerate(batch):
            print(f"Item {i} - original_input_ids size: {item['original_input_ids'].size()}")
            print(f"Item {i} - bias_input_ids size: {item['bias_input_ids'].size()}")
        raise e
def get_dataloader(dataset, batch_size, local_rank):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    return dataloader
dataset = CustomDataset(samples, tokenizer)