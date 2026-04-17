from typing import Any, Dict
import yaml
from torch.utils.data import Dataset
from pathlib import Path
from transformers import AutoTokenizer
from typing import Callable
import torch
import logging
from datetime import datetime
import sys
import wandb
import random
import json
import requests
import os
import shutil
from transformers.trainer_utils import is_main_process

logger = logging.getLogger()
logger.setLevel(logging.INFO)
 # Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
 # Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))



def log_info(message: str, event_name: str = "print"):
    if is_main_process(LOCAL_RANK):
        logger.info(f"{event_name}: {message}")
    # wandb.log({"event": event_name, "message": message})


def apply_save_total_limit(training_args: Any, train_request: Dict[str, Any]) -> None:
    """Cap retained checkpoints to limit disk use (override via ``save_total_limit`` in request)."""
    if train_request.get("save_total_limit") is not None:
        training_args.save_total_limit = int(train_request["save_total_limit"])
    elif getattr(training_args, "save_total_limit", None) in (None, 0):
        training_args.save_total_limit = 3
        log_info(
            "save_total_limit=3 (limits checkpoint folders on disk; set train_request['save_total_limit'] to override)"
        )


def ensure_positive_steps_per_epoch(training_args: Any, effective_sample_count: int) -> int:
    """
    If ``steps_per_epoch`` would be 0, shrink ``per_device_train_batch_size`` so one
    global step can consume at most ``effective_sample_count`` samples (instruct/SFT:
    use ``len(train_ds)``; GRPO: use ``len(train_ds) * num_generations``).
    """
    g = int(getattr(training_args, "gradient_accumulation_steps", 1) or 1) * int(
        getattr(training_args, "world_size", 1) or 1
    )
    if effective_sample_count <= 0 or g <= 0:
        return 0
    pd = int(getattr(training_args, "per_device_train_batch_size", 1) or 1)
    denom = pd * g
    steps = effective_sample_count // denom
    if steps > 0:
        return steps
    max_pd = max(1, effective_sample_count // g)
    if max_pd < pd:
        log_info(
            f"steps_per_epoch would be 0 with per_device_train_batch_size={pd}; "
            f"reducing to {max_pd} so at least one optimizer step fits "
            f"(effective_samples={effective_sample_count}, grad_accum×world={g})."
        )
        training_args.per_device_train_batch_size = max_pd
    denom = int(training_args.per_device_train_batch_size) * g
    steps = effective_sample_count // denom
    if steps == 0:
        log_info(
            f"steps_per_epoch still 0: effective_samples={effective_sample_count} < "
            f"global batch ({training_args.per_device_train_batch_size}×{g}={denom}). "
            "Reduce world_size, gradient_accumulation_steps, or increase data."
        )
    return steps


def pad_sequence(sequence: list[int], pad_value: int, max_length: int, padding_side: str) -> list[int]:
    if padding_side == "left":
        return [pad_value] * (max_length - len(sequence)) + sequence
    else:
        return sequence + [pad_value] * (max_length - len(sequence))


def pad_inputs(tokenizer: AutoTokenizer, input_dict: dict, max_length: int, padding_side: str) -> dict:
    assert padding_side in ["left", "right"]
    result = {
        "input_ids": pad_sequence(input_dict["input_ids"], tokenizer.pad_token_id, max_length, padding_side),
        "attention_mask": pad_sequence(input_dict["attention_mask"], 0, max_length, padding_side),
        "labels": pad_sequence(input_dict["labels"], -100, max_length, padding_side),
    }
    return result


class MyDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, data_path: str, max_length: int) -> None:
        super().__init__()
        with open(data_path, 'r') as file:
            self.eval_dataset = json.load(file)
            
        self.tokenizer = tokenizer
        self.max_length = max_length
        print("padding_side: ", self.tokenizer.padding_side)
        
    def __len__(self):
        return len(self.eval_dataset)
    
    def __getitem__(self, idx):
        dp = self.eval_dataset[idx]
        input_dict = pad_inputs(self.tokenizer, dp, self.max_length, self.tokenizer.padding_side)
        for key in input_dict:
            input_dict[key] = torch.tensor(input_dict[key])
        return input_dict