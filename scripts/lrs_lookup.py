from typing import Optional
from lr_finder import find_lr

_DPO_GRPO_LORA_THRESHOLD = 2_000_000_000
_DPO_GRPO_LORA_R         = 128
_DPO_GRPO_LORA_ALPHA     = 256
_DPO_GRPO_LORA_DROPOUT   = 0.05


def get_instruct_lr(
    model_id: str,
    model_path: str,
    num_params: Optional[int],
    dataset_path: str,
    dataset_type_dict: dict,
) -> Optional[float]:
    if not dataset_path:
        return None
    return find_lr(
        model_id, model_path, num_params,
        dataset_path, dataset_type_dict,
        train_type="instruct",
        min_lr=3e-6,
        max_lr=2.35e-3,
        steps=40,                   # more steps → less noisy loss curve on small datasets
        lora_threshold=None,        # instruct trains full-weight — no LoRA in probe
    )


def get_dpo_lr(
    model_id: str,
    model_path: str,
    num_params: Optional[int],
    dataset_path: str,
    dataset_type_dict: dict,
) -> Optional[float]:
    if not dataset_path:
        return None
    return find_lr(
        model_id, model_path, num_params,
        dataset_path, dataset_type_dict,
        train_type="dpo",
        min_lr=5e-7,
        max_lr=1.70e-4,
        lora_threshold=_DPO_GRPO_LORA_THRESHOLD,
        lora_r=_DPO_GRPO_LORA_R,
        lora_alpha=_DPO_GRPO_LORA_ALPHA,
        lora_dropout=_DPO_GRPO_LORA_DROPOUT,
    )


def get_grpo_lr(
    model_id: str,
    model_path: str,
    num_params: Optional[int],
    dataset_path: str,
    dataset_type_dict: dict,
) -> Optional[float]:
    if not dataset_path:
        return None
    return find_lr(
        model_id, model_path, num_params,
        dataset_path, dataset_type_dict,
        train_type="grpo",
        min_lr=1e-5,
        max_lr=8.64e-4,
        lora_threshold=_DPO_GRPO_LORA_THRESHOLD,
        lora_r=_DPO_GRPO_LORA_R,
        lora_alpha=_DPO_GRPO_LORA_ALPHA,
        lora_dropout=_DPO_GRPO_LORA_DROPOUT,
    )


def get_grpo_python_lr(
    model_id: str,
    model_path: str,
    num_params: Optional[int],
    dataset_path: str,
    dataset_type_dict: dict,
) -> Optional[float]:
    if not dataset_path:
        return None
    return find_lr(
        model_id, model_path, num_params,
        dataset_path, dataset_type_dict,
        train_type="grpo",
        min_lr=3e-6,
        max_lr=1.60e-3,
        lora_threshold=_DPO_GRPO_LORA_THRESHOLD,
        lora_r=_DPO_GRPO_LORA_R,
        lora_alpha=_DPO_GRPO_LORA_ALPHA,
        lora_dropout=_DPO_GRPO_LORA_DROPOUT,
    )
