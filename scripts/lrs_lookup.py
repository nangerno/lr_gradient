from typing import Optional
from lr_finder import find_lr  # find_lr returns Optional[dict] with keys "lr" and "batch_size"

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
) -> Optional[dict]:
    if not dataset_path:
        return None
    return find_lr(
        model_id, model_path, num_params,
        dataset_path, dataset_type_dict,
        train_type="instruct",
        min_lr=5e-6,
        max_lr=3e-3,
        steps=40,
        seq_len=1024,   # instruct trains on prompt+response, typically up to 2048 tokens;
                        # 1024 is a realistic probe length that avoids 4× OOM mismatch
        lora_threshold=None,
    )


def get_dpo_lr(
    model_id: str,
    model_path: str,
    num_params: Optional[int],
    dataset_path: str,
    dataset_type_dict: dict,
) -> Optional[dict]:
    if not dataset_path:
        return None
    return find_lr(
        model_id, model_path, num_params,
        dataset_path, dataset_type_dict,
        train_type="dpo",
        min_lr=5e-7,
        max_lr=5e-5,
        seq_len=512,    # DPO uses LoRA and trains on chosen/rejected pairs;
                        # 512 is safe given the halved start_batch for DPO
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
) -> Optional[dict]:
    if not dataset_path:
        return None
    return find_lr(
        model_id, model_path, num_params,
        dataset_path, dataset_type_dict,
        train_type="grpo",
        min_lr=2e-6,
        max_lr=1.5e-5,
        seq_len=512,    # GRPO only processes prompts; max_prompt_length=512
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
) -> Optional[dict]:
    if not dataset_path:
        return None
    return find_lr(
        model_id, model_path, num_params,
        dataset_path, dataset_type_dict,
        train_type="grpo",
        min_lr=2e-6,
        max_lr=1.5e-5,
        seq_len=512,    # GRPO only processes prompts; max_prompt_length=512
        lora_threshold=_DPO_GRPO_LORA_THRESHOLD,
        lora_r=_DPO_GRPO_LORA_R,
        lora_alpha=_DPO_GRPO_LORA_ALPHA,
        lora_dropout=_DPO_GRPO_LORA_DROPOUT,
    )
