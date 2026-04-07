from typing import Optional
from lr_finder import find_lr  # find_lr returns Optional[dict] with keys "lr" and "batch_size"

_DPO_GRPO_LORA_THRESHOLD = 2_000_000_000
_DPO_GRPO_LORA_R         = 128
_DPO_GRPO_LORA_ALPHA     = 256
_DPO_GRPO_LORA_DROPOUT   = 0.05

_GRPO_PROBE_SEQ_CAP = 8192


def _grpo_probe_seq_len(max_prompt_length: int, max_completion_length: int) -> int:
    """Upper-bound token length for LR probe (aligns batch + micro-train with GRPO training)."""
    total = max(1, int(max_prompt_length) + int(max_completion_length))
    return min(total, _GRPO_PROBE_SEQ_CAP)


def get_instruct_lr(
    model_id: str,
    model_path: str,
    num_params: Optional[int],
    dataset_path: str,
    dataset_type_dict: dict,
    *,
    seq_len: int = 1024,
    steps: int = 40,
    lr_points: int = 35,
    optimizer_name: Optional[str] = None,
) -> Optional[dict]:
    if not dataset_path:
        return None
    return find_lr(
        model_id, model_path, num_params,
        dataset_path, dataset_type_dict,
        train_type="instruct",
        min_lr=1e-6,
        max_lr=9e-3,
        seq_len=seq_len,
        steps=steps,
        lr_points=lr_points,
        optimizer_name=optimizer_name,
        lora_threshold=None,
    )


def get_dpo_lr(
    model_id: str,
    model_path: str,
    num_params: Optional[int],
    dataset_path: str,
    dataset_type_dict: dict,
    *,
    seq_len: int = 512,
    steps: int = 40,
    lr_points: int = 35,
    optimizer_name: Optional[str] = None,
) -> Optional[dict]:
    if not dataset_path:
        return None
    return find_lr(
        model_id, model_path, num_params,
        dataset_path, dataset_type_dict,
        train_type="dpo",
        min_lr=1e-7,
        max_lr=9e-4,
        seq_len=seq_len,
        steps=steps,
        lr_points=lr_points,
        optimizer_name=optimizer_name,
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
    *,
    max_prompt_length: int = 512,
    max_completion_length: int = 512,
    steps: int = 40,
    lr_points: int = 35,
    optimizer_name: Optional[str] = None,
) -> Optional[dict]:
    if not dataset_path:
        return None
    seq_len = _grpo_probe_seq_len(max_prompt_length, max_completion_length)
    print(
        f"[LR Finder] GRPO probe seq_len={seq_len} "
        f"(max_prompt={max_prompt_length} + max_completion={max_completion_length})",
        flush=True,
    )
    return find_lr(
        model_id, model_path, num_params,
        dataset_path, dataset_type_dict,
        train_type="grpo",
        min_lr=1e-5,
        max_lr=9e-4,
        seq_len=seq_len,
        steps=steps,
        lr_points=lr_points,
        optimizer_name=optimizer_name,
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
    *,
    max_prompt_length: int = 512,
    max_completion_length: int = 512,
    steps: int = 40,
    lr_points: int = 30,
    optimizer_name: Optional[str] = None,
) -> Optional[dict]:
    if not dataset_path:
        return None
    seq_len = _grpo_probe_seq_len(max_prompt_length, max_completion_length)
    print(
        f"[LR Finder] GRPO probe seq_len={seq_len} "
        f"(max_prompt={max_prompt_length} + max_completion={max_completion_length})",
        flush=True,
    )
    return find_lr(
        model_id, model_path, num_params,
        dataset_path, dataset_type_dict,
        train_type="grpo",
        min_lr=1e-6,
        max_lr=9e-3,
        seq_len=seq_len,
        steps=steps,
        lr_points=lr_points,
        optimizer_name=optimizer_name,
        lora_threshold=_DPO_GRPO_LORA_THRESHOLD,
        lora_r=_DPO_GRPO_LORA_R,
        lora_alpha=_DPO_GRPO_LORA_ALPHA,
        lora_dropout=_DPO_GRPO_LORA_DROPOUT,
    )
