from typing import Literal, Optional

from lr_finder import find_lr

SmithCurveMode = Literal["steepest", "max_decreasing"]

_DPO_GRPO_LORA_THRESHOLD = 2_000_000_000


def is_dataset_available_for_lr_finder(dataset_path: Optional[str]) -> bool:
    """Empty/missing path skips LR finder (not an error). Callers print a distinct message."""
    return bool(dataset_path and str(dataset_path).strip())


_DPO_GRPO_LORA_R = 128
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
    smith_safety_divisor: Optional[float] = None,
    smith_micro_batches: int = 1,
    smith_early_stop_divergence: bool = True,
    smith_divergence_vs_min: float = 10.0,
    smith_min_points_before_divergence: int = 5,
    lr_sample_frac: float = 0.02,
    lr_sample_min: int = 200,
    lr_sample_max: int = 3000,
    lr_sample_stratify: bool = True,
    lr_sample_seed: int = 42,
    batch_headroom: float = 0.8,
    smith_curve_mode: SmithCurveMode = "max_decreasing",
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
        smith_safety_divisor=smith_safety_divisor,
        smith_micro_batches=smith_micro_batches,
        smith_early_stop_divergence=smith_early_stop_divergence,
        smith_divergence_vs_min=smith_divergence_vs_min,
        smith_min_points_before_divergence=smith_min_points_before_divergence,
        lr_sample_frac=lr_sample_frac,
        lr_sample_min=lr_sample_min,
        lr_sample_max=lr_sample_max,
        lr_sample_stratify=lr_sample_stratify,
        lr_sample_seed=lr_sample_seed,
        batch_headroom=batch_headroom,
        smith_curve_mode=smith_curve_mode,
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
    smith_safety_divisor: Optional[float] = None,
    smith_micro_batches: int = 1,
    smith_early_stop_divergence: bool = True,
    smith_divergence_vs_min: float = 10.0,
    smith_min_points_before_divergence: int = 5,
    lr_sample_frac: float = 0.02,
    lr_sample_min: int = 200,
    lr_sample_max: int = 3000,
    lr_sample_stratify: bool = True,
    lr_sample_seed: int = 42,
    batch_headroom: float = 0.8,
    smith_curve_mode: SmithCurveMode = "max_decreasing",
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
        smith_safety_divisor=smith_safety_divisor,
        smith_micro_batches=smith_micro_batches,
        smith_early_stop_divergence=smith_early_stop_divergence,
        smith_divergence_vs_min=smith_divergence_vs_min,
        smith_min_points_before_divergence=smith_min_points_before_divergence,
        lr_sample_frac=lr_sample_frac,
        lr_sample_min=lr_sample_min,
        lr_sample_max=lr_sample_max,
        lr_sample_stratify=lr_sample_stratify,
        lr_sample_seed=lr_sample_seed,
        batch_headroom=batch_headroom,
        smith_curve_mode=smith_curve_mode,
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
    smith_safety_divisor: Optional[float] = None,
    smith_micro_batches: int = 1,
    smith_early_stop_divergence: bool = True,
    smith_divergence_vs_min: float = 10.0,
    smith_min_points_before_divergence: int = 5,
    lr_sample_frac: float = 0.02,
    lr_sample_min: int = 200,
    lr_sample_max: int = 3000,
    lr_sample_stratify: bool = True,
    lr_sample_seed: int = 42,
    batch_headroom: float = 0.8,
    smith_curve_mode: SmithCurveMode = "max_decreasing",
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
        smith_safety_divisor=smith_safety_divisor,
        smith_micro_batches=smith_micro_batches,
        smith_early_stop_divergence=smith_early_stop_divergence,
        smith_divergence_vs_min=smith_divergence_vs_min,
        smith_min_points_before_divergence=smith_min_points_before_divergence,
        lr_sample_frac=lr_sample_frac,
        lr_sample_min=lr_sample_min,
        lr_sample_max=lr_sample_max,
        lr_sample_stratify=lr_sample_stratify,
        lr_sample_seed=lr_sample_seed,
        batch_headroom=batch_headroom,
        smith_curve_mode=smith_curve_mode,
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
    smith_safety_divisor: Optional[float] = None,
    smith_micro_batches: int = 1,
    smith_early_stop_divergence: bool = True,
    smith_divergence_vs_min: float = 10.0,
    smith_min_points_before_divergence: int = 5,
    lr_sample_frac: float = 0.02,
    lr_sample_min: int = 200,
    lr_sample_max: int = 3000,
    lr_sample_stratify: bool = True,
    lr_sample_seed: int = 42,
    batch_headroom: float = 0.8,
    smith_curve_mode: SmithCurveMode = "max_decreasing",
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
        smith_safety_divisor=smith_safety_divisor,
        smith_micro_batches=smith_micro_batches,
        smith_early_stop_divergence=smith_early_stop_divergence,
        smith_divergence_vs_min=smith_divergence_vs_min,
        smith_min_points_before_divergence=smith_min_points_before_divergence,
        lr_sample_frac=lr_sample_frac,
        lr_sample_min=lr_sample_min,
        lr_sample_max=lr_sample_max,
        lr_sample_stratify=lr_sample_stratify,
        lr_sample_seed=lr_sample_seed,
        batch_headroom=batch_headroom,
        smith_curve_mode=smith_curve_mode,
    )
