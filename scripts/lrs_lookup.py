import math
import os
from pathlib import Path
from typing import Optional

from lr_finder import find_lr
from lr_finder_tasks import LR_TASK_DPO, LR_TASK_GRPO, normalize_lr_finder_task

# When ``train_info`` omits ``max_length`` / ``lr_finder_seq_len``, match ``train_instruct``
# default (``test_axolotl.yml`` ``sequence_len``) so LR probes see the same max length
# as training (e.g. 2048), not a shorter 1024 window.
_DEFAULT_LR_FINDER_SEQ_FALLBACK = 2048


def _sequence_len_from_test_axolotl_yml() -> Optional[int]:
    """Same default path as ``train_instruct.get_max_length_config`` (``scripts/test_axolotl.yml``)."""
    try:
        import yaml
    except ImportError:
        return None
    cfg = Path(__file__).resolve().parent / "test_axolotl.yml"
    if not cfg.is_file():
        return None
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return None
        sl = data.get("sequence_len")
        if sl is None:
            return None
        return max(1, int(sl))
    except (OSError, TypeError, ValueError):
        return None
    except yaml.YAMLError:
        return None


def effective_lr_finder_seq_len(train_info: dict, default: int = _DEFAULT_LR_FINDER_SEQ_FALLBACK) -> int:
    if train_info.get("lr_finder_seq_len") is not None:
        return max(1, int(train_info["lr_finder_seq_len"]))
    ml = train_info.get("max_length")
    try:
        ml_int = int(ml)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        ml_int = -1
    if ml_int > 0:
        return ml_int
    from_yaml = _sequence_len_from_test_axolotl_yml()
    if from_yaml is not None:
        return from_yaml
    return max(1, int(default))


def effective_lr_finder_probe_seq_len(train_info: dict) -> int:
    """
    Sequence length used to pad/collate LR mini-train batches.

    Defaults to the same value as ``effective_lr_finder_seq_len`` (``train_info``
    ``max_length`` / ``lr_finder_seq_len``, else ``test_axolotl.yml`` ``sequence_len``,
    else 2048) so the probe matches real training length. Set
    ``lr_finder_probe_seq_len`` explicitly for a shorter, cheaper probe.
    """
    raw = train_info.get("lr_finder_probe_seq_len")
    if raw is None:
        return effective_lr_finder_seq_len(train_info)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return effective_lr_finder_seq_len(train_info)


def resolve_tokenized_train_path(train_info: dict) -> Optional[str]:
    """Resolve ``train_tokenized_*.json``: explicit path, or ``datasets/train_tokenized_{task_id}.json``."""
    tokenized_train_path = train_info.get("tokenized_train_path")
    if not tokenized_train_path and train_info.get("task_id"):
        candidate = f"datasets/train_tokenized_{train_info['task_id']}.json"
        tokenized_train_path = candidate if os.path.isfile(candidate) else None
    elif tokenized_train_path and not os.path.isfile(tokenized_train_path):
        tokenized_train_path = None
    return tokenized_train_path or None


def resolve_lr_finder_dataset_path(train_info: dict, run_config: dict) -> Optional[str]:
    """
    Path used by ``find_lr`` for mini-train: instruct → ``train_tokenized_*.json``;
    DPO → ``dpo_train_*.json``; GRPO → ``grpo_train_*.json``.
    """
    task_id = train_info.get("task_id") or ""
    task = normalize_lr_finder_task(run_config.get("lr_finder_task", "instruct"))
    if task == LR_TASK_DPO:
        candidate = os.path.join("datasets", f"dpo_train_{task_id}.json")
    elif task == LR_TASK_GRPO:
        candidate = os.path.join("datasets", f"grpo_train_{task_id}.json")
    else:
        return resolve_tokenized_train_path(train_info)
    return candidate if os.path.isfile(candidate) else None


def is_lr_finder_runnable(train_info: dict, run_config: dict) -> bool:
    return resolve_lr_finder_dataset_path(train_info, run_config) is not None


def apply_tokenized_lr_finder_to_run_config(
    run_config: dict,
    train_info: dict,
    model_id: str,
    model_path: str,
    num_params: Optional[int],
    *,
    fallback_learning_rate: float,
) -> None:
    dataset_type_dict = dict(train_info.get("dataset_type", {}))
    tokenized_train_path = resolve_tokenized_train_path(train_info)
    if tokenized_train_path:
        dataset_type_dict["tokenized_train_path"] = tokenized_train_path

    # ``run_config`` is the canonical training JSON; mirror all ``lr_finder_*`` keys.
    for rk, rv in run_config.items():
        if rk.startswith("lr_finder_") and rv is not None:
            dataset_type_dict[rk] = rv

    dataset_type_dict["lr_finder_use_lora"] = bool(run_config.get("use_lora", False))
    try:
        if run_config.get("max_grad_norm") is not None:
            dataset_type_dict["lr_finder_max_grad_norm"] = float(run_config["max_grad_norm"])
    except (TypeError, ValueError):
        pass
    if "lr_finder_gradient_checkpointing" not in dataset_type_dict:
        dataset_type_dict["lr_finder_gradient_checkpointing"] = bool(
            run_config.get("gradient_checkpointing", True)
        )

    _lr_seq = int(run_config.get("lr_finder_seq_len", _DEFAULT_LR_FINDER_SEQ_FALLBACK))
    if "lr_finder_probe_seq_len" not in dataset_type_dict:
        dataset_type_dict["lr_finder_probe_seq_len"] = int(
            run_config.get("lr_finder_probe_seq_len", _lr_seq)
        )

    _task = normalize_lr_finder_task(run_config.get("lr_finder_task", "instruct"))
    if _task == LR_TASK_GRPO and run_config.get("max_completion_length") is not None:
        dataset_type_dict["max_completion_length"] = int(run_config["max_completion_length"])
    if _task == LR_TASK_DPO and run_config.get("beta") is not None:
        try:
            dataset_type_dict["lr_finder_dpo_beta"] = float(run_config["beta"])
        except (TypeError, ValueError):
            pass

    lr_dataset_path = resolve_lr_finder_dataset_path(train_info, run_config)
    if not is_lr_finder_runnable(train_info, run_config):
        print(
            "[LR Finder] Skipping: no dataset file for this task (see lr_finder_task / tokenize output); "
            "using param-based LR and batch.",
            flush=True,
        )
        return

    _planned_bs: Optional[int] = None
    _raw_bs = run_config.get("batch_size")
    try:
        _planned_bs = int(_raw_bs) if _raw_bs is not None else None
    except (TypeError, ValueError):
        _planned_bs = None
    if _planned_bs is not None and _planned_bs < 1:
        _planned_bs = None

    try:
        dataset_type_dict["lr_finder_gradient_accumulation_steps"] = max(
            1, int(run_config.get("gradient_accumulation_steps", 1))
        )
    except (TypeError, ValueError):
        dataset_type_dict["lr_finder_gradient_accumulation_steps"] = 1
    try:
        dataset_type_dict["lr_finder_planned_gpu_nums"] = max(1, int(run_config.get("gpu_nums", 1)))
    except (TypeError, ValueError):
        dataset_type_dict["lr_finder_planned_gpu_nums"] = 1
    if _planned_bs is not None:
        dataset_type_dict["lr_finder_planned_per_device_batch_size"] = int(_planned_bs)
    dataset_type_dict["lr_finder_linear_scale_lr_to_effective_batch"] = bool(
        run_config.get("lr_finder_linear_scale_lr_to_effective_batch", False)
    )

    try:
        _lr_min = float(run_config.get("lr_finder_min_lr", 1e-6))
    except (TypeError, ValueError):
        _lr_min = 1e-6
    try:
        _lr_max = float(run_config.get("lr_finder_max_lr", 9e-3))
    except (TypeError, ValueError):
        _lr_max = 9e-3

    lr_result = get_instruct_lr(
        model_id,
        model_path,
        num_params,
        dataset_type_dict,
        tokenized_dataset_path=lr_dataset_path,
        seq_len=run_config["lr_finder_seq_len"],
        lr_probe_points=run_config["lr_finder_lr_probe_points"],
        mini_train_batches=run_config["lr_finder_mini_train_batches"],
        optimizer_name=run_config["optimizer"],
        lr_sample_seed=run_config["lr_finder_sample_seed"],
        batch_headroom=run_config["lr_finder_batch_headroom"],
        max_lr_probe_batch=_planned_bs,
        min_lr=_lr_min,
        max_lr=_lr_max,
    )

    if lr_result is not None:
        lr = lr_result.get("lr")
        bs = lr_result.get("batch_size")

        if lr is not None:
            print(f"Using lr from finder (task={_task} mini-train dataset): {lr}", flush=True)
            run_config["learning_rate"] = lr
            run_config["learning_rate_set_by_lr_finder"] = True
        else:
            print(
                f"LR finder returned no lr; using param-based fallback: {fallback_learning_rate}",
                flush=True,
            )
            run_config["learning_rate"] = fallback_learning_rate

        if bs is not None:
            print(f"Using batch size from finder (safety batch): {bs}", flush=True)
            run_config["batch_size"] = bs
    else:
        print(
            "[LR Finder] Probe failed; using param-based learning rate: "
            f"{run_config['learning_rate']}",
            flush=True,
        )


def apply_training_lr_scaling_after_lr_finder(run_config: dict, train_info: dict) -> None:
    """
    After ``apply_tokenized_lr_finder_to_run_config``:

    * If the finder set ``learning_rate``, **do not** multiply by ``reg_ratio`` — that
      value was chosen for conservative param-based LRs and silently shrinks a probed LR.
      Optional ``reg_ratio_if_lr_finder`` (default ``1.0``) scales the finder LR if you
      still want dampening.
    * Unless ``lr_finder_keep_warmup`` is true, set ``warmup_ratio`` to
      ``lr_finder_warmup_ratio`` (default ``0``) so the first optimizer step uses the
      finder peak instead of linear ramp from ~0 (HF ``warmup_ratio`` behavior).
    """
    if not run_config.pop("learning_rate_set_by_lr_finder", False):
        run_config["learning_rate"] *= float(train_info["reg_ratio"])
        return
    rr = float(train_info.get("reg_ratio_if_lr_finder", 1.0))
    if math.isfinite(rr) and rr > 0:
        run_config["learning_rate"] *= rr
    if "lr_finder_warmup_ratio" in train_info:
        run_config["warmup_ratio"] = float(train_info["lr_finder_warmup_ratio"])
    elif not bool(train_info.get("lr_finder_keep_warmup", False)):
        prev = float(run_config.get("warmup_ratio", 0.0) or 0.0)
        if prev > 0.0:
            print(
                "[LR Finder] warmup_ratio set to 0 so training uses the finder LR from step 1 "
                "(HF still applies cosine decay). "
                "Set lr_finder_keep_warmup=True or lr_finder_warmup_ratio=<float> to keep warmup.",
                flush=True,
            )
        run_config["warmup_ratio"] = float(train_info.get("lr_finder_warmup_ratio", 0.0))


def is_dataset_available_for_lr_finder(dataset_path: Optional[str]) -> bool:
    """Empty/missing path skips LR finder (not an error). Callers print a distinct message."""
    return bool(dataset_path and str(dataset_path).strip())


def get_instruct_lr(
    model_id: str,
    model_path: str,
    num_params: Optional[int],
    dataset_type_dict: dict,
    *,
    tokenized_dataset_path: str,
    seq_len: int = _DEFAULT_LR_FINDER_SEQ_FALLBACK,
    lr_probe_points: int = 28,
    mini_train_batches: int = 20,
    optimizer_name: Optional[str] = None,
    lr_sample_seed: int = 42,
    batch_headroom: float = 0.8,
    max_lr_probe_batch: Optional[int] = None,
    min_lr: float = 1e-6,
    max_lr: float = 9e-3,
) -> Optional[dict]:
    if not tokenized_dataset_path or not os.path.isfile(tokenized_dataset_path):
        return None
    return find_lr(
        model_id,
        model_path,
        num_params,
        tokenized_dataset_path,
        dataset_type_dict,
        min_lr=min_lr,
        max_lr=max_lr,
        lr_probe_points=lr_probe_points,
        mini_train_batches=mini_train_batches,
        seq_len=seq_len,
        optimizer_name=optimizer_name,
        lr_sample_seed=lr_sample_seed,
        batch_headroom=batch_headroom,
        max_lr_probe_batch=max_lr_probe_batch,
    )
