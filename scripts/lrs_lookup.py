import os
from typing import Optional

from lr_finder import find_lr


def effective_lr_finder_seq_len(train_info: dict, default: int = 1024) -> int:
    if train_info.get("lr_finder_seq_len") is not None:
        return max(1, int(train_info["lr_finder_seq_len"]))
    ml = train_info.get("max_length")
    try:
        ml_int = int(ml)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        ml_int = -1
    if ml_int > 0:
        return ml_int
    return default


def resolve_tokenized_train_path(train_info: dict) -> Optional[str]:
    """Resolve ``train_tokenized_*.json``: explicit path, or ``datasets/train_tokenized_{task_id}.json``."""
    tokenized_train_path = train_info.get("tokenized_train_path")
    if not tokenized_train_path and train_info.get("task_id"):
        candidate = f"datasets/train_tokenized_{train_info['task_id']}.json"
        tokenized_train_path = candidate if os.path.isfile(candidate) else None
    elif tokenized_train_path and not os.path.isfile(tokenized_train_path):
        tokenized_train_path = None
    return tokenized_train_path or None


def apply_tokenized_lr_finder_to_run_config(
    run_config: dict,
    train_info: dict,
    model_id: str,
    model_path: str,
    num_params: Optional[int],
    *,
    fallback_learning_rate: float,
    grpo_slow_reward_proxy_probe: bool = False,
) -> None:
    dataset_type_dict = dict(train_info.get("dataset_type", {}))
    tokenized_train_path = resolve_tokenized_train_path(train_info)
    if tokenized_train_path:
        dataset_type_dict["tokenized_train_path"] = tokenized_train_path

    dataset_type_dict["lr_finder_peak_rel_slack"] = float(
        run_config.get("lr_finder_peak_rel_slack", 0.28)
    )

    if not is_instruct_lr_finder_runnable(tokenized_train_path):
        print(
            "[LR Finder] Skipping: no train_tokenized JSON; using param-based LR and batch.",
            flush=True,
        )
        return

    lr_result = get_instruct_lr(
        model_id,
        model_path,
        num_params,
        dataset_type_dict,
        tokenized_dataset_path=tokenized_train_path,
        seq_len=run_config["lr_finder_seq_len"],
        lr_probe_points=run_config["lr_finder_lr_probe_points"],
        mini_train_batches=run_config["lr_finder_mini_train_batches"],
        optimizer_name=run_config["optimizer"],
        lr_sample_stratify=run_config["lr_finder_stratify_length"],
        lr_sample_seed=run_config["lr_finder_sample_seed"],
        batch_headroom=run_config["lr_finder_batch_headroom"],
        grpo_slow_reward_proxy_probe=grpo_slow_reward_proxy_probe,
    )

    if lr_result is not None:
        lr = lr_result.get("lr")
        bs = lr_result.get("batch_size")

        if lr is not None:
            print(f"Using lr from finder (2% tokenized probe): {lr}", flush=True)
            run_config["learning_rate"] = lr
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


def is_dataset_available_for_lr_finder(dataset_path: Optional[str]) -> bool:
    """Empty/missing path skips LR finder (not an error). Callers print a distinct message."""
    return bool(dataset_path and str(dataset_path).strip())


def is_instruct_lr_finder_runnable(tokenized_train_path: Optional[str]) -> bool:
    """LR finder runs only on ``train_tokenized_*.json`` (2% sample + safety batch)."""
    return bool(tokenized_train_path and os.path.isfile(tokenized_train_path))


def get_instruct_lr(
    model_id: str,
    model_path: str,
    num_params: Optional[int],
    dataset_type_dict: dict,
    *,
    tokenized_dataset_path: str,
    seq_len: int = 1024,
    lr_probe_points: int = 28,
    mini_train_batches: int = 3,
    optimizer_name: Optional[str] = None,
    lr_sample_stratify: bool = True,
    lr_sample_seed: int = 42,
    batch_headroom: float = 0.8,
    grpo_slow_reward_proxy_probe: bool = False,
) -> Optional[dict]:
    if not tokenized_dataset_path or not os.path.isfile(tokenized_dataset_path):
        return None
    return find_lr(
        model_id,
        model_path,
        num_params,
        tokenized_dataset_path,
        dataset_type_dict,
        min_lr=1e-6,
        max_lr=9e-3,
        lr_probe_points=lr_probe_points,
        mini_train_batches=mini_train_batches,
        seq_len=seq_len,
        optimizer_name=optimizer_name,
        lr_sample_stratify=lr_sample_stratify,
        lr_sample_seed=lr_sample_seed,
        batch_headroom=batch_headroom,
        grpo_slow_reward_proxy_probe=grpo_slow_reward_proxy_probe,
    )
