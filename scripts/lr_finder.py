import contextlib
import gc
import json
import math
import os
import random
from typing import Callable, Optional

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from utility import pad_inputs

# Instruct: probe uses the same padded token tensors as ``MyDataset`` / ``train_instruct``.
TOKENIZED_BATCH_KEY = "__lr_finder_tokenized__"

try:
    import bitsandbytes as bnb
    _HAS_BNB = True
except ImportError:
    _HAS_BNB = False

# Defaults used when callers do not override (kept for backwards-compat only).
_DEFAULT_LORA_THRESHOLD = 4_000_000_000
_DEFAULT_LORA_R = 8
_DEFAULT_LORA_ALPHA = 16
_DEFAULT_LORA_DROPOUT = 0.05

# --------------------------------------------------------------------------- #
# LoRA helpers
# --------------------------------------------------------------------------- #

def _find_all_linear_names(model) -> list[str]:
    names: set[str] = set()
    for name, module in model.named_modules():
        is_linear = isinstance(module, torch.nn.Linear)
        if _HAS_BNB:
            is_linear = is_linear or isinstance(module, bnb.nn.Linear4bit)
        if is_linear:
            parts = name.split(".")
            names.add(parts[0] if len(parts) == 1 else parts[-1])
    names.discard("lm_head")
    return list(names)


def _bf16_autocast(device: str):
    """bf16 autocast on CUDA; no-op on CPU (avoids deprecated torch.cuda.amp.autocast)."""
    if device == "cuda" and torch.cuda.is_available():
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# --------------------------------------------------------------------------- #
# Auto batch-size finder
# --------------------------------------------------------------------------- #

def _can_run(model, tokenizer, batch_size: int, seq_len: int, device: str) -> bool:
    try:
        vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
        if vocab_size <= 0:
            vocab_size = int(len(tokenizer))
        ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        with _bf16_autocast(device):
            out = model(ids, labels=ids)
        out.loss.backward()
        model.zero_grad()
        torch.cuda.empty_cache()
        return True
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            return False
        raise


def find_max_batch_size(
    model,
    tokenizer,
    seq_len: int = 512,
    device: str = "cuda",
    *,
    headroom: float = 0.8,
) -> tuple[int, int]:
    if not _can_run(model, tokenizer, 1, seq_len, device):
        # Avoid low=0 in binary search; real-data path in LR finder retries with smaller batch.
        return 1, 1

    batch = 1
    while _can_run(model, tokenizer, batch, seq_len, device):
        batch *= 2

    low, high, best = batch // 2, batch, batch // 2
    while low <= high:
        mid = (low + high) // 2
        if mid < 1:
            break
        if _can_run(model, tokenizer, mid, seq_len, device):
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    b_max = max(1, int(best))
    hr = headroom if 0 < headroom <= 1.0 else 0.8
    b_train = max(1, int(b_max * hr))
    return b_max, b_train


# --------------------------------------------------------------------------- #
# LR candidate grid
# --------------------------------------------------------------------------- #

def get_lr_candidates(min_lr: float, max_lr: float, points: int) -> list[float]:
    if min_lr <= 0 or max_lr <= 0:
        raise ValueError("get_lr_candidates requires positive min_lr and max_lr.")
    if min_lr > max_lr:
        min_lr, max_lr = max_lr, min_lr
    if points <= 1:
        return [min_lr]
    log_min = math.log(min_lr)
    log_max = math.log(max_lr)
    step = (log_max - log_min) / (points - 1)
    return [math.exp(log_min + i * step) for i in range(points)]


def _pick_lr_peak_edge_of_stability(
    lr_losses: dict[float, float],
    *,
    rel_slack: float,
) -> float:
    finite = [(lr, loss) for lr, loss in lr_losses.items() if math.isfinite(loss)]
    if not finite:
        return min(lr_losses.keys())
    finite.sort(key=lambda x: x[0])
    min_loss = min(loss for _, loss in finite)
    if min_loss <= 0:
        min_loss = max(min_loss, 1e-8)
    cap = min_loss * (1.0 + rel_slack)
    acceptable = [lr for lr, loss in finite if loss <= cap]
    if acceptable:
        chosen = max(acceptable)
        print(
            f"  [LR Finder] peak rule: L_min={min_loss:.4f}  cap=(1+{rel_slack:g})·L_min={cap:.4f}  "
            f"→ pick largest LR under cap = {chosen:.2e}",
            flush=True,
        )
        return chosen
    # Pathological: widen once, else minimum-loss LR
    cap_wide = min_loss * (1.0 + 2.0 * rel_slack)
    acceptable_w = [lr for lr, loss in finite if loss <= cap_wide]
    if acceptable_w:
        return max(acceptable_w)
    return min(finite, key=lambda x: x[1])[0]


# --------------------------------------------------------------------------- #
# Micro-training kernels
# --------------------------------------------------------------------------- #

def _trainable_params(model) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _make_optimizer(
    model,
    lr: float,
    optimizer_name: Optional[str],
    weight_decay: float = 0.0,
):
    """Match training optim when possible; fall back to AdamW on trainable params only."""
    params = _trainable_params(model)
    key = (optimizer_name or "adamw_torch").lower().replace("-", "_")
    if key in ("paged_adamw_8bit", "pagedadamw8bit"):
        try:
            import bitsandbytes as bnb

            return bnb.optim.PagedAdamW8bit(
                params,
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=weight_decay,
            )
        except Exception as exc:
            print(
                f"[LR Finder] PagedAdamW8bit unavailable ({exc}); using AdamW.",
                flush=True,
            )
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _iter_batches_forever(dataloader):
    if len(dataloader) == 0:
        return
    while True:
        for batch in dataloader:
            yield batch


def _mini_train_mean_loss(
    model,
    dataloader,
    lr: float,
    device: str,
    optimizer_name: Optional[str],
    *,
    num_batches: int,
) -> float:
    """Run ``num_batches`` optimizer steps at ``lr``; return mean loss (tokenized batches only)."""
    optimizer = _make_optimizer(model, lr, optimizer_name)
    trainable = _trainable_params(model)
    model.train()
    losses: list[float] = []
    batch_iter = _iter_batches_forever(dataloader)
    for _ in range(max(1, int(num_batches))):
        batch = next(batch_iter)
        inputs = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        with _bf16_autocast(device):
            # Batch already includes ``labels``; do not pass ``labels=`` again (TypeError).
            loss = model(**inputs).loss
        if not torch.isfinite(loss):
            return float("inf")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("inf")


def _run_mini_train_lr_grid(
    model,
    dataset: Dataset,
    safe_batch: int,
    lr_candidates: list[float],
    device: str,
    optimizer_name: Optional[str],
    *,
    mini_train_batches: int,
    peak_rel_slack: float,
    collate_fn: Optional[Callable] = None,
) -> tuple[float, int]:
    trainable_snapshot = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    def _restore_trainable_weights() -> None:
        for name, param in model.named_parameters():
            if name in trainable_snapshot:
                param.data.copy_(trainable_snapshot[name])
        model.zero_grad(set_to_none=True)

    start_batch = max(1, safe_batch)
    batch_size = start_batch

    while batch_size >= 1:
        try:
            lr_losses: dict[float, float] = {}
            dl = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=False,
            )
            for lr in lr_candidates:
                _restore_trainable_weights()
                print(f"  [LR Finder] mini-train probe LR={lr:.2e} …", flush=True)
                avg_loss = _mini_train_mean_loss(
                    model,
                    dl,
                    lr,
                    device,
                    optimizer_name,
                    num_batches=mini_train_batches,
                )
                lr_losses[lr] = avg_loss
                print(f"    mean loss = {avg_loss:.4f}", flush=True)
                # Mitigate host/GPU memory growth across 20+ optimizers (OOM killer often
                # shows as ``Killed`` with no Python traceback).
                gc.collect()
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            best_lr = _pick_lr_peak_edge_of_stability(
                lr_losses, rel_slack=peak_rel_slack
            )
            print(f"  [LR Finder] selected LR from mini-train grid: {best_lr:.2e}", flush=True)
            return best_lr, batch_size
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(
                    f"  [LR Finder] OOM at batch_size={batch_size}, halving …",
                    flush=True,
                )
                batch_size //= 2
                torch.cuda.empty_cache()
            else:
                raise

    raise RuntimeError("Even batch_size=1 does not fit in GPU memory.")


# --------------------------------------------------------------------------- #
# Dataset loading — 2% of tokenized instruct JSON
# --------------------------------------------------------------------------- #

def _config_bool_or_default(value, default: bool) -> bool:
    """Parse booleans from JSON/YAML; treat ``None`` as *use default*."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("0", "false", "no", "n", "off", ""):
            return False
        if s in ("1", "true", "yes", "y", "on"):
            return True
    return bool(value)


def _stratified_sample_by_token_length(
    rows: list[dict],
    k: int,
    seed: int,
) -> list[dict]:
    """Stratify probe rows by ``len(input_ids)`` (matches training sequence lengths)."""
    rng = random.Random(seed)
    valid_idx = [
        i
        for i, r in enumerate(rows)
        if isinstance(r.get("input_ids"), list) and len(r["input_ids"]) > 0
    ]
    if not valid_idx:
        valid_idx = list(range(len(rows)))
    if len(valid_idx) <= k:
        return [rows[i] for i in valid_idx]

    valid_idx.sort(key=lambda i: len(rows[i]["input_ids"]))
    if k < 4 or len(valid_idx) < 4:
        rng.shuffle(valid_idx)
        return [rows[i] for i in valid_idx[:k]]

    n_bins = 4
    per, rem = divmod(k, n_bins)
    picked: list[int] = []
    for b in range(n_bins):
        lo = b * len(valid_idx) // n_bins
        hi = (b + 1) * len(valid_idx) // n_bins
        bucket = valid_idx[lo:hi]
        take = per + (1 if b < rem else 0)
        if not bucket or take <= 0:
            continue
        rng.shuffle(bucket)
        picked.extend(bucket[:take])

    if len(picked) < k:
        pool = [i for i in valid_idx if i not in set(picked)]
        rng.shuffle(pool)
        picked.extend(pool[: k - len(picked)])
    rng.shuffle(picked)
    return [rows[i] for i in picked[:k]]


def _normalize_tokenized_row(ex: dict) -> Optional[dict]:
    """Keep only keys needed for ``pad_inputs`` / causal LM forward."""
    ids = ex.get("input_ids")
    if not isinstance(ids, list) or not ids:
        return None
    attn = ex.get("attention_mask")
    if not isinstance(attn, list) or len(attn) != len(ids):
        attn = [1] * len(ids)
    lab = ex.get("labels")
    if not isinstance(lab, list) or len(lab) != len(ids):
        lab = list(ids)
    return {
        "input_ids": [int(x) for x in ids],
        "attention_mask": [int(x) for x in attn],
        "labels": [int(x) for x in lab],
    }


def _stack_tokenized_batch(
    rows: list[dict],
    tokenizer: AutoTokenizer,
    seq_len: int,
) -> dict:
    """Pad each example to ``seq_len`` like ``MyDataset``, then stack to batch tensors."""
    tensors: dict[str, list[torch.Tensor]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for raw in rows:
        dp = _normalize_tokenized_row(raw)
        if dp is None:
            continue
        padded = pad_inputs(tokenizer, dp, seq_len, tokenizer.padding_side)
        for key in tensors:
            tensors[key].append(torch.tensor(padded[key], dtype=torch.long))
    if not tensors["input_ids"]:
        raise RuntimeError("tokenized batch has no valid rows")
    return {k: torch.stack(v, dim=0) for k, v in tensors.items()}


def _make_tokenized_collate_fn(
    tokenizer: AutoTokenizer,
    seq_len: int,
) -> Callable[[list[dict]], dict]:
    def _collate(batch: list[dict]) -> dict:
        return _stack_tokenized_batch(batch, tokenizer, seq_len)

    return _collate


def _row_token_count(row: dict) -> int:
    ids = row.get("input_ids")
    return len(ids) if isinstance(ids, list) else 0


def _expand_indices_to_token_targets(
    data: list[dict],
    picked: set[int],
    rng: random.Random,
    target_tokens_min: int,
    target_tokens_max: int,
) -> set[int]:
    """Add rows until total tokens ≥ ``target_tokens_min`` (cap at ``target_tokens_max`` sum)."""
    total = sum(_row_token_count(data[i]) for i in picked)
    if total >= target_tokens_min:
        return picked
    pool = [i for i in range(len(data)) if i not in picked]
    rng.shuffle(pool)
    for idx in pool:
        if total >= target_tokens_min:
            break
        picked.add(idx)
        total += _row_token_count(data[idx])
        if total >= target_tokens_max:
            break
    return picked


def _load_tokenized_probe_dataset(
    tokenized_path: str,
    *,
    stratify_by_length: bool,
    seed: int,
    min_probe_rows: int = 32,
    probe_fraction: float = 0.02,
    target_tokens_min: int = 50_000,
    target_tokens_max: int = 200_000,
) -> tuple[Dataset, str]:
    """
    Probe subset for LR search: at least ``max(min_probe_rows, ceil(probe_fraction * N))``
    rows (capped by ``N``), then expand with random rows until total token count reaches
    ``target_tokens_min`` when data allows (up to ``target_tokens_max`` total tokens).

    Same JSON format as ``MyDataset`` / ``train_tokenized_*.json``.
    """
    with open(tokenized_path, "r", encoding="utf-8") as f:
        raw_list = json.load(f)
    if not isinstance(raw_list, list):
        raise ValueError(f"Tokenized dataset must be a JSON list: {tokenized_path}")
    data = []
    for ex in raw_list:
        if isinstance(ex, dict):
            norm = _normalize_tokenized_row(ex)
            if norm is not None:
                data.append(norm)
    n_rows = len(data)
    if n_rows == 0:
        return Dataset.from_list([]), TOKENIZED_BATCH_KEY, 0

    pf = max(0.0, min(1.0, float(probe_fraction)))
    base_k = max(min_probe_rows, int(math.ceil(n_rows * pf)))
    base_k = max(1, min(base_k, n_rows))

    rng = random.Random(seed)
    if stratify_by_length:
        initial_rows = _stratified_sample_by_token_length(data, base_k, seed)
        picked = {
            i
            for i, r in enumerate(data)
            if any(r is x for x in initial_rows)
        }
        if not picked:
            picked = set(range(min(base_k, n_rows)))
    else:
        order = list(range(n_rows))
        rng.shuffle(order)
        picked = set(order[:base_k])

    picked = _expand_indices_to_token_targets(
        data, picked, rng, target_tokens_min, target_tokens_max
    )
    subset = [data[i] for i in sorted(picked)]
    total_tokens = sum(_row_token_count(r) for r in subset)

    return Dataset.from_list(subset), TOKENIZED_BATCH_KEY, total_tokens


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def find_lr(
    model_id: str,
    model_path: str,
    num_params: Optional[int],
    tokenized_dataset_path: str,
    dataset_type_dict: dict,
    *,
    min_lr: float = 1e-6,
    max_lr: float = 9e-3,
    lr_probe_points: int = 28,
    mini_train_batches: int = 3,
    seq_len: int = 1024,
    lora_threshold: Optional[int] = None,
    lora_r: int = _DEFAULT_LORA_R,
    lora_alpha: int = _DEFAULT_LORA_ALPHA,
    lora_dropout: float = _DEFAULT_LORA_DROPOUT,
    optimizer_name: Optional[str] = None,
    lr_sample_stratify: bool = True,
    lr_sample_seed: int = 42,
    batch_headroom: float = 0.8,
    grpo_slow_reward_proxy_probe: bool = False,
    peak_rel_slack: float = 0.28,
) -> Optional[dict]:
    if not tokenized_dataset_path or not os.path.isfile(tokenized_dataset_path):
        print(
            "[LR Finder] tokenized_dataset_path missing or not a file; need train_tokenized JSON.",
            flush=True,
        )
        return None

    use_lora = (
        lora_threshold is not None
        and num_params is not None
        and num_params >= lora_threshold
    )
    _points = max(
        2,
        int(dataset_type_dict.get("lr_finder_lr_probe_points", lr_probe_points)),
    )
    _mini_b = max(
        1,
        int(dataset_type_dict.get("lr_finder_mini_train_batches", mini_train_batches)),
    )
    _peak_sl = float(dataset_type_dict.get("lr_finder_peak_rel_slack", peak_rel_slack))
    _peak_sl = max(0.0, min(3.0, _peak_sl))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _proxy_grpo = bool(
        dataset_type_dict.get("lr_finder_grpo_slow_reward_proxy", grpo_slow_reward_proxy_probe)
    )
    _opt_disp = optimizer_name or "adamw_torch"
    _probe_tag = (
        "GRPO_slow_reward→SFT_CE_proxy"
        if _proxy_grpo
        else "tokenized_SFT_CE"
    )
    print(
        f"[LR Finder] model={model_id}  mini_train={_probe_tag}  params={num_params}  lora={use_lora}"
        + (f"  lora_r={lora_r}" if use_lora else "")
        + f"  optim={_opt_disp}  probe=tokenized(min_rows+fraction+tok_budget)  lr_probe_points={_points}"
        + f"  mini_train_batches={_mini_b}"
        + f"  peak_rel_slack={_peak_sl:g}",
        flush=True,
    )
    if _proxy_grpo:
        print(
            "[LR Finder] GRPO slow-reward task: this probe uses **SFT cross-entropy only** "
            "on tokenized JSON (no rollouts, no reward funcs — langcheck/detoxify/textstat/etc.). "
            "Found LR/batch apply to full GRPO training afterward.",
            flush=True,
        )

    try:
        # ------------------------------------------------------------------ #
        # Load tokenizer
        # ------------------------------------------------------------------ #
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ------------------------------------------------------------------ #
        # Load model in bfloat16 (same dtype as production training).
        # float16 has a much smaller dynamic range and produces NaN on backward.
        # ------------------------------------------------------------------ #
        print(f"[LR Finder] Loading model from {model_path} …", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        # ------------------------------------------------------------------ #
        # Apply LoRA when the task trains with LoRA (matches real training).
        # ------------------------------------------------------------------ #
        if use_lora:
            target_modules = _find_all_linear_names(model)
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()
            # Required so gradient checkpointing can propagate gradients through
            # frozen base-model layers (LoRA only trains adapter params).
            model.enable_input_require_grads()

        # use_cache is incompatible with gradient checkpointing — disable it first.
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

        # ------------------------------------------------------------------ #
        # Find B_max (synthetic), then B_train = headroom × B_max (default 0.8).
        # ------------------------------------------------------------------ #
        _hr = float(
            dataset_type_dict.get("lr_finder_batch_headroom", batch_headroom)
        )
        _hr = _hr if 0 < _hr <= 1.0 else 0.8
        print("[LR Finder] Probing max batch size (synthetic seq_len) …", flush=True)
        b_max, b_train = find_max_batch_size(
            model, tokenizer, seq_len=seq_len, device=device, headroom=_hr
        )
        print(
            f"[LR Finder] B_max={b_max}  B_train={b_train}  (headroom={_hr:g}× for stability)",
            flush=True,
        )
        safe_batch = b_train

        _strat = _config_bool_or_default(
            dataset_type_dict.get("lr_finder_stratify_length"),
            lr_sample_stratify,
        )
        _sseed = int(dataset_type_dict.get("lr_finder_sample_seed", lr_sample_seed))

        _min_rows = int(dataset_type_dict.get("lr_finder_min_probe_rows", 32))
        _probe_frac = float(dataset_type_dict.get("lr_finder_probe_fraction", 0.02))
        _tok_min = int(dataset_type_dict.get("lr_finder_target_tokens_min", 50_000))
        _tok_max = int(dataset_type_dict.get("lr_finder_target_tokens_max", 200_000))

        print(
            f"[LR Finder] Loading probe subset from tokenized data: {tokenized_dataset_path} …",
            flush=True,
        )
        sample_ds, _, probe_tokens = _load_tokenized_probe_dataset(
            tokenized_dataset_path,
            stratify_by_length=_strat,
            seed=_sseed,
            min_probe_rows=max(1, _min_rows),
            probe_fraction=_probe_frac,
            target_tokens_min=max(0, _tok_min),
            target_tokens_max=max(_tok_min, _tok_max),
        )
        collate_fn = _make_tokenized_collate_fn(tokenizer, seq_len)
        print(
            f"[LR Finder] Probe rows: {len(sample_ds)}  ~{probe_tokens} tokens  "
            f"(floor=max({_min_rows},{_probe_frac:g}·N), target {_tok_min}–{_tok_max} tok; "
            f"stratify={_strat})",
            flush=True,
        )

        if len(sample_ds) == 0:
            print("[LR Finder] Empty probe sample; cannot run LR search.", flush=True)
            return None

        lr_candidates = get_lr_candidates(min_lr, max_lr, _points)
        print(
            f"[LR Finder] Mini-train LR grid: {_points} candidates  {min_lr:.2e} → {max_lr:.2e}",
            flush=True,
        )
        best_lr, probe_batch = _run_mini_train_lr_grid(
            model,
            sample_ds,
            safe_batch,
            lr_candidates,
            device,
            optimizer_name=optimizer_name,
            mini_train_batches=_mini_b,
            peak_rel_slack=_peak_sl,
            collate_fn=collate_fn,
        )

        train_batch = min(b_train, probe_batch)
        print(
            f"[LR Finder] Selected LR: {best_lr:.2e}  "
            f"(probe batch_size={probe_batch}; B_train={b_train}; "
            f"training batch_size=min(B_train, probe)={train_batch})",
            flush=True,
        )
        return {
            "lr": best_lr,
            "batch_size": train_batch,
            "b_max_synthetic": b_max,
            "b_train_synthetic": b_train,
            "probe_batch_real_data": probe_batch,
            "lr_probe_points": _points,
            "mini_train_batches": _mini_b,
        }

    except Exception as exc:
        print(f"[LR Finder] Error during LR search: {exc}", flush=True)
        return None

    finally:
        torch.cuda.empty_cache()
