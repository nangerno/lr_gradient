import contextlib
import json
import math
import random
from typing import Literal, Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

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
# Text extraction helpers
# --------------------------------------------------------------------------- #

def example_to_text(example: dict, mapping: Optional[dict] = None) -> str:
    mapping = mapping or {}
    inst_key = mapping.get("field_instruction", "instruction")
    input_key = mapping.get("field_input", "input")
    output_key = mapping.get("field_output", "output")

    parts = []
    for key in (inst_key, input_key, output_key):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    if parts:
        return "\n".join(parts)

    for key in ("text", "content", "prompt", "question"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for value in example.values():
        if isinstance(value, str) and value.strip():
            return value.strip()

    return json.dumps(example, ensure_ascii=True)


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


def _moving_average_np(x: np.ndarray, window: int) -> np.ndarray:
    """Moving average with `mode=same` length; no-op when `window <= 1`."""
    x = np.asarray(x, dtype=np.float64)
    if window <= 1 or len(x) < 2:
        return x
    w = min(window, len(x))
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(x, kernel, mode="same")


def _pick_lr_from_smith_curve(
    lrs: list[float],
    losses: list[float],
    *,
    min_lr: float,
    max_lr: float,
    smith_safety_divisor: float,
    mode: Literal["steepest", "max_decreasing"] = "steepest",
) -> float:
    """
    Leslie Smith ramp — pick a training LR from the recorded (lr, loss) curve.

    - ``steepest``: most negative ``d(loss)/d(log lr)`` (fastai-style ``steep``).
    - ``max_decreasing``: largest LR while smoothed loss is still **non-increasing**
      along the ramp (first sustained rise ends the “stable decreasing” zone), then
      ``/ smith_safety_divisor`` and clamp to ``[min_lr, max_lr]``.
    """

    def _clamp(r: float) -> float:
        return float(max(min_lr, min(r, max_lr)))

    lr_arr = np.array(lrs, dtype=np.float64)
    loss_arr = np.array(losses, dtype=np.float64)
    finite = np.isfinite(loss_arr) & np.isfinite(lr_arr) & (lr_arr > 0)
    if not finite.any():
        return _clamp(float(lr_arr[0]) if len(lr_arr) else 1e-6)

    lr_f = lr_arr[finite]
    loss_f = loss_arr[finite]
    # Match fastai Learner.lr_find: drop noisy start and divergent tail before suggesting.
    _trim_lo = len(lr_f) // 10
    _trim_hi = 5
    if len(lr_f) > _trim_lo + _trim_hi + 4:
        lr_f = lr_f[_trim_lo : len(lr_f) - _trim_hi]
        loss_f = loss_f[_trim_lo : len(loss_f) - _trim_hi]

    if len(lr_f) < 3:
        j = int(np.nanargmin(loss_f))
        raw = float(lr_f[j])
        out = _clamp(raw / smith_safety_divisor)
        print(
            f"  [LR Finder] Smith curve: short ramp — min-loss anchor {raw:.2e} → "
            f"LR {out:.2e} (÷{smith_safety_divisor:g})",
            flush=True,
        )
        return out

    smooth = _moving_average_np(loss_f, window=min(5, len(loss_f)))
    if len(smooth) < 2:
        return _clamp(float(lr_f[0]))

    if mode == "max_decreasing":
        # Relative tolerance: tolerate tiny noise while LR increases.
        # Require a short *streak* of rises so one noisy point does not end the zone.
        rise_patience = 2
        rise_streak = 0
        rise_start_idx = -1
        raw_anchor: Optional[float] = None
        for i in range(1, len(smooth)):
            if not (np.isfinite(smooth[i]) and np.isfinite(smooth[i - 1])):
                continue
            prev = float(smooth[i - 1])
            tol = 1e-4 * (1.0 + abs(prev))
            if float(smooth[i]) > prev + tol:
                if rise_streak == 0:
                    rise_start_idx = i
                rise_streak += 1
                if rise_streak >= rise_patience:
                    # Anchor at the last LR before the sustained rise starts.
                    raw_anchor = float(lr_f[max(0, rise_start_idx - 1)])
                    break
            else:
                rise_streak = 0
                rise_start_idx = -1
        if raw_anchor is None:
            raw_anchor = float(lr_f[-1])
            reason = "smoothed loss still decreasing at highest LR"
        else:
            reason = "last LR before sustained smoothed-loss rise"
        out = _clamp(raw_anchor / smith_safety_divisor)
        print(
            f"  [LR Finder] Smith curve (max_decreasing): anchor {raw_anchor:.2e} "
            f"({reason}) → training LR {out:.2e} (÷{smith_safety_divisor:g})",
            flush=True,
        )
        return out

    log_lr = np.log(lr_f)
    dlog = np.diff(log_lr)
    dloss = np.diff(smooth)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(dlog != 0, dloss / dlog, np.nan)

    valid = np.isfinite(slope)
    if not valid.any():
        return _clamp(float(lr_f[0]))

    masked = np.where(valid, slope, np.nan)
    if np.nanmin(masked) >= 0:
        print(
            "  [LR Finder] Smith curve: loss never decreases along ramp; "
            f"using min LR {lr_f[0]:.2e}",
            flush=True,
        )
        return _clamp(float(lr_f[0]))

    # Steepest descent: most negative slope (loss drops fastest as LR increases).
    # Index j is the left edge of that segment (same convention as fastai ``steep``).
    j = int(np.nanargmin(masked))
    raw_steep = float(lr_f[j])
    out = _clamp(raw_steep / smith_safety_divisor)
    print(
        f"  [LR Finder] Smith curve: steepest ~{raw_steep:.2e} → training LR {out:.2e} "
        f"(÷{smith_safety_divisor:g})",
        flush=True,
    )
    return out


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


def _micro_train(
    model,
    tokenizer,
    dataloader,
    lr: float,
    text_key: str,
    steps: int,
    device: str,
    seq_len: int = 512,
    optimizer_name: Optional[str] = None,
) -> float:
    """Generic causal-LM micro-training used for instruct, grpo, and dpo."""
    optimizer = _make_optimizer(model, lr, optimizer_name)
    trainable = _trainable_params(model)
    model.train()
    losses: list[float] = []
    batch_iter = _iter_batches_forever(dataloader)

    for _ in range(max(1, int(steps))):
        try:
            batch = next(batch_iter)
        except StopIteration:
            # Empty dataloader: caller should handle this as unusable LR.
            break
        optimizer.zero_grad()

        texts = batch[text_key]
        # Truncate to seq_len so memory usage matches the find_max_batch_size probe.
        inputs = tokenizer(
            texts, return_tensors="pt", truncation=True, padding=True,
            max_length=seq_len,
        ).to(device)

        with _bf16_autocast(device):
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

        if not torch.isfinite(loss):
            continue

        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        # Early-stop if loss diverges sharply
        if len(losses) > 5 and loss.item() > 3 * min(losses):
            break

    return float(np.mean(losses)) if losses else float("inf")


def _iter_batches_forever(dataloader):
    if len(dataloader) == 0:
        return
    while True:
        for batch in dataloader:
            yield batch


def _micro_train_leslie_smith(
    model,
    tokenizer,
    dataloader,
    lr_schedule: list[float],
    text_key: str,
    device: str,
    seq_len: int,
    optimizer_name: Optional[str],
    *,
    micro_batches_per_lr: int = 1,
    early_stop_on_divergence: bool = True,
    divergence_vs_min_loss: float = 10.0,
    min_points_before_divergence_check: int = 5,
) -> tuple[list[float], list[float]]:
    """
    Single continuous run with exponentially increasing LR each step (Leslie Smith).

    ``micro_batches_per_lr``: gradient accumulation — averages gradient over this many
    batches per LR point for a less noisy loss curve (linear time cost).

    ``early_stop_on_divergence``: stop the ramp when loss spikes vs the running minimum
    (saves time once the useful LR range has been passed).
    """
    if not lr_schedule:
        return [], []

    mb = max(1, int(micro_batches_per_lr))
    optimizer = _make_optimizer(model, lr_schedule[0], optimizer_name)
    trainable = _trainable_params(model)
    model.train()
    lrs_recorded: list[float] = []
    losses_recorded: list[float] = []
    batch_iter = _iter_batches_forever(dataloader)

    for step in range(len(lr_schedule)):
        lr = lr_schedule[step]
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        batch_losses: list[float] = []

        for _ in range(mb):
            batch = next(batch_iter)
            texts = batch[text_key]
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=seq_len,
            ).to(device)

            with _bf16_autocast(device):
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

            if not torch.isfinite(loss):
                lrs_recorded.append(lr)
                losses_recorded.append(float("nan"))
                print(
                    "  [LR Finder] Smith ramp: non-finite loss; stopping curve early.",
                    flush=True,
                )
                return lrs_recorded, losses_recorded

            batch_losses.append(loss.item())
            (loss / mb).backward()

        mean_loss = float(np.mean(batch_losses))

        if (
            early_stop_on_divergence
            and len(losses_recorded) >= min_points_before_divergence_check
            and losses_recorded
        ):
            running_min = min(losses_recorded)
            if math.isfinite(running_min) and running_min > 0:
                if mean_loss > divergence_vs_min_loss * running_min:
                    lrs_recorded.append(lr)
                    losses_recorded.append(mean_loss)
                    optimizer.zero_grad(set_to_none=True)
                    print(
                        "  [LR Finder] Smith ramp: early stop — loss spiked vs minimum "
                        f"(lr={lr:.2e}, loss={mean_loss:.4f}); saving wall time.",
                        flush=True,
                    )
                    break

        lrs_recorded.append(lr)
        losses_recorded.append(mean_loss)
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

    return lrs_recorded, losses_recorded


def _run_smith_with_auto_batch(
    model,
    tokenizer,
    dataset: Dataset,
    safe_batch: int,
    lr_schedule: list[float],
    text_key: str,
    device: str,
    seq_len: int = 512,
    optimizer_name: Optional[str] = None,
    *,
    min_lr: float,
    max_lr: float,
    smith_safety_divisor: float,
    smith_micro_batches: int = 1,
    smith_early_stop_divergence: bool = True,
    smith_divergence_vs_min: float = 10.0,
    smith_min_points_before_divergence: int = 5,
    smith_curve_mode: Literal["steepest", "max_decreasing"] = "steepest",
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

    # One causal-LM forward per row (DPO uses ``chosen`` only; not full DPO pairwise loss).
    # Start at the headroom batch (e.g. 0.8× synthetic max); halve on real-data OOM.
    start_batch = max(1, safe_batch)
    batch_size = start_batch

    while batch_size >= 1:
        try:
            _restore_trainable_weights()
            dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            lrs_out, losses_out = _micro_train_leslie_smith(
                model,
                tokenizer,
                dl,
                lr_schedule,
                text_key,
                device,
                seq_len,
                optimizer_name,
                micro_batches_per_lr=smith_micro_batches,
                early_stop_on_divergence=smith_early_stop_divergence,
                divergence_vs_min_loss=smith_divergence_vs_min,
                min_points_before_divergence_check=smith_min_points_before_divergence,
            )
            best_lr = _pick_lr_from_smith_curve(
                lrs_out,
                losses_out,
                min_lr=min_lr,
                max_lr=max_lr,
                smith_safety_divisor=smith_safety_divisor,
                mode=smith_curve_mode,
            )
            # If max_decreasing clamps to min_lr, steepest descent often sits higher — use it as fallback.
            if (
                smith_curve_mode == "max_decreasing"
                and lrs_out
                and math.isclose(best_lr, min_lr, rel_tol=0.0, abs_tol=max(min_lr * 1e-9, 1e-15))
            ):
                steepest_lr = _pick_lr_from_smith_curve(
                    lrs_out,
                    losses_out,
                    min_lr=min_lr,
                    max_lr=max_lr,
                    smith_safety_divisor=smith_safety_divisor,
                    mode="steepest",
                )
                if steepest_lr > best_lr + max(min_lr * 1e-9, 1e-15):
                    print(
                        f"  [LR Finder] Primary mode hit min_lr; steepest-curve fallback "
                        f"{steepest_lr:.2e} (was {best_lr:.2e})",
                        flush=True,
                    )
                    best_lr = steepest_lr
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


def _pick_best_lr(
    model,
    tokenizer,
    dataloader,
    lr_candidates: list[float],
    text_key: str,
    steps: int,
    device: str,
    seq_len: int = 512,
    optimizer_name: Optional[str] = None,
) -> float:
    # Reset trainable weights before each LR so probes do not compound on prior steps.
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

    lr_losses: dict[float, float] = {}
    for lr in lr_candidates:
        _restore_trainable_weights()
        print(f"  [LR Finder] Testing LR {lr:.2e} ...", flush=True)
        loss = _micro_train(
            model,
            tokenizer,
            dataloader,
            lr,
            text_key,
            steps,
            device,
            seq_len=seq_len,
            optimizer_name=optimizer_name,
        )
        lr_losses[lr] = loss
        print(f"    avg loss = {loss:.4f}", flush=True)

    finite_losses = {lr: loss for lr, loss in lr_losses.items() if math.isfinite(loss)}
    if not finite_losses:
        print("  [LR Finder] All LR candidates produced NaN/Inf loss.", flush=True)
        # Return the smallest LR as the safest fallback
        return min(lr_losses.keys())

    # Prefer the largest LR whose loss is close to the best loss.
    # This better matches "largest stable LR" than pure argmin on noisy micro-runs.
    best_loss = min(finite_losses.values())
    tol = max(1e-4, 0.02 * abs(best_loss))
    near_best = [lr for lr, loss in finite_losses.items() if loss <= best_loss + tol]
    best_lr = max(near_best) if near_best else min(finite_losses, key=finite_losses.get)
    print(
        f"  [LR Finder] Best LR: {best_lr:.2e}  "
        f"(best_loss={best_loss:.4f}, tol=+{tol:.4f})",
        flush=True,
    )
    return best_lr


def _run_with_auto_batch(
    model,
    tokenizer,
    dataset: Dataset,
    safe_batch: int,
    lr_candidates: list[float],
    text_key: str,
    steps: int,
    device: str,
    seq_len: int = 512,
    optimizer_name: Optional[str] = None,
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

    # Same as Leslie Smith ramp: one forward per row (DPO: ``chosen`` only).
    start_batch = max(1, safe_batch)
    batch_size = start_batch

    while batch_size >= 1:
        try:
            _restore_trainable_weights()
            # shuffle=False: _load_sample_dataset already shuffled; keeps batch order
            # identical across LR candidates so losses are comparable.
            dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            best_lr = _pick_best_lr(
                model,
                tokenizer,
                dl,
                lr_candidates,
                text_key,
                steps,
                device,
                seq_len=seq_len,
                optimizer_name=optimizer_name,
            )
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
# Dataset loading — shuffled subset for probing
# --------------------------------------------------------------------------- #

def _resolve_smith_curve_mode(
    dataset_type_dict: dict,
    fallback: Literal["steepest", "max_decreasing"],
) -> Literal["steepest", "max_decreasing"]:
    """Normalize mode strings so typos in ``train_info`` cannot silently select the wrong branch."""
    fb = str(fallback).strip().lower().replace("-", "_")
    if fb not in ("steepest", "max_decreasing"):
        fb = "max_decreasing"
    raw = dataset_type_dict.get("lr_finder_smith_curve_mode")
    if raw is None:
        return fb  # type: ignore[return-value]
    t = str(raw).strip().lower().replace("-", "_")
    if t in ("steepest", "max_decreasing"):
        return t  # type: ignore[return-value]
    return fb  # type: ignore[return-value]


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


def _adaptive_probe_sample_count(
    n_rows: int,
    *,
    frac: float,
    min_n: int,
    max_n: int,
) -> int:
    """How many rows to use for the LR probe given full dataset size."""
    if n_rows <= 0:
        return 0
    target = max(min_n, int(n_rows * frac))
    capped = min(target, max_n, n_rows)
    # Avoid sample_count=0 when rows exist (empty DataLoader would make
    # ``_iter_batches_forever`` spin forever).
    return max(1, capped)


def _stratified_sample_by_text_length(
    rows: list[dict],
    text_key: str,
    k: int,
    seed: int,
) -> list[dict]:
    """
    Spread probe examples across short→long texts (quartiles) so the loss
    landscape is not dominated by a random slice of similar lengths.
    """
    rng = random.Random(seed)
    valid_idx = [i for i, r in enumerate(rows) if (r.get(text_key) or "").strip()]
    if not valid_idx:
        valid_idx = list(range(len(rows)))
    if len(valid_idx) <= k:
        return [rows[i] for i in valid_idx]

    valid_idx.sort(key=lambda i: len((rows[i].get(text_key) or "")))
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


def _load_sample_dataset(
    dataset_path: str,
    dataset_type_dict: dict,
    train_type: str,
    *,
    sample_frac: float = 0.02,
    sample_min: int = 200,
    sample_max: int = 3000,
    stratify_by_length: bool = True,
    seed: int = 42,
) -> tuple[Dataset, str]:
    raw = load_dataset("json", data_files=dataset_path)["train"]

    if train_type == "dpo":
        chosen_key = dataset_type_dict.get("field_chosen", "chosen")
        rejected_key = dataset_type_dict.get("field_rejected", "rejected")

        def _map_dpo(ex):
            return {
                "text": example_to_text(ex, dataset_type_dict),
                "chosen": ex.get(chosen_key, ""),
                "rejected": ex.get(rejected_key, ""),
            }

        ds = raw.map(_map_dpo, remove_columns=raw.column_names)
        text_key = "chosen"  # forward pass on chosen text for loss

    elif train_type == "grpo":
        prompt_key = dataset_type_dict.get("field_prompt", "prompt")

        def _map_grpo(ex):
            return {"text": ex.get(prompt_key) or example_to_text(ex, dataset_type_dict)}

        ds = raw.map(_map_grpo, remove_columns=raw.column_names)
        text_key = "text"

    else:  # instruct (default)
        def _map_instruct(ex):
            return {"text": example_to_text(ex, dataset_type_dict)}

        ds = raw.map(_map_instruct, remove_columns=raw.column_names)
        text_key = "text"

    data = list(ds)
    sample_count = _adaptive_probe_sample_count(
        len(data), frac=sample_frac, min_n=sample_min, max_n=sample_max
    )
    if stratify_by_length:
        subset = _stratified_sample_by_text_length(data, text_key, sample_count, seed)
    else:
        rng = random.Random(seed)
        rng.shuffle(data)
        subset = data[:sample_count]

    return Dataset.from_list(subset), text_key


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def find_lr(
    model_id: str,
    model_path: str,
    num_params: Optional[int],
    dataset_path: str,
    dataset_type_dict: dict,
    train_type: str = "instruct",
    min_lr: float = 6e-6,
    max_lr: float = 1e-3,
    # Grid search only; Leslie Smith uses `steps` as the number of schedule points.
    lr_points: int = 30,
    steps: int = 20,
    seq_len: int = 512,
    # LoRA config — should match the actual training setup for this task so
    # that the probed LR is representative of the real loss landscape.
    # Pass lora_threshold=None to disable LoRA entirely (e.g. full-weight SFT).
    lora_threshold: Optional[int] = _DEFAULT_LORA_THRESHOLD,
    lora_r: int = _DEFAULT_LORA_R,
    lora_alpha: int = _DEFAULT_LORA_ALPHA,
    lora_dropout: float = _DEFAULT_LORA_DROPOUT,
    # e.g. "paged_adamw_8bit" to match GRPO/SFT training; None → AdamW on trainable only.
    optimizer_name: Optional[str] = None,
    # Leslie Smith exponential ramp (single run) vs independent grid micro-runs.
    search_mode: Literal["leslie_smith", "grid"] = "leslie_smith",
    # Divide raw steepest-descent LR by this (then clamp to [min_lr, max_lr]).
    # None → 10 for full finetune, 5 for LoRA (steepest point is often ~10× too high to train).
    smith_safety_divisor: Optional[float] = None,
    # Smith ramp: accumulate this many micro-batches per LR point (stabler curve; linear time).
    smith_micro_batches: int = 1,
    # Stop ramp early when loss ≫ min loss so far (saves time past the useful LR band).
    smith_early_stop_divergence: bool = True,
    smith_divergence_vs_min: float = 10.0,
    smith_min_points_before_divergence: int = 5,
    # Probe subset: size scales with dataset rows; cap avoids huge JSON files.
    lr_sample_frac: float = 0.02,
    lr_sample_min: int = 200,
    lr_sample_max: int = 3000,
    lr_sample_stratify: bool = True,
    lr_sample_seed: int = 42,
    # Synthetic max batch → B_train = headroom × B_max (default 0.8); training uses
    # min(B_train, batch that fits on real tokenized batches).
    batch_headroom: float = 0.8,
    smith_curve_mode: Literal["steepest", "max_decreasing"] = "max_decreasing",
) -> Optional[dict]:
    use_lora = (
        lora_threshold is not None
        and num_params is not None
        and num_params >= lora_threshold
    )
    eff_smith_div = (
        smith_safety_divisor
        if smith_safety_divisor is not None
        else (5.0 if use_lora else 10.0)
    )
    if eff_smith_div <= 0:
        raise ValueError("smith_safety_divisor must be positive.")
    _mb = max(1, min(8, int(smith_micro_batches)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _opt_disp = optimizer_name or "adamw_torch"
    print(
        f"[LR Finder] model={model_id}  type={train_type}  "
        f"params={num_params}  lora={use_lora}"
        + (f"  lora_r={lora_r}" if use_lora else "")
        + f"  optim={_opt_disp}  search={search_mode}  micro_steps={steps}"
        + (f"  lr_points={lr_points}" if search_mode == "grid" else "")
        + (
            f"  smith_safety_divisor={eff_smith_div:g}"
            if search_mode == "leslie_smith"
            else ""
        )
        + (
            f"  smith_micro_batches={_mb}"
            f"  smith_early_stop={smith_early_stop_divergence}"
            if search_mode == "leslie_smith"
            else ""
        )
        + (
            f"  smith_curve={_resolve_smith_curve_mode(dataset_type_dict, smith_curve_mode)}"
            if search_mode == "leslie_smith"
            else ""
        ),
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
        _curve_mode = _resolve_smith_curve_mode(dataset_type_dict, smith_curve_mode)

        # ------------------------------------------------------------------ #
        # Load dataset sample (size from row count; see _load_sample_dataset)
        # ------------------------------------------------------------------ #
        _frac = float(
            dataset_type_dict.get("lr_finder_sample_frac", lr_sample_frac)
        )
        _smin = int(dataset_type_dict.get("lr_finder_sample_min", lr_sample_min))
        _smax = int(dataset_type_dict.get("lr_finder_sample_max", lr_sample_max))
        _strat = _config_bool_or_default(
            dataset_type_dict.get("lr_finder_stratify_length"),
            lr_sample_stratify,
        )
        _sseed = int(
            dataset_type_dict.get("lr_finder_sample_seed", lr_sample_seed)
        )
        print(f"[LR Finder] Loading sample subset of {dataset_path} …", flush=True)
        sample_ds, text_key = _load_sample_dataset(
            dataset_path,
            dataset_type_dict,
            train_type,
            sample_frac=_frac,
            sample_min=_smin,
            sample_max=_smax,
            stratify_by_length=_strat,
            seed=_sseed,
        )
        print(
            f"[LR Finder] Sample: {len(sample_ds)} rows  "
            f"(frac={_frac:g}  min={_smin}  max={_smax}  "
            f"stratify_len={_strat})",
            flush=True,
        )
        if len(sample_ds) == 0:
            print(
                "[LR Finder] Empty dataset sample; cannot run LR search.",
                flush=True,
            )
            return None

        # ------------------------------------------------------------------ #
        # LR search: Leslie Smith ramp or discrete grid
        # ------------------------------------------------------------------ #
        if search_mode == "leslie_smith":
            lr_schedule = get_lr_candidates(min_lr, max_lr, steps)
            print(
                f"[LR Finder] Leslie Smith LR ramp: {steps} steps  "
                f"{min_lr:.2e} → {max_lr:.2e}",
                flush=True,
            )
            best_lr, probe_batch = _run_smith_with_auto_batch(
                model,
                tokenizer,
                sample_ds,
                safe_batch,
                lr_schedule,
                text_key,
                device,
                seq_len=seq_len,
                optimizer_name=optimizer_name,
                min_lr=min_lr,
                max_lr=max_lr,
                smith_safety_divisor=eff_smith_div,
                smith_micro_batches=_mb,
                smith_early_stop_divergence=smith_early_stop_divergence,
                smith_divergence_vs_min=smith_divergence_vs_min,
                smith_min_points_before_divergence=smith_min_points_before_divergence,
                smith_curve_mode=_curve_mode,
            )
        else:
            lr_candidates = get_lr_candidates(min_lr, max_lr, lr_points)
            print(
                f"[LR Finder] Grid sweep: {lr_points} LRs: {min_lr:.2e} → {max_lr:.2e}",
                flush=True,
            )
            best_lr, probe_batch = _run_with_auto_batch(
                model,
                tokenizer,
                sample_ds,
                safe_batch,
                lr_candidates,
                text_key,
                steps,
                device,
                seq_len=seq_len,
                optimizer_name=optimizer_name,
            )

        train_batch = min(b_train, probe_batch)
        print(
            f"[LR Finder] Selected LR: {best_lr:.2e}  "
            f"(probe on real data: batch_size={probe_batch}; "
            f"B_train={b_train}; training batch_size=min(B_train, probe)={train_batch})",
            flush=True,
        )
        return {
            "lr": best_lr,
            # Prefer B_train = headroom×B_max, but never exceed what fit real tokenized batches.
            "batch_size": train_batch,
            "b_max_synthetic": b_max,
            "b_train_synthetic": b_train,
            "probe_batch_real_data": probe_batch,
            "smith_curve_mode": _curve_mode,
        }

    except Exception as exc:
        print(f"[LR Finder] Error during LR search: {exc}", flush=True)
        return None

    finally:
        torch.cuda.empty_cache()
