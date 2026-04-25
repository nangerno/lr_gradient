import contextlib
import gc
import json
import math
import os
import tempfile
import warnings
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from utility import pad_inputs

from lr_finder_tasks import (
    LR_TASK_DPO,
    LR_TASK_GRPO,
    LR_TASK_INSTRUCT,
    _PreferenceCollator,
    _pad_batch_dict as _grpo_pad_batch_dict,
    build_dpo_preference_dataset,
    build_grpo_teacher_forced_dataset,
    dpo_sigmoid_loss,
    grpo_style_clipped_loss,
    mini_train_mean_loss_for_task,
    normalize_lr_finder_task,
    synthetic_dpo_batch,
    synthetic_grpo_batch,
)

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

# Full fine-tune: keep snapshot on disk instead of a permanent ~model-size CPU dict (host OOM).
_SNAPSHOT_DISK_THRESHOLD_BYTES = 256 * 1024 * 1024

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


def _can_run_dpo(
    model,
    tokenizer,
    batch_pairs: int,
    seq_len: int,
    device: str,
    beta: float,
    ref_trainable_cpu: Optional[Union[dict[str, torch.Tensor], str]] = None,
) -> bool:
    """``batch_pairs`` preference pairs → ``2 * batch_pairs`` rows (TRL-shaped batch)."""
    try:
        inputs = synthetic_dpo_batch(tokenizer, batch_pairs, seq_len, device)
        with _bf16_autocast(device):
            loss = dpo_sigmoid_loss(
                model, inputs, beta=beta, ref_trainable_cpu=ref_trainable_cpu
            )
        loss.backward()
        model.zero_grad()
        torch.cuda.empty_cache()
        return True
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            return False
        raise


def _can_run_grpo(
    model,
    tokenizer,
    batch_size: int,
    seq_len: int,
    device: str,
    *,
    epsilon_low: float,
    epsilon_high: float,
    beta: float,
) -> bool:
    try:
        inputs = synthetic_grpo_batch(tokenizer, batch_size, seq_len, device)
        with _bf16_autocast(device):
            loss = grpo_style_clipped_loss(
                model,
                inputs,
                epsilon_low=epsilon_low,
                epsilon_high=epsilon_high,
                beta=beta,
            )
        loss.backward()
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
    b_train_cap: Optional[int] = None,
    lr_task: str = LR_TASK_INSTRUCT,
    dpo_beta: float = 0.1,
    dpo_ref_trainable_cpu: Optional[Union[dict[str, torch.Tensor], str]] = None,
    grpo_epsilon_low: float = 0.2,
    grpo_epsilon_high: float = 0.2,
    grpo_beta: float = 0.04,
) -> tuple[int, int]:
    task = normalize_lr_finder_task(lr_task)

    def _probe_ok(unit_batch: int) -> bool:
        if task == LR_TASK_DPO:
            return _can_run_dpo(
                model,
                tokenizer,
                unit_batch,
                seq_len,
                device,
                dpo_beta,
                ref_trainable_cpu=dpo_ref_trainable_cpu,
            )
        if task == LR_TASK_GRPO:
            return _can_run_grpo(
                model,
                tokenizer,
                unit_batch,
                seq_len,
                device,
                epsilon_low=grpo_epsilon_low,
                epsilon_high=grpo_epsilon_high,
                beta=grpo_beta,
            )
        return _can_run(model, tokenizer, unit_batch, seq_len, device)

    if not _probe_ok(1):
        return 1, 1

    batch = 1
    while _probe_ok(batch):
        batch *= 2

    low, high, best = batch // 2, batch, batch // 2
    while low <= high:
        mid = (low + high) // 2
        if mid < 1:
            break
        if _probe_ok(mid):
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    b_max = max(1, int(best))
    hr = headroom if 0 < headroom <= 1.0 else 0.8
    b_train = max(1, int(b_max * hr))
    if b_train_cap is not None and b_train_cap > 0:
        b_train = min(b_train, int(b_train_cap))
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


def _effective_mini_train_batches(
    mini_train_batches: int,
    samples_per_lr: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    """
    Optimizer steps per LR probe: at least ``mini_train_batches``, and enough so that
    ``optimizer_steps × batch_size × gas`` is at least ``samples_per_lr`` (when > 0).
    """
    base = max(1, int(mini_train_batches))
    try:
        sp = int(samples_per_lr)
    except (TypeError, ValueError):
        sp = 0
    if sp <= 0:
        return base
    gas = max(1, int(gradient_accumulation_steps))
    b = max(1, int(per_device_batch_size))
    micro_per_opt_step = b * gas
    need = (sp + micro_per_opt_step - 1) // micro_per_opt_step
    return max(base, max(1, need))


def _longest_strictly_decreasing_loss_run(losses: list[float]) -> tuple[int, int]:
    """Inclusive index range of the longest contiguous run with losses[i] < losses[i-1]."""
    n = len(losses)
    if n == 0:
        return 0, -1
    if n == 1:
        return 0, 0
    best_lo, best_hi, best_len = 0, 0, 1
    lo = 0
    for i in range(1, n):
        if not (losses[i] < losses[i - 1]):
            hi = i - 1
            ln = hi - lo + 1
            if ln > best_len or (ln == best_len and hi > best_hi):
                best_len, best_lo, best_hi = ln, lo, hi
            lo = i
    hi = n - 1
    ln = hi - lo + 1
    if ln > best_len or (ln == best_len and hi > best_hi):
        best_lo, best_hi = lo, hi
    return best_lo, best_hi


def _lr_at_log_frac_inclusive(lrs: list[float], lo: int, hi: int, frac: float) -> float:
    """``frac`` in [0,1] along log-spanned LR interval [lrs[lo], lrs[hi]]."""
    if not lrs or lo > hi or lo < 0 or hi >= len(lrs):
        return lrs[0] if lrs else 1e-6
    frac = max(0.0, min(1.0, float(frac)))
    if lo == hi:
        return float(lrs[lo])
    t0 = math.log(max(lrs[lo], 1e-300))
    t1 = math.log(max(lrs[hi], 1e-300))
    return float(math.exp(t0 + frac * (t1 - t0)))


def _pick_lr_descending_segment(
    lr_losses: dict[float, float],
    *,
    trim_low_lr_frac: float = 0.2,
    explosion_rel_rolling_min: float = 2.5,
    explosion_step_ratio: float = 1.5,
    segment_pick_frac: float = 0.4,
) -> tuple[float, str]:
    """
    Post-process coarse (lr → mean loss) curve:

    1. Drop the lowest ``trim_low_lr_frac`` of LRs (too small / noisy left tail).
    2. Truncate before the first **explosion**: loss ≫ rolling min or sharp jump vs previous.
    3. Take the **longest** contiguous strictly loss-decreasing run (higher LR → lower loss).
    4. Pick LR at ``segment_pick_frac`` along that segment in **log-LR** (0=start, 1=end).
       Default 0.4 targets the 30–50% band center.
    """
    finite = [(lr, loss) for lr, loss in lr_losses.items() if math.isfinite(loss)]
    if not finite:
        lr0 = min(lr_losses.keys())
        return lr0, "no finite mean losses; returning smallest LR key"
    finite.sort(key=lambda x: x[0])
    lrs = [float(x[0]) for x in finite]
    losses = [float(x[1]) for x in finite]
    n = len(lrs)
    trim_k = int(math.floor(max(0.0, min(0.95, trim_low_lr_frac)) * n))
    trim_k = min(trim_k, max(0, n - 3))
    lrs, losses = lrs[trim_k:], losses[trim_k:]
    n = len(lrs)
    if n == 0:
        return finite[len(finite) // 2][0], "trim removed all points; median LR"
    if n == 1:
        return lrs[0], "single point after low-LR trim"

    end = n - 1
    for i in range(1, n):
        roll_min = min(losses[0 : i + 1])
        floor = max(roll_min, 1e-12)
        if losses[i] > explosion_rel_rolling_min * floor:
            end = i - 1
            break
        prev = max(losses[i - 1], 1e-12)
        if losses[i] > explosion_step_ratio * prev:
            end = i - 1
            break
    if end < 0:
        end = 0
    lrs, losses = lrs[: end + 1], losses[: end + 1]
    n = len(lrs)
    if n == 0:
        return finite[-1][0], "explosion clip emptied curve; last finite LR"

    lo_s, hi_s = _longest_strictly_decreasing_loss_run(losses)
    if lo_s > hi_s:
        j = int(np.argmin(losses)) if losses else 0
        j = max(0, min(j, len(lrs) - 1))
        return lrs[j], "no decreasing run; argmin loss LR in clipped window"

    pick_frac = max(0.0, min(1.0, float(segment_pick_frac)))
    chosen = _lr_at_log_frac_inclusive(lrs, lo_s, hi_s, pick_frac)
    msg = (
        f"descending_segment: trim_low={trim_k} dropped LRs, "
        f"explosion_end_index={end}, segment=[{lrs[lo_s]:.2e},{lrs[hi_s]:.2e}] "
        f"log-frac={pick_frac:g} → lr={chosen:.2e}"
    )
    return chosen, msg


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


def _lr_elbow_index_first_half_gain(
    losses: np.ndarray,
    lo: int,
    hi: int,
    *,
    improve_frac: float = 0.5,
) -> int:
    """
    First index in ``[lo, hi]`` (LR ascending) whose loss has improved by at least
    ``improve_frac`` of the drop from a short high-loss plateau (median of first ≤3
    points in the window) down to the window minimum.

    This tracks the LR-finder **elbow** (e.g. ~1.2e-5 when a later point is a batch-noise
    needle at ~6e-5) instead of always using the raw argmin loss.
    """
    if lo > hi:
        return max(0, int(lo))
    lo = max(0, min(int(lo), int(len(losses)) - 1))
    hi = max(lo, min(int(hi), int(len(losses)) - 1))
    Lw = losses[lo : hi + 1]
    if Lw.size == 0:
        return lo
    k = int(min(3, int(Lw.size)))
    L_high = float(np.median(Lw[:k]))
    L_min = float(np.min(Lw))
    span = max(L_high - L_min, 1e-18)
    frac = max(0.0, min(1.0, float(improve_frac)))
    thr = L_high - frac * span
    for gi in range(lo, hi + 1):
        if float(losses[gi]) <= thr:
            return gi
    return int(lo + int(np.argmin(Lw)))


def _pick_lr_torch_lr_finder_style(
    lr_losses: dict[float, float],
    *,
    smooth_beta: float = 0.98,
    skip_start_frac: float = 0.1,
    skip_end_frac: float = 0.1,
    lr_divisor: float = 10.0,
    elbow_improve_frac: float = 0.5,
) -> tuple[float, str, int]:
    finite = [(lr, loss) for lr, loss in lr_losses.items() if math.isfinite(loss)]
    if not finite:
        lr0 = min(lr_losses.keys())
        return lr0, "no finite mean losses; returning smallest LR key", 0
    finite.sort(key=lambda x: x[0])
    lrs = np.array([float(x[0]) for x in finite], dtype=np.float64)
    losses = np.array([float(x[1]) for x in finite], dtype=np.float64)
    n = int(lrs.shape[0])
    if n < 3:
        mid = max(0, n // 2)
        return float(lrs[mid]), "insufficient finite points for gradient rule", int(mid)

    beta = min(0.9999, max(0.0, float(smooth_beta)))
    ema = np.empty_like(losses, dtype=np.float64)
    avg = 0.0
    for i, loss in enumerate(losses):
        avg = beta * avg + (1.0 - beta) * loss
        ema[i] = avg / (1.0 - beta ** (i + 1))

    grads = np.gradient(ema, np.log(np.maximum(lrs, 1e-300)))
    i0 = int(math.floor(max(0.0, min(0.9, float(skip_start_frac))) * n))
    i1 = int(math.ceil((1.0 - max(0.0, min(0.9, float(skip_end_frac)))) * n))
    i0 = min(i0, n - 1)
    i1 = max(i0 + 1, min(i1, n))
    idx = int(i0 + np.argmin(grads[i0:i1])) if i1 > i0 else int(np.argmin(grads))

    # Detect divergence boundary on smoothed loss and keep only the stable prefix.
    div_end = n - 1
    for i in range(1, n):
        roll_min = float(np.min(ema[: i + 1]))
        prev = max(float(ema[i - 1]), 1e-12)
        if ema[i] > 1.8 * max(roll_min, 1e-12) or ema[i] > 1.35 * prev:
            div_end = i - 1
            break
    hi = max(i0, min(i1 - 1, div_end))
    lo = min(i0, hi)
    best_local = int(lo + np.argmin(losses[lo : hi + 1]))
    elbow_i = _lr_elbow_index_first_half_gain(
        losses, lo, hi, improve_frac=elbow_improve_frac
    )

    raw = float(lrs[idx])  # steepest-descent LR (classic torch-lr-finder anchor)
    loss_floor_lr = float(lrs[best_local])  # LR at minimum observed loss in stable window
    elbow_lr = float(lrs[elbow_i])
    # Prefer the more conservative (lower) LR between elbow and loss-floor — avoids
    # single-batch needles to the right of the real knee.
    best_lr = float(min(elbow_lr, loss_floor_lr))
    div = max(1.0, float(lr_divisor))
    floor_lr = raw / div
    chosen = max(best_lr, floor_lr)
    chosen = max(float(lrs[0]), min(float(lrs[-1]), chosen))
    msg = (
        f"torch_lr_finder: n={n}, smooth_beta={beta:g}, "
        f"search_window=[{i0},{i1 - 1}], stable_window=[{lo},{hi}], "
        f"steepest_lr={raw:.2e}, elbow_lr={elbow_lr:.2e} (half-gain frac={elbow_improve_frac:g}), "
        f"loss_floor_lr={loss_floor_lr:.2e}, best_pre_div={best_lr:.2e}, "
        f"lr_floor=steepest/{div:g}={floor_lr:.2e} -> lr={chosen:.2e}"
    )
    return chosen, msg, int(elbow_i)


def _env_flag(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _refinement_downward_spike_mask(
    losses_in_lr_order: list[float],
    *,
    neighbor_ratio_max: float = 0.22,
) -> list[bool]:
    """
    Mark points whose mean loss is far below **both** neighbors on a log-LR line.

    With ``batch_size=1`` mini-trains, a single lucky batch can produce a bogus sharp
    minimum; those points should not win ``argmin(loss)`` outright.
    """
    n = len(losses_in_lr_order)
    out = [False] * n
    if n < 2:
        return out
    thr = max(1e-9, min(0.45, float(neighbor_ratio_max)))
    if n == 2:
        a, b = losses_in_lr_order[0], losses_in_lr_order[1]
        if a < thr * b:
            out[0] = True
        if b < thr * a:
            out[1] = True
        return out
    for i in range(n):
        mid = losses_in_lr_order[i]
        if i == 0:
            right = losses_in_lr_order[1]
            if right > 1e-18 and mid < thr * right:
                out[i] = True
            continue
        if i == n - 1:
            left = losses_in_lr_order[i - 1]
            if left > 1e-18 and mid < thr * left:
                out[i] = True
            continue
        left, right = losses_in_lr_order[i - 1], losses_in_lr_order[i + 1]
        m = min(left, right)
        if m > 1e-18 and mid < thr * m:
            out[i] = True
    return out


def _pick_refinement_lr_with_tie_break(
    refine_lrs: list[float],
    refine_losses: dict[float, float],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    median_margin_frac: float = 0.0,
) -> tuple[Optional[int], float, float]:
    """
    Among refinement points, let ``min_v`` be the minimum finite mean loss. Keep every
    point with loss ≤ ``min_v + atol + rtol·max(|min_v|, 1e-8)`` (near-optimal band), then
    pick the **lowest LR** in that band (conservative; breaks DPO/GRPO ``0.0000`` ties).

    When ``median_margin_frac`` > 0, widen the band by at least that fraction of the
    median observed loss so tiny ``min_v`` (often noise at batch_size=1) does not shrink
    the pool to a single spurious point.
    """
    triples: list[tuple[int, float, float]] = []
    for i, lr in enumerate(refine_lrs):
        v = refine_losses.get(lr, float("inf"))
        if math.isfinite(v):
            triples.append((i, lr, v))
    if not triples:
        return None, float("inf"), float("inf")
    min_v = min(t[2] for t in triples)
    med_v = float(np.median([t[2] for t in triples]))
    margin = atol + rtol * max(abs(min_v), 1e-8)
    if median_margin_frac > 0.0 and math.isfinite(med_v):
        margin = max(margin, float(median_margin_frac) * max(abs(med_v), 1e-12))
    pool = [(i, lr, v) for i, lr, v in triples if v <= min_v + margin]
    i_best, lr_best, v_best = min(pool, key=lambda t: (t[1], t[0]))
    return i_best, v_best, lr_best


def _pick_refinement_lr_robust(
    refine_lrs: list[float],
    refine_losses: dict[float, float],
    *,
    atol: float,
    rtol: float,
    neighbor_ratio_max: float = 0.22,
    median_margin_frac: float = 0.12,
) -> tuple[float, float, str]:
    """
    Refinement LR: drop downward-spike points, then apply ``_pick_refinement_lr_with_tie_break``.

    Returns ``(lr_best, loss_at_best, note)``.
    """
    pairs = [(lr, refine_losses[lr]) for lr in refine_lrs if math.isfinite(refine_losses.get(lr, float("inf")))]
    if not pairs:
        return float("nan"), float("inf"), "no finite refinement losses"
    pairs.sort(key=lambda x: x[0])
    vals = [v for _, v in pairs]
    mask = _refinement_downward_spike_mask(vals, neighbor_ratio_max=neighbor_ratio_max)
    dropped = sum(1 for b in mask if b)
    kept = [pairs[i] for i in range(len(pairs)) if not mask[i]]
    note = ""
    if dropped and len(kept) >= 1:
        note = f"dropped {dropped} downward-spike point(s) (batch-noise guard)"
    if len(kept) < 1:
        kept = pairs
        note = "spike guard skipped (would empty)"
    k_lrs = [p[0] for p in kept]
    k_loss = {lr: v for lr, v in kept}
    idx, v_best, lr_best = _pick_refinement_lr_with_tie_break(
        k_lrs,
        k_loss,
        atol=atol,
        rtol=rtol,
        median_margin_frac=median_margin_frac,
    )
    if idx is None or not math.isfinite(v_best):
        lr_fb, v_fb = min(kept, key=lambda x: x[1])
        fb = "tie-break fallback to raw min"
        return lr_fb, v_fb, f"{note}; {fb}" if note else fb
    return lr_best, v_best, note


def _local_window_quadratic_log_lr_fit(
    lrs: np.ndarray,
    losses: np.ndarray,
    *,
    half_window: int = 2,
) -> Optional[tuple[np.ndarray, int, int, Optional[float]]]:
    """
    Fit loss ≈ a·(log lr)² + b·log lr + c on a local window around the empirical loss minimum.
    Returns ``(coeffs, lo, hi, lr_vertex)`` when the fit is convex (a > 0).
    ``lr_vertex`` is the analytical minimum exp(-b/2a) only if it lies inside the window's
    log-LR interval; otherwise ``None`` (refinement mini-train can still use the window).
    """
    n = int(lrs.shape[0])
    if n < 3:
        return None
    i_min = int(np.argmin(losses))
    lo = max(0, i_min - half_window)
    hi = min(n, i_min + half_window + 1)
    if hi - lo < 3:
        lo = max(0, n - 3)
        hi = n
    t = np.ascontiguousarray(np.log(lrs[lo:hi]), dtype=np.float64)
    y = np.ascontiguousarray(losses[lo:hi], dtype=np.float64)
    if t.shape[0] < 3:
        return None
    try:
        coeffs = np.polyfit(t, y, 2, rcond=1e-7)
    except (np.linalg.LinAlgError, ValueError):
        return None
    a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    if not all(math.isfinite(x) for x in (a, b, c)):
        return None
    if a <= 1e-14:
        return None
    t_star = -b / (2.0 * a)
    t_lo = float(np.log(lrs[lo]))
    t_hi = float(np.log(lrs[hi - 1]))
    lr_vertex: Optional[float] = None
    if t_lo <= t_star <= t_hi:
        lr_v = math.exp(t_star)
        if math.isfinite(lr_v) and lr_v > 0:
            lr_vertex = lr_v
    return coeffs, lo, hi, lr_vertex


def _print_quadratic_interp_table(
    coeffs: np.ndarray,
    lr_lo: float,
    lr_hi: float,
    *,
    steps: int,
    lr_mark: float,
) -> None:
    """Log estimated loss along the fitted parabola (log-spaced LR between lr_lo and lr_hi)."""
    if steps < 2:
        return
    a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    t_lo = math.log(lr_lo)
    t_hi = math.log(lr_hi)
    print(
        f"  [LR Finder] quadratic fit on log(lr): loss ≈ {a:.4g}·t² + {b:.4g}·t + {c:.4g}  "
        f"(t = log lr), convex (a>0)",
        flush=True,
    )
    print(
        f"  [LR Finder] interpolated loss along fit ({steps} points between {lr_lo:.2e} and {lr_hi:.2e}):",
        flush=True,
    )
    rows: list[tuple[float, float, float]] = []
    for i in range(steps):
        frac = i / (steps - 1) if steps > 1 else 0.0
        t = t_lo + frac * (t_hi - t_lo)
        lr = math.exp(t)
        est = a * t * t + b * t + c
        rows.append(
            (lr, est, abs(math.log(max(lr, 1e-300)) - math.log(max(lr_mark, 1e-300))))
        )
    j_star = min(range(len(rows)), key=lambda k: rows[k][2])
    for j, (lr, est, _) in enumerate(rows):
        mark = "  [closest to vertex]" if j == j_star else ""
        print(f"    lr={lr:.2e}  est_loss={est:.4f}{mark}", flush=True)


def _effective_quadratic_interp_steps(default: int) -> int:
    env_steps = os.environ.get("LR_FINDER_INTERP_STEPS", "").strip()
    if env_steps:
        try:
            return max(0, int(env_steps))
        except ValueError:
            pass
    return max(0, default)


def _log_uniform_lrs(lr_lo: float, lr_hi: float, steps: int) -> list[float]:
    """``steps`` LRs evenly spaced in log-space between ``lr_lo`` and ``lr_hi`` (inclusive)."""
    if steps < 2:
        return [lr_lo]
    lo = min(lr_lo, lr_hi)
    hi = max(lr_lo, lr_hi)
    t_lo = math.log(lo)
    t_hi = math.log(hi)
    out: list[float] = []
    for i in range(steps):
        frac = i / (steps - 1) if steps > 1 else 0.0
        t = t_lo + frac * (t_hi - t_lo)
        out.append(math.exp(t))
    return out


# --------------------------------------------------------------------------- #
# Micro-training kernels
# --------------------------------------------------------------------------- #

def _trainable_params(model) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


_LR_FINDER_PROBE_OPTIM_NOTICE = False


def _make_optimizer(
    model,
    lr: float,
    optimizer_name: Optional[str],
    weight_decay: float = 0.0,
):
    """
    LR mini-train probes always use ``torch.optim.AdamW``.

    Training may use PagedAdamW8bit, but that optimizer **pages state to host RAM**;
    creating it dozens of times across the LR grid spikes cgroup memory and the Linux
    OOM killer often SIGKILLs the process (``Killed`` with no Python traceback).
    """
    global _LR_FINDER_PROBE_OPTIM_NOTICE
    params = _trainable_params(model)
    key = (optimizer_name or "adamw_torch").lower().replace("-", "_")
    if key in ("paged_adamw_8bit", "pagedadamw8bit") and not _LR_FINDER_PROBE_OPTIM_NOTICE:
        print(
            "[LR Finder] Mini-train probes use torch AdamW (PagedAdamW8bit skipped here "
            "to avoid host-RAM OOM kills across many LR steps). Full training still uses "
            "your configured optimizer.",
            flush=True,
        )
        _LR_FINDER_PROBE_OPTIM_NOTICE = True
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
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
) -> float:
    """
    Run ``num_batches`` **optimizer** steps at ``lr``; return mean **post-update** loss.

    When ``gradient_accumulation_steps`` > 1, each optimizer step matches HF Trainer:
    ``loss / gas`` per micro-batch, accumulate backward, then ``clip_grad_norm`` + ``step()``.
    """
    gas = max(1, int(gradient_accumulation_steps))
    optimizer = _make_optimizer(model, lr, optimizer_name)
    try:
        trainable = _trainable_params(model)
        model.train()
        losses: list[float] = []
        batch_iter = _iter_batches_forever(dataloader)
        for _ in range(max(1, int(num_batches))):
            optimizer.zero_grad()
            last_inputs: Optional[dict[str, torch.Tensor]] = None
            for _micro in range(gas):
                batch = next(batch_iter)
                inputs = {k: v.to(device) for k, v in batch.items()}
                last_inputs = inputs
                with _bf16_autocast(device):
                    loss = model(**inputs).loss / float(gas)
                if not torch.isfinite(loss):
                    return float("inf")
                loss.backward()
            assert last_inputs is not None
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=max_grad_norm)
            optimizer.step()
            model.eval()
            with torch.no_grad():
                with _bf16_autocast(device):
                    loss_after = model(**last_inputs).loss
            model.train()
            if not torch.isfinite(loss_after):
                return float("inf")
            losses.append(loss_after.item())
        return float(np.mean(losses)) if losses else float("inf")
    finally:
        del optimizer
        gc.collect()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _synthetic_lm_batch(tokenizer, batch_size: int, seq_len: int, device: str) -> dict[str, torch.Tensor]:
    """Random integer tokens on ``device`` (same layout as tokenized mini-train)."""
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    if vocab_size <= 0:
        vocab_size = int(len(tokenizer))
    ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attn = torch.ones_like(ids)
    labels = ids.clone()
    return {"input_ids": ids, "attention_mask": attn, "labels": labels}


def _lr_finder_cap_batch_with_optimizer_probe(
    model,
    tokenizer,
    *,
    seq_len: int,
    device: str,
    optimizer_name: Optional[str],
    synthetic_b_train: int,
    probe_lrs: list[float],
    max_grad_norm: float = 1.0,
    lr_task: str = LR_TASK_INSTRUCT,
    dpo_beta: float = 0.1,
    dpo_ref_trainable_cpu: Optional[Union[dict[str, torch.Tensor], str]] = None,
    grpo_epsilon_low: float = 0.2,
    grpo_epsilon_high: float = 0.2,
    grpo_beta: float = 0.04,
) -> int:
    """
    ``find_max_batch_size`` only checks forward + backward (no optimizer states).

    LR search uses the same step pattern as ``_mini_train_mean_loss`` (Adam + ``step()`` +
    eval forward). For each **probe LR** (typically ``min_lr`` and ``max_lr`` from the
    finder grid), shrink batch from the synthetic cap until that pattern fits (OOM or
    non-finite loss). Return ``min(...)`` so the batch is safe across the LR range.
    """
    task = normalize_lr_finder_task(lr_task)
    cap = max(1, int(synthetic_b_train))
    unique_lrs = sorted(
        {float(x) for x in probe_lrs if math.isfinite(x) and float(x) > 0.0}
    )
    if not unique_lrs:
        unique_lrs = [1e-6]

    if device != "cuda" or not torch.cuda.is_available():
        return cap

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        return cap

    total_trainable_bytes = sum(p.numel() * p.element_size() for _, p in trainable)
    ckpt_path: Optional[str] = None
    cpu_snapshot: Optional[dict[str, torch.Tensor]] = None
    if total_trainable_bytes > _SNAPSHOT_DISK_THRESHOLD_BYTES:
        fd, ckpt_path = tempfile.mkstemp(prefix="lr_finder_optim_probe_", suffix=".pt")
        os.close(fd)
        with torch.no_grad():
            torch.save({n: p.detach().cpu() for n, p in trainable}, ckpt_path)
    else:
        with torch.no_grad():
            cpu_snapshot = {n: p.detach().cpu().clone() for n, p in trainable}

    def _restore_trainable_weights() -> None:
        with torch.no_grad():
            if ckpt_path is not None:
                blob = _load_trainable_checkpoint(ckpt_path)
                try:
                    for name, param in model.named_parameters():
                        if name in blob:
                            param.copy_(
                                blob[name].to(
                                    device=param.device,
                                    dtype=param.dtype,
                                    non_blocking=True,
                                )
                            )
                finally:
                    del blob
            else:
                assert cpu_snapshot is not None
                for name, param in model.named_parameters():
                    if name in cpu_snapshot:
                        param.copy_(
                            cpu_snapshot[name].to(
                                device=param.device,
                                dtype=param.dtype,
                                non_blocking=True,
                            )
                        )
        model.zero_grad(set_to_none=True)
        gc.collect()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _max_batch_one_lr(probe_lr: float) -> int:
        b = cap
        while b >= 1:
            optimizer: Any = None
            try:
                _restore_trainable_weights()
                if task == LR_TASK_DPO:
                    inputs = synthetic_dpo_batch(tokenizer, b, seq_len, device)
                elif task == LR_TASK_GRPO:
                    inputs = synthetic_grpo_batch(tokenizer, b, seq_len, device)
                else:
                    inputs = _synthetic_lm_batch(tokenizer, b, seq_len, device)
                optimizer = _make_optimizer(model, probe_lr, optimizer_name)
                trainable_params = _trainable_params(model)
                model.train()
                optimizer.zero_grad()
                with _bf16_autocast(device):
                    if task == LR_TASK_DPO:
                        loss = dpo_sigmoid_loss(
                            model,
                            inputs,
                            beta=dpo_beta,
                            ref_trainable_cpu=dpo_ref_trainable_cpu,
                        )
                    elif task == LR_TASK_GRPO:
                        loss = grpo_style_clipped_loss(
                            model,
                            inputs,
                            epsilon_low=grpo_epsilon_low,
                            epsilon_high=grpo_epsilon_high,
                            beta=grpo_beta,
                        )
                    else:
                        loss = model(**inputs).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite loss in LR finder optimizer probe")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    with _bf16_autocast(device):
                        if task == LR_TASK_DPO:
                            loss_after = dpo_sigmoid_loss(
                                model,
                                inputs,
                                beta=dpo_beta,
                                ref_trainable_cpu=dpo_ref_trainable_cpu,
                            )
                        elif task == LR_TASK_GRPO:
                            loss_after = grpo_style_clipped_loss(
                                model,
                                inputs,
                                epsilon_low=grpo_epsilon_low,
                                epsilon_high=grpo_epsilon_high,
                                beta=grpo_beta,
                            )
                        else:
                            loss_after = model(**inputs).loss
                model.train()
                if not torch.isfinite(loss_after):
                    raise RuntimeError("non-finite post-step loss in LR finder optimizer probe")
            except RuntimeError as exc:
                _restore_trainable_weights()
                msg = str(exc).lower()
                if "out of memory" in msg or "non-finite" in msg:
                    if optimizer is not None:
                        del optimizer
                        optimizer = None
                    gc.collect()
                    if device == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if b <= 1:
                        return 1
                    b //= 2
                    continue
                raise
            finally:
                if optimizer is not None:
                    del optimizer
                    gc.collect()
                if device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            _restore_trainable_weights()
            return b

        return 1

    try:
        per: list[tuple[float, int]] = []
        for plr in unique_lrs:
            b_m = _max_batch_one_lr(plr)
            per.append((plr, b_m))
            print(
                f"[LR Finder] Optimizer probe at lr={plr:.2e}: max batch = {b_m}",
                flush=True,
            )
        out = min(b for _, b in per)
        if len(per) > 1:
            print(
                f"[LR Finder] Min batch over probe LRs → B_train = {out}",
                flush=True,
            )
        return out
    finally:
        if cpu_snapshot is not None:
            del cpu_snapshot
        if ckpt_path is not None:
            try:
                os.remove(ckpt_path)
            except OSError:
                pass
        gc.collect()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_trainable_checkpoint(path: str) -> dict:
    """
    Prefer ``mmap=True`` when supported so restoring large FT checkpoints does not
    allocate a full second copy of all tensors in host RAM at once.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (TypeError, OSError):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")


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
    quadratic_interp_steps: int = 10,
    collate_fn: Optional[Callable] = None,
    max_grad_norm: float = 1.0,
    shuffle_train_batches: bool = False,
    dataloader_seed: int = 42,
    mini_train_fn: Optional[Callable[..., float]] = None,
    refinement_tie_atol: float = 1e-5,
    refinement_tie_rtol: float = 1e-4,
    gradient_accumulation_steps: int = 1,
    samples_per_lr: int = 0,
    lr_pick_mode: str = "quadratic",
    pick_trim_low_lr_frac: float = 0.2,
    pick_explosion_rel_rolling_min: float = 2.5,
    pick_explosion_step_ratio: float = 1.5,
    pick_segment_pick_frac: float = 0.4,
) -> tuple[float, int]:
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    total_trainable_bytes = sum(p.numel() * p.element_size() for _, p in trainable)

    ckpt_path: Optional[str] = None
    cpu_snapshot: Optional[dict[str, torch.Tensor]] = None

    if total_trainable_bytes > _SNAPSHOT_DISK_THRESHOLD_BYTES:
        fd, ckpt_path = tempfile.mkstemp(prefix="lr_finder_w_", suffix=".pt")
        os.close(fd)
        with torch.no_grad():
            torch.save({n: p.detach().cpu() for n, p in trainable}, ckpt_path)
        gc.collect()
        print(
            f"[LR Finder] Trainable weights ≈{total_trainable_bytes / 1e9:.2f} GB → "
            f"disk snapshot (saves host RAM vs. keeping a full CPU copy): {ckpt_path}",
            flush=True,
        )
    else:
        with torch.no_grad():
            cpu_snapshot = {n: p.detach().cpu().clone() for n, p in trainable}

    def _restore_trainable_weights() -> None:
        with torch.no_grad():
            if ckpt_path is not None:
                blob = _load_trainable_checkpoint(ckpt_path)
                try:
                    for name, param in model.named_parameters():
                        if name in blob:
                            param.copy_(
                                blob[name].to(
                                    device=param.device,
                                    dtype=param.dtype,
                                    non_blocking=True,
                                )
                            )
                finally:
                    del blob
            else:
                assert cpu_snapshot is not None
                for name, param in model.named_parameters():
                    if name in cpu_snapshot:
                        param.copy_(
                            cpu_snapshot[name].to(
                                device=param.device,
                                dtype=param.dtype,
                                non_blocking=True,
                            )
                        )
        model.zero_grad(set_to_none=True)
        gc.collect()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _invoke_mini_train(
        m,
        dl: DataLoader,
        lr: float,
        dev: str,
        on: Optional[str],
        gn: float,
    ) -> float:
        _gas = max(1, int(gradient_accumulation_steps))
        bs = int(getattr(dl, "batch_size", None) or 1)
        nb = _effective_mini_train_batches(
            mini_train_batches, samples_per_lr, bs, _gas
        )
        if mini_train_fn is not None:
            return float(
                mini_train_fn(
                    m,
                    dl,
                    lr,
                    dev,
                    on,
                    num_batches=nb,
                    max_grad_norm=gn,
                    gradient_accumulation_steps=_gas,
                )
            )
        return _mini_train_mean_loss(
            m,
            dl,
            lr,
            dev,
            on,
            num_batches=nb,
            max_grad_norm=gn,
            gradient_accumulation_steps=_gas,
        )

    start_batch = max(1, safe_batch)
    batch_size = start_batch

    try:
        while batch_size >= 1:
            try:
                lr_losses: dict[float, float] = {}
                if shuffle_train_batches:
                    _gen = torch.Generator()
                    _gen.manual_seed(int(dataloader_seed))
                    dl = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        generator=_gen,
                        collate_fn=collate_fn,
                        num_workers=0,
                        pin_memory=False,
                    )
                else:
                    dl = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=collate_fn,
                        num_workers=0,
                        pin_memory=False,
                    )
                _gas_probe = max(1, int(gradient_accumulation_steps))
                _nb_eff = _effective_mini_train_batches(
                    mini_train_batches, samples_per_lr, batch_size, _gas_probe
                )
                _approx_seq = _nb_eff * batch_size * _gas_probe
                print(
                    f"  [LR Finder] probe batch_size={batch_size}  "
                    f"optimizer_steps/LR={_nb_eff}  "
                    f"(mini_train_batches≥{mini_train_batches}"
                    + (
                        f", samples_per_lr≥{samples_per_lr} → ≈{_approx_seq} seqs/LR)"
                        if samples_per_lr > 0
                        else ")"
                    ),
                    flush=True,
                )
                _mode_disp = str(lr_pick_mode or "quadratic").strip().lower()
                print(
                    f"  [LR Finder] lr_pick_mode={_mode_disp}",
                    flush=True,
                )
                for lr in lr_candidates:
                    _restore_trainable_weights()
                    print(f"  [LR Finder] mini-train probe LR={lr:.2e} …", flush=True)
                    try:
                        avg_loss = _invoke_mini_train(
                            model,
                            dl,
                            lr,
                            device,
                            optimizer_name,
                            max_grad_norm,
                        )
                        lr_losses[lr] = avg_loss
                        print(f"    mean loss = {avg_loss:.6g}", flush=True)
                    except RuntimeError as exc:
                        if "out of memory" in str(exc).lower():
                            lr_losses[lr] = float("inf")
                            print(
                                f"    [LR Finder] OOM at LR={lr:.2e}; "
                                "marking invalid, clearing cache, continuing.",
                                flush=True,
                            )
                            _restore_trainable_weights()
                            gc.collect()
                            if device == "cuda" and torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                        raise
                    gc.collect()
                    if device == "cuda" and torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()

                # --- LR selection: convex quadratic window + optional refinement mini-train ---
                finite_pairs = [
                    (lr, loss) for lr, loss in lr_losses.items() if math.isfinite(loss)
                ]
                if not finite_pairs:
                    return min(lr_losses.keys()), batch_size
                finite_pairs.sort(key=lambda x: x[0])
                lrs_arr = np.array([x[0] for x in finite_pairs], dtype=np.float64)
                losses_arr = np.array([x[1] for x in finite_pairs], dtype=np.float64)

                if _env_flag("LR_FINDER_USE_PEAK_RULE_ONLY"):
                    best_lr = _pick_lr_peak_edge_of_stability(
                        lr_losses, rel_slack=peak_rel_slack
                    )
                    return best_lr, batch_size

                _pick_mode = "torch_lr_finder"
                try:
                    _elbow_frac = float(os.environ.get("LR_FINDER_ELBOW_IMPROVE_FRAC", "0.5"))
                except ValueError:
                    _elbow_frac = 0.5
                _elbow_frac = max(0.0, min(1.0, _elbow_frac))
                anchor_lr, pick_msg, refine_center_idx = _pick_lr_torch_lr_finder_style(
                    lr_losses,
                    elbow_improve_frac=_elbow_frac,
                )
                print(f"  [LR Finder] {pick_msg}", flush=True)

                # Second pass: repeat mini-train sweep in the best stable coarse range.
                div_end = len(losses_arr) - 1
                for i in range(1, len(losses_arr)):
                    roll_min = float(np.min(losses_arr[: i + 1]))
                    prev = max(float(losses_arr[i - 1]), 1e-12)
                    if losses_arr[i] > 1.8 * max(roll_min, 1e-12) or losses_arr[i] > 1.35 * prev:
                        div_end = i - 1
                        break
                div_end = max(0, div_end)
                # Center refinement on the elbow index (half-gain), not raw argmin — avoids
                # mini-train grids locked around a batch-noise loss needle.
                best_idx = int(max(0, min(refine_center_idx, div_end)))
                lo_idx = max(0, best_idx - 1)
                hi_idx = min(len(lrs_arr) - 1, best_idx + 1)
                lr_lo = float(lrs_arr[lo_idx])
                lr_hi = float(lrs_arr[hi_idx])
                if not (lr_hi > lr_lo):
                    lr_lo = max(float(lrs_arr[0]), float(anchor_lr) / 3.0)
                    lr_hi = min(float(lrs_arr[-1]), float(anchor_lr) * 3.0)
                    if lr_hi <= lr_lo:
                        lr_hi = max(lr_lo * 1.01, lr_lo + 1e-12)
                refine_lrs = _log_uniform_lrs(lr_lo, lr_hi, 5)
                print(
                    f"  [LR Finder] refinement mini-train in stable range "
                    f"[{lr_lo:.2e}, {lr_hi:.2e}] ({len(refine_lrs)} points) …",
                    flush=True,
                )
                refine_losses: dict[float, float] = {}
                for lr in refine_lrs:
                    _restore_trainable_weights()
                    try:
                        avg_loss = _invoke_mini_train(
                            model,
                            dl,
                            lr,
                            device,
                            optimizer_name,
                            max_grad_norm,
                        )
                        refine_losses[lr] = avg_loss
                    except RuntimeError as exc:
                        if "out of memory" in str(exc).lower():
                            refine_losses[lr] = float("inf")
                            _restore_trainable_weights()
                            gc.collect()
                            if device == "cuda" and torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                        raise
                    gc.collect()
                    if device == "cuda" and torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    print(f"    refine lr={lr:.2e}  mean loss={refine_losses[lr]:.6g}", flush=True)

                finite_ref = [(lr, v) for lr, v in refine_losses.items() if math.isfinite(v)]
                if finite_ref:
                    best_lr, _ref_v, ref_note = _pick_refinement_lr_robust(
                        refine_lrs,
                        refine_losses,
                        atol=refinement_tie_atol,
                        rtol=refinement_tie_rtol,
                    )
                    extra = f"  ({ref_note})" if ref_note else ""
                    print(
                        f"  [LR Finder] selected LR (torch coarse + stable-range refinement): "
                        f"{best_lr:.2e}{extra}",
                        flush=True,
                    )
                    return best_lr, batch_size

                print(
                    f"  [LR Finder] refinement failed; fallback to torch anchor LR: {anchor_lr:.2e}",
                    flush=True,
                )
                return anchor_lr, batch_size

                fit = _local_window_quadratic_log_lr_fit(lrs_arr, losses_arr, half_window=2)
                if fit is None:
                    print(
                        "  [LR Finder] convex quadratic fit unavailable; "
                        "using peak-edge-of-stability rule.",
                        flush=True,
                    )
                    best_lr = _pick_lr_peak_edge_of_stability(
                        lr_losses, rel_slack=peak_rel_slack
                    )
                    return best_lr, batch_size

                coeffs, lo, hi, lr_vertex = fit
                lr_lo_w = float(lrs_arr[lo])
                lr_hi_w = float(lrs_arr[hi - 1])
                ref_steps = _effective_quadratic_interp_steps(quadratic_interp_steps)
                vertex_only = _env_flag("LR_FINDER_USE_QUADRATIC_VERTEX_ONLY")

                # Default: run mini-train again on ``ref_steps`` log-spaced LRs in the window; pick lowest loss.
                if ref_steps >= 2 and not vertex_only:
                    a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
                    print(
                        f"  [LR Finder] quadratic fit on log(lr): loss ≈ {a:.4g}·t² + {b:.4g}·t + {c:.4g}  "
                        f"(t = log lr), convex (a>0)",
                        flush=True,
                    )
                    refine_lrs = _log_uniform_lrs(lr_lo_w, lr_hi_w, ref_steps)
                    print(
                        f"  [LR Finder] refinement mini-train ({ref_steps} log-spaced LRs between "
                        f"{lr_lo_w:.2e} and {lr_hi_w:.2e}) …",
                        flush=True,
                    )
                    refine_losses: dict[float, float] = {}
                    for lr in refine_lrs:
                        _restore_trainable_weights()
                        try:
                            avg_loss = _invoke_mini_train(
                                model,
                                dl,
                                lr,
                                device,
                                optimizer_name,
                                max_grad_norm,
                            )
                            refine_losses[lr] = avg_loss
                        except RuntimeError as exc:
                            if "out of memory" in str(exc).lower():
                                refine_losses[lr] = float("inf")
                                _restore_trainable_weights()
                                gc.collect()
                                if device == "cuda" and torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                continue
                            raise
                        gc.collect()
                        if device == "cuda" and torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()

                    # Pick lowest mean loss; ties within atol/rtol → **lower LR** (conservative).
                    idx_best, best_val, _best_lr_chosen = _pick_refinement_lr_with_tie_break(
                        refine_lrs,
                        refine_losses,
                        atol=refinement_tie_atol,
                        rtol=refinement_tie_rtol,
                        median_margin_frac=0.12,
                    )
                    if idx_best is not None:
                        best_lr = refine_lrs[idx_best]
                        print(
                            f"  [LR Finder] refinement mean loss ({ref_steps} points, "
                            f"same optimizer_steps/LR as coarse probe; ties → lower LR):",
                            flush=True,
                        )
                        for i, lr in enumerate(refine_lrs):
                            v = refine_losses[lr]
                            loss_s = f"{v:.6g}" if math.isfinite(v) else "inf"
                            mark = "  [best mini-train loss]" if i == idx_best else ""
                            print(f"    lr={lr:.2e}  mean_loss={loss_s}{mark}", flush=True)
                        print(
                            f"  [LR Finder] selected LR from refinement mini-train: {best_lr:.2e}",
                            flush=True,
                        )
                        return best_lr, batch_size

                    print(
                        "  [LR Finder] all refinement mini-trains failed; "
                        "using peak-edge-of-stability rule.",
                        flush=True,
                    )
                    best_lr = _pick_lr_peak_edge_of_stability(
                        lr_losses, rel_slack=peak_rel_slack
                    )
                    return best_lr, batch_size

                # Vertex-only / no refinement table: analytical vertex or peak fallback.
                if lr_vertex is not None:
                    lr_mark = lr_vertex
                    if ref_steps >= 2:
                        _print_quadratic_interp_table(
                            coeffs,
                            lr_lo_w,
                            lr_hi_w,
                            steps=ref_steps,
                            lr_mark=lr_mark,
                        )
                    print(
                        f"  [LR Finder] selected LR from convex quadratic vertex "
                        f"(window [{lr_lo_w:.2e}, {lr_hi_w:.2e}]): {lr_vertex:.2e}",
                        flush=True,
                    )
                    return lr_vertex, batch_size

                print(
                    "  [LR Finder] analytical vertex outside window; "
                    "using peak-edge-of-stability rule.",
                    flush=True,
                )
                best_lr = _pick_lr_peak_edge_of_stability(
                    lr_losses, rel_slack=peak_rel_slack
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
    finally:
        if cpu_snapshot is not None:
            del cpu_snapshot
        if ckpt_path is not None:
            try:
                os.remove(ckpt_path)
            except OSError:
                pass
        gc.collect()


# --------------------------------------------------------------------------- #
# Dataset loading — full ``train_tokenized_*.json`` (same file as real training)
# --------------------------------------------------------------------------- #

def _lr_finder_tight_headroom_after_kill(dataset_type_dict: dict) -> bool:
    """0.6 headroom when resubmitting after OOM kill (tournaments): env or train_request flag."""
    env = os.environ.get("LR_FINDER_TIGHT_BATCH_HEADROOM", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    return _config_bool_or_default(
        dataset_type_dict.get("lr_finder_tight_after_job_kill"),
        False,
    )


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


def _load_full_tokenized_train_dataset(tokenized_path: str) -> tuple[Dataset, str, int]:
    """
    Load **every** example from ``train_tokenized_*.json`` — the same on-disk artifact
    produced before real training (no separate probe subset).
    """
    with open(tokenized_path, "r", encoding="utf-8") as f:
        raw_list = json.load(f)
    if not isinstance(raw_list, list):
        raise ValueError(f"Tokenized dataset must be a JSON list: {tokenized_path}")
    data: list[dict] = []
    for ex in raw_list:
        if isinstance(ex, dict):
            norm = _normalize_tokenized_row(ex)
            if norm is not None:
                data.append(norm)
    if not data:
        return Dataset.from_list([]), TOKENIZED_BATCH_KEY, 0
    total_tokens = sum(len(r["input_ids"]) for r in data)
    return Dataset.from_list(data), TOKENIZED_BATCH_KEY, total_tokens


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
    mini_train_batches: int = 20,
    seq_len: int = 1024,
    lora_threshold: Optional[int] = None,
    lora_r: int = _DEFAULT_LORA_R,
    lora_alpha: int = _DEFAULT_LORA_ALPHA,
    lora_dropout: float = _DEFAULT_LORA_DROPOUT,
    optimizer_name: Optional[str] = None,
    lr_sample_seed: int = 42,
    batch_headroom: float = 0.8,
    peak_rel_slack: float = 0.28,
    max_lr_probe_batch: Optional[int] = None,
) -> Optional[dict]:
    if not tokenized_dataset_path or not os.path.isfile(tokenized_dataset_path):
        print(
            "[LR Finder] dataset_path missing or not a file (see lr_finder_task / tokenize output).",
            flush=True,
        )
        return None

    min_lr = float(min_lr)
    max_lr = float(max_lr)
    if min_lr > max_lr:
        min_lr, max_lr = max_lr, min_lr

    if "lr_finder_use_lora" in dataset_type_dict:
        use_lora = bool(dataset_type_dict["lr_finder_use_lora"])
    else:
        use_lora = (
            lora_threshold is not None
            and num_params is not None
            and num_params >= lora_threshold
        )
    try:
        _lora_r = int(dataset_type_dict.get("lr_finder_lora_r", lora_r))
    except (TypeError, ValueError):
        _lora_r = int(lora_r)
    try:
        _lora_alpha = int(dataset_type_dict.get("lr_finder_lora_alpha", lora_alpha))
    except (TypeError, ValueError):
        _lora_alpha = int(lora_alpha)
    try:
        _lora_do = float(dataset_type_dict.get("lr_finder_lora_dropout", lora_dropout))
    except (TypeError, ValueError):
        _lora_do = float(lora_dropout)
    try:
        _max_gn = float(dataset_type_dict.get("lr_finder_max_grad_norm", 1.0))
    except (TypeError, ValueError):
        _max_gn = 1.0
    if not math.isfinite(_max_gn) or _max_gn <= 0:
        _max_gn = 1.0
    _gc_train = _config_bool_or_default(
        dataset_type_dict.get("lr_finder_gradient_checkpointing"), True
    )

    _points = max(
        2,
        min(12, int(dataset_type_dict.get("lr_finder_lr_probe_points", lr_probe_points))),
    )
    _mini_b = max(
        1,
        min(6, int(dataset_type_dict.get("lr_finder_mini_train_batches", mini_train_batches))),
    )
    try:
        _gas_lr = int(dataset_type_dict.get("lr_finder_gradient_accumulation_steps", 1))
    except (TypeError, ValueError):
        _gas_lr = 1
    _gas_lr = max(1, _gas_lr)
    try:
        _samples_plr = int(dataset_type_dict.get("lr_finder_samples_per_lr", 24))
    except (TypeError, ValueError):
        _samples_plr = 24
    _samples_plr = max(0, min(24, _samples_plr))
    _lr_pick_mode = "torch_lr_finder"
    try:
        _pick_trim = float(dataset_type_dict.get("lr_finder_pick_trim_low_lr_frac", 0.2))
    except (TypeError, ValueError):
        _pick_trim = 0.2
    _pick_trim = max(0.0, min(0.95, _pick_trim))
    try:
        _pick_expl_rel = float(
            dataset_type_dict.get("lr_finder_pick_explosion_rel_rolling_min", 2.5)
        )
    except (TypeError, ValueError):
        _pick_expl_rel = 2.5
    _pick_expl_rel = max(1.01, _pick_expl_rel)
    try:
        _pick_expl_step = float(
            dataset_type_dict.get("lr_finder_pick_explosion_step_ratio", 1.5)
        )
    except (TypeError, ValueError):
        _pick_expl_step = 1.5
    _pick_expl_step = max(1.01, _pick_expl_step)
    try:
        _pick_seg_frac = float(
            dataset_type_dict.get("lr_finder_pick_segment_pick_frac", 0.4)
        )
    except (TypeError, ValueError):
        _pick_seg_frac = 0.4
    _pick_seg_frac = max(0.0, min(1.0, _pick_seg_frac))
    _peak_sl = float(dataset_type_dict.get("lr_finder_peak_rel_slack", peak_rel_slack))
    _peak_sl = max(0.0, min(3.0, _peak_sl))
    try:
        _qinterp = int(dataset_type_dict.get("lr_finder_quadratic_interp_steps", 10))
    except (TypeError, ValueError):
        _qinterp = 10
    _qinterp = max(0, _qinterp)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _lr_task = normalize_lr_finder_task(dataset_type_dict.get("lr_finder_task", LR_TASK_INSTRUCT))
    try:
        _dpo_beta = float(
            dataset_type_dict.get("lr_finder_dpo_beta", dataset_type_dict.get("beta", 0.1))
        )
    except (TypeError, ValueError):
        _dpo_beta = 0.1
    try:
        _grpo_beta = float(dataset_type_dict.get("lr_finder_grpo_beta", 0.04))
    except (TypeError, ValueError):
        _grpo_beta = 0.04
    try:
        _geps_lo = float(dataset_type_dict.get("lr_finder_grpo_epsilon_low", 0.2))
    except (TypeError, ValueError):
        _geps_lo = 0.2
    try:
        _geps_hi = float(dataset_type_dict.get("lr_finder_grpo_epsilon_high", 0.2))
    except (TypeError, ValueError):
        _geps_hi = 0.2
    _grpo_beta_probe = 0.0 if (_lr_task == LR_TASK_GRPO and not use_lora) else _grpo_beta

    _opt_disp = optimizer_name or "adamw_torch"
    if _lr_task == LR_TASK_DPO:
        _probe_tag = "DPO_sigmoid"
    elif _lr_task == LR_TASK_GRPO:
        _probe_tag = "GRPO_clipped_surrogate"
    else:
        _probe_tag = "SFT_CE"
    _probe_seq_cfg = dataset_type_dict.get("lr_finder_probe_seq_len")
    try:
        _probe_cap = (
            int(_probe_seq_cfg)
            if _probe_seq_cfg is not None
            else max(1, int(seq_len))
        )
    except (TypeError, ValueError):
        _probe_cap = max(1, int(seq_len))
    _probe_cap = max(1, _probe_cap)

    print(
        f"[LR Finder] model={model_id}  mini_train={_probe_tag}  params={num_params}  lora={use_lora}"
        + (f"  lora_r={_lora_r}" if use_lora else "")
        + f"  optim={_opt_disp}  task={_lr_task}  lr_probe_points={_points}"
        + f"  mini_train_batches={_mini_b}"
        + f"  samples_per_lr={_samples_plr}"
        + f"  lr_pick_mode={_lr_pick_mode}"
        + f"  grad_accum_steps={_gas_lr} (match training GAS in run_config)"
        + f"  peak_rel_slack={_peak_sl:g}"
        + f"  quadratic_interp_steps={_qinterp}"
        + f"  max_grad_norm={_max_gn:g}",
        flush=True,
    )
    if _lr_task == LR_TASK_GRPO and not use_lora:
        print(
            "[LR Finder] GRPO probe without LoRA: reference KL uses the same weights as policy "
            "(degenerate); KL coefficient forced to 0.0 for this search.",
            flush=True,
        )

    if _lr_task == LR_TASK_DPO and not use_lora:
        print(
            "[LR Finder] DPO without LoRA: using a frozen CPU/disk snapshot of trainable weights "
            "as the reference policy π_ref (same idea as a separate ref model, but weight-replay).",
            flush=True,
        )

    dpo_ref_trainable: Optional[Union[dict[str, torch.Tensor], str]] = None
    dpo_ref_ckpt_path: Optional[str] = None

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
        with warnings.catch_warnings():
            # Some model repos ship generation configs with sampling params while
            # do_sample=False; this is harmless for training/LR probe but noisy.
            warnings.filterwarnings(
                "ignore",
                message=r"`do_sample` is set to `False`.*`temperature` is set",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"`do_sample` is set to `False`.*`top_p` is set",
                category=UserWarning,
            )
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
                r=_lora_r,
                lora_alpha=_lora_alpha,
                target_modules=target_modules,
                lora_dropout=_lora_do,
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
        if _gc_train:
            model.gradient_checkpointing_enable()
        else:
            try:
                model.gradient_checkpointing_disable()
            except AttributeError:
                pass

        # ------------------------------------------------------------------ #
        # DPO full fine-tune: π_ref is a frozen snapshot of initial trainable weights.
        # ------------------------------------------------------------------ #
        if _lr_task == LR_TASK_DPO and not use_lora:
            trainable_rows = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            if not trainable_rows:
                print(
                    "[LR Finder] DPO without LoRA but no trainable parameters; cannot define π_ref. "
                    "Skipping LR finder.",
                    flush=True,
                )
                return None
            total_trainable_bytes = sum(p.numel() * p.element_size() for _, p in trainable_rows)
            if total_trainable_bytes > _SNAPSHOT_DISK_THRESHOLD_BYTES:
                fd, dpo_ref_ckpt_path = tempfile.mkstemp(prefix="lr_finder_dpo_ref_", suffix=".pt")
                os.close(fd)
                with torch.no_grad():
                    torch.save({n: p.detach().cpu() for n, p in trainable_rows}, dpo_ref_ckpt_path)
                dpo_ref_trainable = dpo_ref_ckpt_path
                gc.collect()
                print(
                    f"[LR Finder] DPO π_ref snapshot ≈{total_trainable_bytes / 1e9:.2f} GB trainable "
                    f"→ disk mmap checkpoint: {dpo_ref_ckpt_path}",
                    flush=True,
                )
            else:
                with torch.no_grad():
                    dpo_ref_trainable = {n: p.detach().cpu().clone() for n, p in trainable_rows}
                print(
                    "[LR Finder] DPO π_ref snapshot: in-memory CPU copy of trainable weights "
                    f"(≈{total_trainable_bytes / 1e6:.1f} MB).",
                    flush=True,
                )

        # ------------------------------------------------------------------ #
        # Probe seq cap: synthetic search + collate pad to min(train seq, lr_finder_probe_seq_len).
        # ------------------------------------------------------------------ #
        effective_probe_seq = max(1, min(int(seq_len), _probe_cap))
        if effective_probe_seq < int(seq_len):
            print(
                f"[LR Finder] Probe sequence length capped: train_seq_len={seq_len} → "
                f"probe_seq_len={effective_probe_seq} (lr_finder_probe_seq_len={_probe_cap})",
                flush=True,
            )

        # ------------------------------------------------------------------ #
        # Find B_max (synthetic), then B_train = headroom × B_max (default 0.8).
        # Use 0.6 only when retrying after a killed job (OOM): see _lr_finder_tight_headroom_after_kill.
        # ------------------------------------------------------------------ #
        _base_hr = float(dataset_type_dict.get("lr_finder_batch_headroom", batch_headroom))
        if _lr_finder_tight_headroom_after_kill(dataset_type_dict):
            _hr = 0.6
            print(
                "[LR Finder] Tight batch headroom 0.6 (post-kill retry: "
                "lr_finder_tight_after_job_kill or LR_FINDER_TIGHT_BATCH_HEADROOM=1).",
                flush=True,
            )
        else:
            _hr = _base_hr
        _hr = _hr if 0 < _hr <= 1.0 else 0.8
        _bcap_raw = dataset_type_dict.get("lr_finder_b_train_cap", 0)
        try:
            _b_train_cap = int(_bcap_raw) if _bcap_raw is not None else 0
        except (TypeError, ValueError):
            _b_train_cap = 0
        if _b_train_cap <= 0:
            _b_train_cap = None
        _grid_lr_lo = min(float(min_lr), float(max_lr))
        _grid_lr_hi = max(float(min_lr), float(max_lr))
        print(
            "[LR Finder] Probing max batch size (synthetic seq_len=probe), "
            f"then optimizer fit at grid lr ∈ {{{_grid_lr_lo:.2e}, {_grid_lr_hi:.2e}}} …",
            flush=True,
        )
        b_max, b_train_syn = find_max_batch_size(
            model,
            tokenizer,
            seq_len=effective_probe_seq,
            device=device,
            headroom=_hr,
            b_train_cap=_b_train_cap,
            lr_task=_lr_task,
            dpo_beta=_dpo_beta,
            dpo_ref_trainable_cpu=dpo_ref_trainable,
            grpo_epsilon_low=_geps_lo,
            grpo_epsilon_high=_geps_hi,
            grpo_beta=_grpo_beta_probe,
        )
        _probe_lrs = (
            [_grid_lr_lo]
            if _grid_lr_lo == _grid_lr_hi
            else [_grid_lr_lo, _grid_lr_hi]
        )
        b_train = _lr_finder_cap_batch_with_optimizer_probe(
            model,
            tokenizer,
            seq_len=effective_probe_seq,
            device=device,
            optimizer_name=optimizer_name,
            synthetic_b_train=b_train_syn,
            probe_lrs=_probe_lrs,
            max_grad_norm=_max_gn,
            lr_task=_lr_task,
            dpo_beta=_dpo_beta,
            dpo_ref_trainable_cpu=dpo_ref_trainable,
            grpo_epsilon_low=_geps_lo,
            grpo_epsilon_high=_geps_hi,
            grpo_beta=_grpo_beta_probe,
        )
        _bmsg = f"headroom={_hr:g}"
        if _b_train_cap:
            _bmsg += f"  B_train_cap={_b_train_cap}"
        if b_train < b_train_syn:
            print(
                f"[LR Finder] Optimizer + mini-train probe capped batch: "
                f"synthetic B_train={b_train_syn} → {b_train}",
                flush=True,
            )
        print(
            f"[LR Finder] B_max={b_max}  B_train={b_train}  ({_bmsg})",
            flush=True,
        )
        safe_batch = b_train
        _cap_bs: Optional[int] = None
        if max_lr_probe_batch is not None:
            try:
                _cap_bs = int(max_lr_probe_batch)
            except (TypeError, ValueError):
                _cap_bs = None
            if _cap_bs is not None and _cap_bs >= 1:
                if safe_batch > _cap_bs:
                    print(
                        "[LR Finder] Capping LR mini-train batch by run_config batch_size="
                        f"{_cap_bs} (synthetic B_train={b_train} → safe_batch={_cap_bs})",
                        flush=True,
                    )
                safe_batch = min(safe_batch, _cap_bs)

        _sseed = int(dataset_type_dict.get("lr_finder_sample_seed", lr_sample_seed))

        print(
            f"[LR Finder] Loading training data for mini-train: {tokenized_dataset_path} …",
            flush=True,
        )
        if _lr_task == LR_TASK_INSTRUCT:
            sample_ds, _, probe_tokens = _load_full_tokenized_train_dataset(tokenized_dataset_path)
            collate_fn = _make_tokenized_collate_fn(tokenizer, effective_probe_seq)
            _ds_note = "train_tokenized JSON (SFT CE)"
        elif _lr_task == LR_TASK_DPO:
            sample_ds = build_dpo_preference_dataset(
                tokenized_dataset_path, dataset_type_dict, tokenizer
            )
            _pad_id = int(tokenizer.pad_token_id or 0)
            collate_fn = _PreferenceCollator(
                pad_token_id=_pad_id,
                max_length=effective_probe_seq,
                truncation_mode="keep_start",
            )
            probe_tokens = 0
            for _i in range(len(sample_ds)):
                ex = sample_ds[_i]
                probe_tokens += len(ex["prompt_ids"]) + len(ex["chosen_ids"]) + len(
                    ex["rejected_ids"]
                )
            _ds_note = "dpo_train JSON (TRL-style DPO loss)"
        else:
            try:
                _mcl_cfg = int(
                    dataset_type_dict.get(
                        "lr_finder_grpo_max_completion_length",
                        dataset_type_dict.get("max_completion_length", 256),
                    )
                )
            except (TypeError, ValueError):
                _mcl_cfg = 256
            _mcl = max(1, min(_mcl_cfg, 128))
            _mpl = max(1, int(effective_probe_seq) - _mcl)
            sample_ds = build_grpo_teacher_forced_dataset(
                tokenized_dataset_path,
                dataset_type_dict,
                tokenizer,
                max_prompt_tokens=_mpl,
                completion_len=_mcl,
            )
            _pad_id = int(tokenizer.pad_token_id or 0)
            collate_fn = lambda batch: _grpo_pad_batch_dict(batch, _pad_id)
            probe_tokens = (_mpl + _mcl) * len(sample_ds)
            _ds_note = "grpo_train JSON (clipped surrogate + KL, teacher-forced completions)"

        print(
            f"[LR Finder] Dataset rows: {len(sample_ds)}  ~{probe_tokens} tokens  "
            f"({_ds_note}; probe_seq_cap={effective_probe_seq})",
            flush=True,
        )

        _mini_train_fn: Optional[Callable[..., float]] = None
        if _lr_task != LR_TASK_INSTRUCT:
            def _mini_train_fn(
                m,
                dl: DataLoader,
                lr: float,
                dev: str,
                on: Optional[str],
                *,
                num_batches: int,
                max_grad_norm: float,
                gradient_accumulation_steps: int = 1,
            ) -> float:
                return mini_train_mean_loss_for_task(
                    _lr_task,
                    m,
                    dl,
                    lr,
                    dev,
                    on,
                    num_batches=num_batches,
                    max_grad_norm=max_grad_norm,
                    dpo_beta=_dpo_beta,
                    grpo_epsilon_low=_geps_lo,
                    grpo_epsilon_high=_geps_hi,
                    grpo_beta=_grpo_beta_probe,
                    _make_optimizer=_make_optimizer,
                    _trainable_params=_trainable_params,
                    _bf16_autocast=_bf16_autocast,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    dpo_ref_trainable_cpu=dpo_ref_trainable,
                )

        if len(sample_ds) == 0:
            print("[LR Finder] Empty tokenized train set; cannot run LR search.", flush=True)
            return None

        lr_candidates = get_lr_candidates(min_lr, max_lr, _points)
        print(
            f"[LR Finder] Mini-train LR grid: {_points} candidates  {min_lr:.2e} → {max_lr:.2e}",
            flush=True,
        )
        try:
            _tie_atol = float(dataset_type_dict.get("lr_finder_refinement_tie_atol", 1e-5))
        except (TypeError, ValueError):
            _tie_atol = 1e-5
        try:
            _tie_rtol = float(dataset_type_dict.get("lr_finder_refinement_tie_rtol", 1e-4))
        except (TypeError, ValueError):
            _tie_rtol = 1e-4
        _tie_atol = max(0.0, _tie_atol)
        _tie_rtol = max(0.0, _tie_rtol)

        best_lr, probe_batch = _run_mini_train_lr_grid(
            model,
            sample_ds,
            safe_batch,
            lr_candidates,
            device,
            optimizer_name=optimizer_name,
            mini_train_batches=_mini_b,
            peak_rel_slack=_peak_sl,
            quadratic_interp_steps=_qinterp,
            collate_fn=collate_fn,
            max_grad_norm=_max_gn,
            shuffle_train_batches=True,
            dataloader_seed=_sseed,
            mini_train_fn=_mini_train_fn,
            refinement_tie_atol=_tie_atol,
            refinement_tie_rtol=_tie_rtol,
            gradient_accumulation_steps=_gas_lr,
            samples_per_lr=_samples_plr,
            lr_pick_mode=_lr_pick_mode,
            pick_trim_low_lr_frac=_pick_trim,
            pick_explosion_rel_rolling_min=_pick_expl_rel,
            pick_explosion_step_ratio=_pick_expl_step,
            pick_segment_pick_frac=_pick_seg_frac,
        )

        best_lr_raw = float(best_lr)
        best_lr_out = best_lr_raw
        _linear_scale = bool(
            dataset_type_dict.get("lr_finder_linear_scale_lr_to_effective_batch", False)
        )
        _scale_msg = ""
        if _linear_scale:
            try:
                _plan_bs = int(
                    dataset_type_dict.get(
                        "lr_finder_planned_per_device_batch_size", probe_batch
                    )
                )
            except (TypeError, ValueError):
                _plan_bs = int(probe_batch)
            try:
                _plan_gpus = max(
                    1, int(dataset_type_dict.get("lr_finder_planned_gpu_nums", 1))
                )
            except (TypeError, ValueError):
                _plan_gpus = 1
            _plan_bs = max(1, _plan_bs)
            probe_bs = max(1, int(probe_batch))
            num = _plan_bs * _gas_lr * _plan_gpus
            den = probe_bs * _gas_lr * 1
            if den > 0 and math.isfinite(num):
                ratio = num / den
                if math.isfinite(ratio) and ratio > 0:
                    best_lr_out = best_lr_raw * ratio
                    _scale_msg = (
                        f"  linear_batch_scale={ratio:.4g} "
                        f"(eff_batch planned={_plan_bs}×gas={_gas_lr}×{_plan_gpus}gpu "
                        f"vs probe={probe_bs}×gas={_gas_lr}×1gpu; "
                        f"lr_probe={best_lr_raw:.2e} → lr={best_lr_out:.2e})"
                    )

        train_batch = min(b_train, probe_batch)
        if _cap_bs is not None and _cap_bs >= 1:
            train_batch = min(train_batch, _cap_bs)
        _tb_desc = (
            f"min(B_train, probe, run_config batch_size cap)={train_batch}"
            if _cap_bs is not None and _cap_bs >= 1
            else f"min(B_train, probe)={train_batch}"
        )
        print(
            f"[LR Finder] Selected LR: {best_lr_out:.2e}  "
            f"(probe batch_size={probe_batch}; B_train={b_train}; "
            f"training batch_size={_tb_desc})"
            f"{_scale_msg}",
            flush=True,
        )
        return {
            "lr": best_lr_out,
            "lr_mini_train_probe": best_lr_raw,
            "linear_lr_batch_scale_applied": bool(_scale_msg),
            "batch_size": train_batch,
            "b_max_synthetic": b_max,
            "b_train_synthetic": b_train_syn,
            "probe_batch_real_data": probe_batch,
            "lr_probe_points": _points,
            "mini_train_batches": _mini_b,
            "gradient_accumulation_steps": _gas_lr,
            "samples_per_lr": _samples_plr,
            "lr_pick_mode": _lr_pick_mode,
            "pick_segment_pick_frac": _pick_seg_frac,
        }

    except Exception as exc:
        print(f"[LR Finder] Error during LR search: {exc}", flush=True)
        return None

    finally:
        if dpo_ref_ckpt_path:
            try:
                os.remove(dpo_ref_ckpt_path)
            except OSError:
                pass
        torch.cuda.empty_cache()
