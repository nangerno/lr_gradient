"""
Task-specific datasets and mini-train losses for ``lr_finder.find_lr``.

Instruct uses causal LM CE on ``train_tokenized_*.json``. DPO uses TRL-aligned
sigmoid DPO loss on ``dpo_train_*.json``. GRPO uses a clipped surrogate on
teacher-forced placeholder completions (same algebraic form as TRL GRPO when
``old_per_token_logps`` matches detach policy, plus optional KL to a PEFT
reference), without rollouts or reward servers.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader

LR_TASK_INSTRUCT = "instruct"
LR_TASK_DPO = "dpo"
LR_TASK_GRPO = "grpo"


def _load_trainable_checkpoint_cpu(path: str) -> dict[str, torch.Tensor]:
    """
    Load a ``{param_name: tensor}`` checkpoint onto CPU.

    Prefer ``mmap=True`` when supported (large full fine-tunes).
    """
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (TypeError, OSError):
        try:
            blob = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            blob = torch.load(path, map_location="cpu")
    if not isinstance(blob, dict):
        raise TypeError("trainable checkpoint must be a dict[str, Tensor]")
    return blob


def normalize_lr_finder_task(raw: Any) -> str:
    s = str(raw or LR_TASK_INSTRUCT).strip().lower()
    if s in (LR_TASK_DPO, "preference", "dp"):
        return LR_TASK_DPO
    if s in (LR_TASK_GRPO, "rl", "grpo"):
        return LR_TASK_GRPO
    return LR_TASK_INSTRUCT


# --------------------------------------------------------------------------- #
# Shared tensor helpers (mirror common TRL patterns)
# --------------------------------------------------------------------------- #


def selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Per-position log p(index) from logits ``[..., vocab]`` and token ``index``."""
    lp = torch.log_softmax(logits.float(), dim=-1)
    out = lp.gather(dim=-1, index=index.unsqueeze(-1).long()).squeeze(-1)
    return out.to(logits.dtype)


def _pad_sequences(
    tensors: list[torch.Tensor],
    *,
    pad_value: int,
    pad_side: str = "right",
) -> torch.Tensor:
    max_len = max(int(t.shape[0]) for t in tensors)
    out = []
    for t in tensors:
        pad_len = max_len - int(t.shape[0])
        if pad_len <= 0:
            out.append(t)
            continue
        pad = torch.full((pad_len,), pad_value, dtype=t.dtype, device=t.device)
        if pad_side == "left":
            out.append(torch.cat([pad, t], dim=0))
        else:
            out.append(torch.cat([t, pad], dim=0))
    return torch.stack(out, dim=0)


@dataclass
class _PreferenceCollator:
    pad_token_id: int
    max_length: int | None
    truncation_mode: str = "keep_start"

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        prompt_chosen = [e["prompt_ids"] + e["chosen_ids"] for e in examples]
        prompt_rejected = [e["prompt_ids"] + e["rejected_ids"] for e in examples]
        chosen_mask = [[0] * len(e["prompt_ids"]) + [1] * len(e["chosen_ids"]) for e in examples]
        rejected_mask = [[0] * len(e["prompt_ids"]) + [1] * len(e["rejected_ids"]) for e in examples]

        if self.max_length is not None:
            m = int(self.max_length)
            if self.truncation_mode == "keep_start":
                sl = slice(None, m)
            elif self.truncation_mode == "keep_end":
                sl = slice(-m, None)
            else:
                raise ValueError(f"bad truncation_mode {self.truncation_mode}")
            prompt_chosen = [ids[sl] for ids in prompt_chosen]
            prompt_rejected = [ids[sl] for ids in prompt_rejected]
            chosen_mask = [mm[sl] for mm in chosen_mask]
            rejected_mask = [mm[sl] for mm in rejected_mask]

        input_ids = prompt_chosen + prompt_rejected
        attention_mask = [[1] * len(ids) for ids in input_ids]
        completion_mask = chosen_mask + rejected_mask

        t_ids = [torch.tensor(x, dtype=torch.long) for x in input_ids]
        t_attn = [torch.tensor(x, dtype=torch.long) for x in attention_mask]
        t_cmpl = [torch.tensor(x, dtype=torch.long) for x in completion_mask]
        return {
            "input_ids": _pad_sequences(t_ids, pad_value=self.pad_token_id),
            "attention_mask": _pad_sequences(t_attn, pad_value=0),
            "completion_mask": _pad_sequences(t_cmpl, pad_value=0),
        }


def _add_eos_if_needed(text: str, eos: str) -> str:
    if eos and not text.endswith(eos):
        return text + eos
    return text


def build_dpo_preference_dataset(
    json_path: str,
    dataset_type: dict,
    tokenizer,
) -> Dataset:
    from tokenize_dpo import get_dataset as get_dpo_dataset

    ds = get_dpo_dataset(json_path, dataset_type)
    eos = tokenizer.eos_token or ""

    def row_tokenize(ex: dict[str, Any]) -> dict[str, Any]:
        prompt = ex["prompt"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]
        if not isinstance(prompt, str):
            prompt = str(prompt)
        if not isinstance(chosen, str):
            chosen = str(chosen)
        if not isinstance(rejected, str):
            rejected = str(rejected)
        chosen = _add_eos_if_needed(chosen, eos)
        rejected = _add_eos_if_needed(rejected, eos)
        p_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        pc_ids = tokenizer(prompt + chosen, add_special_tokens=True)["input_ids"]
        pr_ids = tokenizer(prompt + rejected, add_special_tokens=True)["input_ids"]
        if not (pc_ids[: len(p_ids)] == p_ids and pr_ids[: len(p_ids)] == p_ids):
            # Still split consistently with TRL when tokenizer boundary quirks appear.
            pass
        return {
            "prompt_ids": p_ids,
            "chosen_ids": pc_ids[len(p_ids) :],
            "rejected_ids": pr_ids[len(p_ids) :],
        }

    return ds.map(row_tokenize, remove_columns=ds.column_names)


def dpo_sigmoid_loss(
    model,
    batch: dict[str, torch.Tensor],
    *,
    beta: float,
    ref_trainable_cpu: Optional[Union[dict[str, torch.Tensor], str]] = None,
) -> torch.Tensor:
    """TRL DPO sigmoid loss (reverse KL), PEFT-ref via ``disable_adapter`` when applicable."""
    from peft import PeftModel

    model_kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "use_cache": False,
    }
    input_ids = batch["input_ids"]
    completion_mask = batch["completion_mask"]
    shift_labels = input_ids[..., 1:].contiguous()
    shift_cm = completion_mask[..., 1:].contiguous()

    # π_ref must be computed without mutating trainable tensors *after* the policy forward,
    # otherwise in-place weight replay (copy_) invalidates autograd on lm_head / embeddings.
    # PEFT: adapter forward is separate from disable_adapter ref forward — safe after policy.
    # Full-FT snapshot: do weight-replay ref *first*, then policy forward for gradients.
    ref_logps: torch.Tensor
    if isinstance(model, PeftModel):
        outputs = model(**model_kwargs)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        per_token = selective_log_softmax(shift_logits, shift_labels)
        per_token = per_token * shift_cm
        logps = per_token.sum(dim=-1)
        with torch.no_grad():
            with model.disable_adapter():
                ref_out = model(**model_kwargs)
            ref_shift = ref_out.logits[..., :-1, :].contiguous()
            ref_per = selective_log_softmax(ref_shift, shift_labels)
            ref_per = ref_per * shift_cm
            ref_logps = ref_per.sum(dim=-1)
    elif ref_trainable_cpu is not None:
        ref_blob: Optional[dict[str, torch.Tensor]] = None
        try:
            if isinstance(ref_trainable_cpu, str):
                ref_blob = _load_trainable_checkpoint_cpu(ref_trainable_cpu)
                ref_map: dict[str, torch.Tensor] = ref_blob
            else:
                ref_map = ref_trainable_cpu
            trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            saved: list[tuple[str, torch.Tensor]] = []
            with torch.no_grad():
                prev_training = model.training
                model.eval()
                try:
                    for name, param in trainable:
                        if name in ref_map:
                            saved.append((name, param.detach().clone()))
                            param.copy_(
                                ref_map[name].to(
                                    device=param.device,
                                    dtype=param.dtype,
                                    non_blocking=True,
                                )
                            )
                    ref_out = model(**model_kwargs)
                finally:
                    for name, tensor in saved:
                        for p_name, p in model.named_parameters():
                            if p_name == name:
                                p.copy_(tensor)
                                break
                    model.train(prev_training)
                ref_shift = ref_out.logits[..., :-1, :].contiguous()
                ref_per = selective_log_softmax(ref_shift, shift_labels)
                ref_per = ref_per * shift_cm
                ref_logps = ref_per.sum(dim=-1)
        finally:
            if ref_blob is not None:
                del ref_blob

        outputs = model(**model_kwargs)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        per_token = selective_log_softmax(shift_logits, shift_labels)
        per_token = per_token * shift_cm
        logps = per_token.sum(dim=-1)
    else:
        outputs = model(**model_kwargs)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        per_token = selective_log_softmax(shift_logits, shift_labels)
        per_token = per_token * shift_cm
        logps = per_token.sum(dim=-1)
        with torch.no_grad():
            ref_out = model(**model_kwargs)
            ref_shift = ref_out.logits[..., :-1, :].contiguous()
            ref_per = selective_log_softmax(ref_shift, shift_labels)
            ref_per = ref_per * shift_cm
            ref_logps = ref_per.sum(dim=-1)

    chosen_logps, rejected_logps = logps.chunk(2, dim=0)
    ref_chosen, ref_rejected = ref_logps.chunk(2, dim=0)
    chosen_ratios = chosen_logps - ref_chosen
    rejected_ratios = rejected_logps - ref_rejected
    delta = chosen_ratios - rejected_ratios
    return (-F.logsigmoid(beta * delta)).mean()


def _mini_train_mean_loss_dpo(
    model,
    dataloader: DataLoader,
    lr: float,
    device: str,
    optimizer_name: Optional[str],
    *,
    num_batches: int,
    max_grad_norm: float,
    beta: float,
    _make_optimizer,
    _trainable_params,
    _bf16_autocast,
    gradient_accumulation_steps: int = 1,
    ref_trainable_cpu: Optional[Union[dict[str, torch.Tensor], str]] = None,
) -> float:
    gas = max(1, int(gradient_accumulation_steps))
    optimizer = _make_optimizer(model, lr, optimizer_name)
    try:
        trainable = _trainable_params(model)
        model.train()
        losses: list[float] = []
        it = dataloader.__iter__()
        for _ in range(max(1, int(num_batches))):
            optimizer.zero_grad()
            last_inputs: Optional[dict[str, torch.Tensor]] = None
            for _micro in range(gas):
                batch = next(it)
                inputs = {k: v.to(device) for k, v in batch.items()}
                last_inputs = inputs
                with _bf16_autocast(device):
                    loss = dpo_sigmoid_loss(
                        model, inputs, beta=beta, ref_trainable_cpu=ref_trainable_cpu
                    ) / float(gas)
                if not torch.isfinite(loss):
                    return float("inf")
                loss.backward()
            assert last_inputs is not None
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=max_grad_norm)
            optimizer.step()
            model.eval()
            with torch.no_grad():
                with _bf16_autocast(device):
                    loss_after = dpo_sigmoid_loss(
                        model, last_inputs, beta=beta, ref_trainable_cpu=ref_trainable_cpu
                    )
            model.train()
            if not torch.isfinite(loss_after):
                return float("inf")
            losses.append(float(loss_after.item()))
        return float(np.mean(losses)) if losses else float("inf")
    finally:
        del optimizer
        gc.collect()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_grpo_teacher_forced_dataset(
    json_path: str,
    dataset_type: dict,
    tokenizer,
    *,
    max_prompt_tokens: int,
    completion_len: int,
) -> Dataset:
    from tokenize_grpo import get_dataset as get_grpo_dataset

    ds = get_grpo_dataset(json_path, dataset_type)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.pad_token_id or 0
    comp_len = max(1, int(completion_len))

    def row_build(ex: dict[str, Any]) -> dict[str, Any]:
        prompt = ex["prompt"]
        if not isinstance(prompt, str):
            prompt = str(prompt)
        enc = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=max_prompt_tokens)
        p_ids = enc["input_ids"]
        p_mask = enc.get("attention_mask")
        if not isinstance(p_mask, list) or len(p_mask) != len(p_ids):
            p_mask = [1] * len(p_ids)
        c_ids = [int(eos_id)] * comp_len
        c_mask = [1] * comp_len
        return {
            "prompt_ids": p_ids,
            "prompt_mask": p_mask,
            "completion_ids": c_ids,
            "completion_mask": c_mask,
        }

    cols = ds.column_names
    return ds.map(row_build, remove_columns=cols)


def _pad_batch_dict(rows: list[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    def stack_key(key: str, pad_v: int) -> torch.Tensor:
        ts = [torch.tensor(r[key], dtype=torch.long) for r in rows]
        return _pad_sequences(ts, pad_value=pad_v)

    prompt_ids = stack_key("prompt_ids", pad_id)
    prompt_mask = stack_key("prompt_mask", 0)
    completion_ids = stack_key("completion_ids", pad_id)
    completion_mask = stack_key("completion_mask", 0)
    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
    }


def grpo_style_clipped_loss(
    model,
    batch: dict[str, torch.Tensor],
    *,
    epsilon_low: float,
    epsilon_high: float,
    beta: float,
) -> torch.Tensor:
    """Simplified TRL ``loss_type=grpo`` token objective on teacher-forced completions."""
    from peft import PeftModel

    p_ids, p_m = batch["prompt_ids"], batch["prompt_mask"]
    c_ids, c_m = batch["completion_ids"], batch["completion_mask"]
    input_ids = torch.cat([p_ids, c_ids], dim=1)
    attention_mask = torch.cat([p_m, c_m], dim=1)
    comp_len = int(c_ids.size(1))
    mask = c_m
    prompt_len = int(p_ids.size(1))

    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits
    # Causal LM: logits[..., i, :] predicts input_ids[..., i+1].
    kept_logits = logits[:, prompt_len - 1 : prompt_len - 1 + comp_len, :].contiguous()
    labels = input_ids[:, prompt_len : prompt_len + comp_len]
    per_token_logps = selective_log_softmax(kept_logits, labels)
    old_per_token_logps = per_token_logps.detach()
    log_ratio = per_token_logps - old_per_token_logps
    coef_1 = torch.exp(log_ratio)
    advantages = torch.ones_like(coef_1)
    coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
    per_token_loss1 = coef_1 * advantages
    per_token_loss2 = coef_2 * advantages
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

    # LR-finder proxy for GRPO-without-reference:
    # when beta==0 and old_logps are detached from the same forward pass,
    # the clipped ratio objective collapses to a constant (-1), carrying no LR signal.
    # Use completion-token NLL as a stable proxy so LR search remains informative.
    if beta == 0.0:
        per_token_loss = -per_token_logps

    if beta != 0.0:
        with torch.no_grad():
            if isinstance(model, PeftModel):
                with model.disable_adapter():
                    ref_out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            else:
                ref_out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            ref_logits = ref_out.logits[:, prompt_len - 1 : prompt_len - 1 + comp_len, :].contiguous()
            ref_per = selective_log_softmax(ref_logits, labels)
        per_token_kl = torch.exp(ref_per - per_token_logps) - (ref_per - per_token_logps) - 1
        per_token_loss = per_token_loss + beta * per_token_kl

    loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
    return loss


def _mini_train_mean_loss_grpo(
    model,
    dataloader: DataLoader,
    lr: float,
    device: str,
    optimizer_name: Optional[str],
    *,
    num_batches: int,
    max_grad_norm: float,
    epsilon_low: float,
    epsilon_high: float,
    beta: float,
    _make_optimizer,
    _trainable_params,
    _bf16_autocast,
    gradient_accumulation_steps: int = 1,
) -> float:
    gas = max(1, int(gradient_accumulation_steps))
    optimizer = _make_optimizer(model, lr, optimizer_name)
    try:
        trainable = _trainable_params(model)
        model.train()
        losses: list[float] = []
        it = dataloader.__iter__()
        for _ in range(max(1, int(num_batches))):
            optimizer.zero_grad()
            last_inputs: Optional[dict[str, torch.Tensor]] = None
            for _micro in range(gas):
                batch = next(it)
                inputs = {k: v.to(device) for k, v in batch.items()}
                last_inputs = inputs
                with _bf16_autocast(device):
                    loss = (
                        grpo_style_clipped_loss(
                            model,
                            inputs,
                            epsilon_low=epsilon_low,
                            epsilon_high=epsilon_high,
                            beta=beta,
                        )
                        / float(gas)
                    )
                if not torch.isfinite(loss):
                    return float("inf")
                loss.backward()
            assert last_inputs is not None
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=max_grad_norm)
            optimizer.step()
            model.eval()
            with torch.no_grad():
                with _bf16_autocast(device):
                    loss_after = grpo_style_clipped_loss(
                        model,
                        last_inputs,
                        epsilon_low=epsilon_low,
                        epsilon_high=epsilon_high,
                        beta=beta,
                    )
            model.train()
            if not torch.isfinite(loss_after):
                return float("inf")
            losses.append(float(loss_after.item()))
        return float(np.mean(losses)) if losses else float("inf")
    finally:
        del optimizer
        gc.collect()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def synthetic_dpo_batch(tokenizer, batch_pairs: int, seq_len: int, device: str) -> dict[str, torch.Tensor]:
    """Memory-shaped like DPO collator: ``2 * batch_pairs`` rows, length ``seq_len``."""
    vocab = int(getattr(tokenizer, "vocab_size", 0) or 0) or len(tokenizer)
    b_total = max(1, int(batch_pairs)) * 2
    ids = torch.randint(0, vocab, (b_total, seq_len), device=device)
    attn = torch.ones_like(ids)
    plen = max(1, seq_len // 3)
    cm = torch.zeros_like(ids)
    cm[:, plen:] = 1
    return {"input_ids": ids, "attention_mask": attn, "completion_mask": cm}


def synthetic_grpo_batch(tokenizer, batch_size: int, seq_len: int, device: str) -> dict[str, torch.Tensor]:
    vocab = int(getattr(tokenizer, "vocab_size", 0) or 0) or len(tokenizer)
    half = max(1, seq_len // 2)
    comp = max(1, seq_len - half)
    p_ids = torch.randint(0, vocab, (batch_size, half), device=device)
    p_m = torch.ones_like(p_ids)
    c_ids = torch.randint(0, vocab, (batch_size, comp), device=device)
    c_m = torch.ones_like(c_ids)
    # Pad to seq_len total per row for stable memory (extra pad on right)
    pad_len = seq_len - half - comp
    if pad_len > 0:
        pad_id = int(tokenizer.pad_token_id or 0)
        pad = torch.full((batch_size, pad_len), pad_id, dtype=torch.long, device=device)
        p_ids = torch.cat([p_ids, pad], dim=1)
        p_m = torch.cat([p_m, torch.zeros((batch_size, pad_len), device=device, dtype=torch.long)], dim=1)
    return {"prompt_ids": p_ids, "prompt_mask": p_m, "completion_ids": c_ids, "completion_mask": c_m}


def mini_train_mean_loss_for_task(
    task: str,
    model,
    dataloader: DataLoader,
    lr: float,
    device: str,
    optimizer_name: Optional[str],
    *,
    num_batches: int,
    max_grad_norm: float,
    dpo_beta: float,
    grpo_epsilon_low: float,
    grpo_epsilon_high: float,
    grpo_beta: float,
    _make_optimizer,
    _trainable_params,
    _bf16_autocast,
    gradient_accumulation_steps: int = 1,
    dpo_ref_trainable_cpu: Optional[Union[dict[str, torch.Tensor], str]] = None,
) -> float:
    task_n = normalize_lr_finder_task(task)
    if task_n == LR_TASK_DPO:
        return _mini_train_mean_loss_dpo(
            model,
            dataloader,
            lr,
            device,
            optimizer_name,
            num_batches=num_batches,
            max_grad_norm=max_grad_norm,
            beta=dpo_beta,
            gradient_accumulation_steps=gradient_accumulation_steps,
            _make_optimizer=_make_optimizer,
            _trainable_params=_trainable_params,
            _bf16_autocast=_bf16_autocast,
            ref_trainable_cpu=dpo_ref_trainable_cpu,
        )
    if task_n == LR_TASK_GRPO:
        return _mini_train_mean_loss_grpo(
            model,
            dataloader,
            lr,
            device,
            optimizer_name,
            num_batches=num_batches,
            max_grad_norm=max_grad_norm,
            epsilon_low=grpo_epsilon_low,
            epsilon_high=grpo_epsilon_high,
            beta=grpo_beta,
            gradient_accumulation_steps=gradient_accumulation_steps,
            _make_optimizer=_make_optimizer,
            _trainable_params=_trainable_params,
            _bf16_autocast=_bf16_autocast,
        )
    # instruct: caller uses existing CE path in lr_finder
    raise RuntimeError("mini_train_mean_loss_for_task: instruct uses CE kernel in lr_finder")
