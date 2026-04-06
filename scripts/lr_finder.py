import json
import math
import random
from typing import Optional

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


# --------------------------------------------------------------------------- #
# Auto batch-size finder
# --------------------------------------------------------------------------- #

def _can_run(model, tokenizer, batch_size: int, seq_len: int, device: str) -> bool:
    try:
        ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len)).to(device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
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


def find_max_batch_size(model, tokenizer, seq_len: int = 512, device: str = "cuda") -> int:
    """Exponential + binary search for the largest fitting batch size."""
    batch = 1
    while _can_run(model, tokenizer, batch, seq_len, device):
        batch *= 2

    low, high, best = batch // 2, batch, batch // 2
    while low <= high:
        mid = (low + high) // 2
        if _can_run(model, tokenizer, mid, seq_len, device):
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    # Apply a 0.8 safety factor; DPO needs extra headroom so callers can halve it.
    return max(1, int(best * 0.8))


# --------------------------------------------------------------------------- #
# LR candidate grid
# --------------------------------------------------------------------------- #

def get_lr_candidates(min_lr: float, max_lr: float, points: int) -> list[float]:
    if points <= 1:
        return [min_lr]
    log_min = math.log(min_lr)
    log_max = math.log(max_lr)
    step = (log_max - log_min) / (points - 1)
    return [math.exp(log_min + i * step) for i in range(points)]


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

    for i, batch in enumerate(dataloader):
        if i >= steps:
            break

        optimizer.zero_grad()

        texts = batch[text_key]
        # Truncate to seq_len so memory usage matches the find_max_batch_size probe.
        inputs = tokenizer(
            texts, return_tensors="pt", truncation=True, padding=True,
            max_length=seq_len,
        ).to(device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
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

    best_lr = min(finite_losses, key=finite_losses.get)
    print(
        f"  [LR Finder] Best LR: {best_lr:.2e}  (loss={finite_losses[best_lr]:.4f})",
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
    train_type: str,
    steps: int,
    device: str,
    seq_len: int = 512,
    optimizer_name: Optional[str] = None,
) -> tuple[float, int]:
    # DPO processes two sequences per sample, so start smaller.
    start_batch = max(1, safe_batch // 4 if train_type == "dpo" else safe_batch // 2)
    batch_size = start_batch

    while batch_size >= 1:
        try:
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

def _load_sample_dataset(
    dataset_path: str,
    dataset_type_dict: dict,
    train_type: str,
) -> tuple[Dataset, str]:
    """
    Load the raw JSON dataset, extract the relevant text column, then return
    a shuffled subset (~2 %, min 200 rows) together with the text column name.
    """
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

    # Shuffle and take ~2 % (floor 200 examples) so the probe is not too noisy on
    # small datasets (e.g. 5 k examples → 50 samples was not enough signal).
    data = list(ds)
    random.seed(42)
    random.shuffle(data)
    sample_count = max(200, int(len(data) * 0.02))
    return Dataset.from_list(data[:sample_count]), text_key


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
) -> Optional[dict]:
    use_lora = (
        lora_threshold is not None
        and num_params is not None
        and num_params >= lora_threshold
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _opt_disp = optimizer_name or "adamw_torch"
    print(
        f"[LR Finder] model={model_id}  type={train_type}  "
        f"params={num_params}  lora={use_lora}"
        + (f"  lora_r={lora_r}" if use_lora else "")
        + f"  optim={_opt_disp}  micro_steps={steps}  lr_points={lr_points}",
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
        # Find optimal (safe) batch size for this model
        # ------------------------------------------------------------------ #
        print("[LR Finder] Probing max batch size …", flush=True)
        safe_batch = find_max_batch_size(model, tokenizer, seq_len=seq_len, device=device)
        print(f"[LR Finder] Safe batch size: {safe_batch}", flush=True)

        # ------------------------------------------------------------------ #
        # Load 1 % dataset sample
        # ------------------------------------------------------------------ #
        print(f"[LR Finder] Loading sample subset of {dataset_path} …", flush=True)
        sample_ds, text_key = _load_sample_dataset(dataset_path, dataset_type_dict, train_type)
        print(f"[LR Finder] Sample size: {len(sample_ds)} examples", flush=True)

        # ------------------------------------------------------------------ #
        # Sweep LR candidates
        # ------------------------------------------------------------------ #
        lr_candidates = get_lr_candidates(min_lr, max_lr, lr_points)
        print(
            f"[LR Finder] Sweeping {lr_points} LRs: {min_lr:.2e} → {max_lr:.2e}",
            flush=True,
        )

        best_lr, probe_batch = _run_with_auto_batch(
            model,
            tokenizer,
            sample_ds,
            safe_batch,
            lr_candidates,
            text_key,
            train_type,
            steps,
            device,
            seq_len=seq_len,
            optimizer_name=optimizer_name,
        )

        print(
            f"[LR Finder] Selected LR: {best_lr:.2e}  "
            f"(micro-train batch_size={probe_batch}; synthetic safe_batch={safe_batch})",
            flush=True,
        )
        return {
            "lr": best_lr,
            # Batch size used during the LR sweep — not safe_batch from the synthetic probe,
            # which can be 2–4× larger and was never validated with real tokenizer + data.
            "batch_size": probe_batch,
        }

    except Exception as exc:
        print(f"[LR Finder] Error during LR search: {exc}", flush=True)
        return None

    finally:
        torch.cuda.empty_cache()
