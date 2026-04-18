from __future__ import annotations

import json
import os
from typing import Any

from transformers import AutoTokenizer

from tokenize_dpo import get_dataset as get_dpo_dataset
from tokenize_grpo import get_dataset as get_grpo_dataset


def _rows_to_json_file(rows: list[dict[str, Any]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"Dumped {len(rows)} samples to {out_path} (LR probe / SFT CE proxy)")


def export_dpo_train_tokenized_for_lr_probe(training_request: dict) -> None:
    """After ``dpo_train_*.json`` exists, write ``train_tokenized_*.json``."""
    tr = training_request["train_request"]
    task_id = tr["task_id"]
    train_path = os.path.join("datasets", f"dpo_train_{task_id}.json")
    out_path = os.path.join("datasets", f"train_tokenized_{task_id}.json")
    if not os.path.isfile(train_path):
        print(f"[LR probe export] Skip: {train_path} not found", flush=True)
        return

    tokenizer = AutoTokenizer.from_pretrained(tr["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = tr.get("max_length", -1)
    try:
        ml = int(max_length)
    except (TypeError, ValueError):
        ml = -1

    ds = get_dpo_dataset(train_path, tr["dataset_type"])
    rows: list[dict[str, Any]] = []
    for i in range(len(ds)):
        ex = ds[i]
        prompt = ex["prompt"]
        chosen = ex["chosen"]
        if not isinstance(prompt, str):
            prompt = str(prompt)
        if not isinstance(chosen, str):
            chosen = str(chosen)
        text = prompt + chosen
        enc = tokenizer(
            text,
            truncation=ml > 0,
            max_length=ml if ml > 0 else None,
            add_special_tokens=True,
        )
        ids = enc["input_ids"]
        attn = enc.get("attention_mask")
        if not isinstance(attn, list) or len(attn) != len(ids):
            attn = [1] * len(ids)
        rows.append(
            {
                "input_ids": list(ids),
                "attention_mask": list(attn),
                "labels": list(ids),
            }
        )

    _rows_to_json_file(rows, out_path)


def export_grpo_train_tokenized_for_lr_probe(training_request: dict) -> None:
    """After ``grpo_train_*.json`` exists, write ``train_tokenized_*.json`` (prompt-only CE)."""
    tr = training_request["train_request"]
    task_id = tr["task_id"]
    train_path = os.path.join("datasets", f"grpo_train_{task_id}.json")
    out_path = os.path.join("datasets", f"train_tokenized_{task_id}.json")
    if not os.path.isfile(train_path):
        print(f"[LR probe export] Skip: {train_path} not found", flush=True)
        return

    tokenizer = AutoTokenizer.from_pretrained(tr["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = tr.get("max_length", -1)
    try:
        ml = int(max_length)
    except (TypeError, ValueError):
        ml = -1

    ds = get_grpo_dataset(train_path, tr["dataset_type"])
    rows: list[dict[str, Any]] = []
    for i in range(len(ds)):
        ex = ds[i]
        prompt = ex["prompt"]
        if not isinstance(prompt, str):
            prompt = str(prompt)
        enc = tokenizer(
            prompt,
            truncation=ml > 0,
            max_length=ml if ml > 0 else None,
            add_special_tokens=True,
        )
        ids = enc["input_ids"]
        attn = enc.get("attention_mask")
        if not isinstance(attn, list) or len(attn) != len(ids):
            attn = [1] * len(ids)
        rows.append(
            {
                "input_ids": list(ids),
                "attention_mask": list(attn),
                "labels": list(ids),
            }
        )

    _rows_to_json_file(rows, out_path)
