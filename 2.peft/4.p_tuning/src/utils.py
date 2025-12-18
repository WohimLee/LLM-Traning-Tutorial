
# utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Reproducibility
# =========================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# JSONL loading (messages)
# =========================
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error at line {i+1} in {path}: {e}")
            rows.append(obj)
    return rows


def validate_messages(sample: Dict[str, Any]) -> None:
    if "messages" not in sample or not isinstance(sample["messages"], list):
        raise ValueError("Each sample must have a list field: 'messages'")

    msgs = sample["messages"]
    if len(msgs) < 2:
        raise ValueError("messages length must be >= 2")

    for m in msgs:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise ValueError("Each message must be a dict with keys: role, content")

    # 训练时一般用最后一轮 assistant 作为监督目标
    if msgs[-1]["role"] != "assistant":
        raise ValueError("The last message must be role='assistant' for SFT supervision.")


# =========================
# Chat template + label masking
# Only compute loss on last assistant message
# =========================
def _find_sublist(haystack: List[int], needle: List[int]) -> Optional[int]:
    """Return start index of needle in haystack, or None."""
    if len(needle) == 0:
        return 0
    if len(needle) > len(haystack):
        return None
    # naive search is fine for typical lengths
    for i in range(0, len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return None


def build_sft_example(
    tokenizer,
    messages: List[Dict[str, str]],
    max_length: int = 2048,
) -> Dict[str, Any]:
    """
    Build one training example:
    - input_ids: tokenized full conversation
    - labels: -100 for prompt part, real token ids for assistant completion part
    """
    # Full text includes assistant content
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Prompt text ends right before assistant content, but includes the assistant "start" marker.
    prompt_messages = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
        add_special_tokens=False,
    )
    prompt = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
        add_special_tokens=False,
    )

    input_ids: List[int] = full["input_ids"]
    prompt_ids: List[int] = prompt["input_ids"]

    # Align: full should contain prompt as a prefix (or near-prefix if truncation)
    start = _find_sublist(input_ids, prompt_ids)
    if start is None:
        # fallback: assume prefix
        start = 0
        # best-effort: find longest common prefix
        lcp = 0
        for a, b in zip(input_ids, prompt_ids):
            if a != b:
                break
            lcp += 1
        prompt_len = lcp
    else:
        prompt_len = start + len(prompt_ids)

    labels = [-100] * len(input_ids)
    for i in range(prompt_len, len(input_ids)):
        labels[i] = input_ids[i]

    attention_mask = full.get("attention_mask", [1] * len(input_ids))

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "full_text": full_text,      # optional debug / qualitative eval
        "prompt_text": prompt_text,  # optional debug / qualitative eval
    }


class SFTMessagesDataset(Dataset):
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data

        # basic validation upfront
        for s in self.data:
            validate_messages(s)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        ex = build_sft_example(
            tokenizer=self.tokenizer,
            messages=sample["messages"],
            max_length=self.max_length,
        )
        # keep only tensors needed by trainer/collator
        return {
            "input_ids": ex["input_ids"],
            "labels": ex["labels"],
            "attention_mask": ex["attention_mask"],
        }


# =========================
# Data collator (pad inputs + pad labels with -100)
# =========================
@dataclass
class DataCollatorForSFT:
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Pad input_ids & attention_mask via tokenizer
        batch = self.tokenizer.pad(
            {
                "input_ids": [f["input_ids"] for f in features],
                "attention_mask": [f["attention_mask"] for f in features],
            },
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Pad labels manually with -100
        max_len = batch["input_ids"].shape[1]
        labels = []
        for f in features:
            l = f["labels"]
            if len(l) < max_len:
                l = l + [-100] * (max_len - len(l))
            else:
                l = l[:max_len]
            labels.append(l)
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


# =========================
# Metrics helpers
# =========================
def ppl_from_eval_loss(eval_loss: float) -> float:
    if eval_loss is None or math.isnan(eval_loss):
        return float("nan")
    try:
        return float(math.exp(eval_loss))
    except OverflowError:
        return float("inf")


# =========================
# Qualitative generation (quick sanity check)
# =========================
@torch.no_grad()
def generate_samples(
    model,
    tokenizer,
    eval_rows: List[Dict[str, Any]],
    n: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    seed: int = 1234,
    device: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    For each row: use messages[:-1] as prompt, generate assistant answer.
    Returns list of dicts: {prompt, pred, ref}
    """
    set_seed(seed)
    model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    picks = eval_rows[:]
    random.shuffle(picks)
    picks = picks[: min(n, len(picks))]

    outputs: List[Dict[str, str]] = []
    for row in picks:
        validate_messages(row)
        msgs = row["messages"]
        ref = msgs[-1]["content"]
        prompt_msgs = msgs[:-1]

        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )

        enc = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        gen_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Only decode newly generated part
        prompt_len = enc["input_ids"].shape[1]
        pred = tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=True).strip()

        outputs.append(
            {
                "prompt": prompt_text,
                "pred": pred,
                "ref": ref,
            }
        )
    return outputs


# =========================
# (Optional) Simple ROUGE-L (character-level, works okay for Chinese quick compare)
# =========================
def _lcs_len(a: str, b: str) -> int:
    # DP LCS length; O(n*m) fine for short answers
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]


def rouge_l_f1(pred: str, ref: str) -> float:
    if not pred or not ref:
        return 0.0
    lcs = _lcs_len(pred, ref)
    prec = lcs / max(len(pred), 1)
    rec = lcs / max(len(ref), 1)
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)


def average_rouge_l(outputs: List[Dict[str, str]]) -> float:
    if not outputs:
        return 0.0
    scores = [rouge_l_f1(o["pred"], o["ref"]) for o in outputs]
    return float(sum(scores) / len(scores))
