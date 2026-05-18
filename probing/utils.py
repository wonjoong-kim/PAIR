"""Shared utilities: logging, seeding, JSON IO, chat-template building, metrics."""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(level)
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        log.addHandler(handler)
    return log


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_assistant_turn_indices(dialogs: List[Dict]) -> List[int]:
    return [i for i, d in enumerate(dialogs) if d["role"] == "assistant"]


def build_chat_messages(dialogs: List[Dict], up_to_idx: Optional[int] = None) -> List[Dict]:
    """Convert dataset dialog format to a list of {role, content} messages
    compatible with `tokenizer.apply_chat_template`. Tool turns are
    flattened into a user-role message so the chat template alternates
    cleanly even for tokenizers that disallow a `tool` role."""
    if up_to_idx is not None:
        dialogs = dialogs[: up_to_idx + 1]

    messages: List[Dict] = []
    for d in dialogs:
        role = d["role"]
        if role == "user":
            messages.append({"role": "user", "content": d["content"]})
        elif role == "assistant":
            parts: List[str] = []
            if "thought" in d:
                parts.append(f"Thought: {d['thought']}")
            if "tool_calls" in d:
                for tc in d["tool_calls"]:
                    fn = tc["function"]
                    parts.append(
                        f"Tool call: {fn['name']}({json.dumps(fn['arguments'])})"
                    )
            if d.get("content"):
                parts.append(d["content"])
            messages.append({"role": "assistant", "content": "\n".join(parts)})
        elif role == "tool":
            content = d["content"]
            if isinstance(content, dict):
                text = content.get("content", json.dumps(content))
            else:
                text = str(content)
            if len(text) > 1000:
                text = text[:1000] + "..."
            messages.append({"role": "user",
                             "content": f"[Tool Result ({d.get('name', '')})]: {text}"})
    return messages


def merge_consecutive_roles(messages: List[Dict]) -> List[Dict]:
    """Some chat templates (Mistral) require strict alternation. Tool turns
    flattened into 'user' can produce consecutive user messages — merge them."""
    if not messages:
        return messages
    merged = [messages[0].copy()]
    for msg in messages[1:]:
        if msg["role"] == merged[-1]["role"]:
            merged[-1]["content"] += "\n\n" + msg["content"]
        else:
            merged.append(msg.copy())
    return merged


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob > bins[i]) & (y_prob <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece / len(y_true))


def pairwise_ranking_accuracy(y_true: np.ndarray, y_prob: np.ndarray,
                              episode_ids: np.ndarray) -> float:
    """Fraction of within-episode (correct, contaminated) pairs whose probe
    score ordering matches the label ordering."""
    correct, total = 0, 0
    for eid in np.unique(episode_ids):
        mask = episode_ids == eid
        yt = y_true[mask]
        yp = y_prob[mask]
        for i in range(len(yt)):
            for j in range(i + 1, len(yt)):
                if yt[i] != yt[j]:
                    total += 1
                    if (yt[i] > yt[j]) == (yp[i] > yp[j]):
                        correct += 1
    return correct / total if total > 0 else 0.0
