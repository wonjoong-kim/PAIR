"""Feature extraction helpers used by PAIR at GRPO inference time.

These mirror the offline extractors in `probing/extract_features.py` and
`probing/extract_multi_layer_attn.py`, but operate on per-turn captures
returned by the policy during rollout.

Conventions:
  - hidden_states: tuple/list of (seq, d_hidden) numpy arrays, one per
    transformer layer (length = num_layers + 1, HuggingFace style).
  - attentions: tuple/list of (num_heads, seq, seq) arrays, one per layer.
  - turn_start, turn_end: token-index range of the current assistant turn.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np


def _to_np(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def feat_last_token(hidden_states: Any, turn_start: int, turn_end: int) -> np.ndarray:
    """Last layer, last token of the eval turn."""
    last_layer = _to_np(hidden_states[-1])
    idx = max(0, turn_end - 1)
    return last_layer[idx].astype(np.float32)


def feat_mean_pooled(hidden_states: Any, turn_start: int, turn_end: int) -> np.ndarray:
    last_layer = _to_np(hidden_states[-1])
    if turn_start >= turn_end:
        turn_end = turn_start + 1
    return last_layer[turn_start:turn_end].mean(axis=0).astype(np.float32)


def feat_multi_layer(hidden_states: Any, turn_start: int, turn_end: int,
                     num_last_layers: int = 4) -> np.ndarray:
    n_total = len(hidden_states)
    start = max(0, n_total - num_last_layers)
    idx = max(0, turn_end - 1)
    parts = [_to_np(hidden_states[i])[idx] for i in range(start, n_total)]
    return np.concatenate(parts).astype(np.float32)


def feat_raw_attention(attentions: Any, turn_start: int, turn_end: int) -> np.ndarray:
    a = _to_np(attentions[-1])
    num_heads = a.shape[0]
    if turn_start >= turn_end:
        turn_end = turn_start + 1
    turn_a = a[:, turn_start:turn_end, :]

    max_attn = turn_a.max(axis=-1).mean(axis=-1)
    std_attn = turn_a.std(axis=-1).mean(axis=-1)
    total = np.clip(turn_a.sum(axis=-1), 1e-10, None)
    if turn_start > 0:
        prefix_ratio = (turn_a[:, :, :turn_start].sum(axis=-1) / total).mean(axis=-1)
    else:
        prefix_ratio = np.zeros(num_heads, dtype=np.float32)
    self_ratio = (turn_a[:, :, turn_start:turn_end].sum(axis=-1) / total).mean(axis=-1)
    return np.concatenate([max_attn, std_attn, prefix_ratio, self_ratio]).astype(np.float32)


def feat_multi_layer_attn(attentions: Any, turn_start: int, turn_end: int) -> np.ndarray:
    """4 stats per head, for every layer — PAIR Stage-2 feature."""
    if turn_start >= turn_end:
        turn_end = turn_start + 1
    all_layer: List[np.ndarray] = []
    for layer_idx in range(len(attentions)):
        a = _to_np(attentions[layer_idx])
        num_heads = a.shape[0]
        turn_a = a[:, turn_start:turn_end, :]
        max_attn = turn_a.max(axis=-1).mean(axis=-1)
        std_attn = turn_a.std(axis=-1).mean(axis=-1)
        total = np.clip(turn_a.sum(axis=-1), 1e-10, None)
        if turn_start > 0:
            prefix = (turn_a[:, :, :turn_start].sum(axis=-1) / total).mean(axis=-1)
        else:
            prefix = np.zeros(num_heads, dtype=np.float32)
        self_r = (turn_a[:, :, turn_start:turn_end].sum(axis=-1) / total).mean(axis=-1)
        all_layer.append(np.concatenate([max_attn, std_attn, prefix, self_r]))
    return np.concatenate(all_layer).astype(np.float32)
