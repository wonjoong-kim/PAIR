"""PAIR-NEW — logit-space residual variant of PAIR.

Difference from `pair.py`:

    PAIR (score-space concat, two sigmoids):
        s_final = σ(w_2^T [a ; s_base] + b_2)

    PAIR-NEW (logit-space residual, single sigmoid):
        z_base      = decision_function(probe_base)(h)
        delta_logit = w_2^T [a ; s_base] + b_2
        s_final     = σ(z_base + delta_logit)

Stage 1 is reused as-is, so only Stage 2 differs.

Loads from
`<PAIR_ROOT>/data/models/methods/PAIR_NEW/{model}/{dataset}/pair_new_{train_mode}.pkl`
(produced by `probing/train_pair_new.py`).
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from . import features as F
from .base import Category, RewardFunction, RewardOutput, Trajectory
from .pair import _model_key, _temp_clip
from ..paths import PAIR_NEW_PROBE_DIR

logger = logging.getLogger("pair_new_reward")


class LogitResidualHead(nn.Module):
    """Stage 2: a single linear layer producing delta_logit."""

    def __init__(self, attn_dim: int):
        super().__init__()
        self.linear = nn.Linear(attn_dim + 1, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, a: torch.Tensor, s_base: torch.Tensor) -> torch.Tensor:
        x = torch.cat([a, s_base], dim=1)
        return self.linear(x).squeeze(-1)


class PAIRNewReward(RewardFunction):
    """PAIR with logit-space residual Stage 2 (see module docstring)."""

    name = "pair_new"
    category: Category = "ours"
    requires_internal = True

    def __init__(
        self,
        policy_name: str,
        dataset: str,
        train_mode: str = "mixed",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)
        self.policy_name = policy_name
        self.dataset = dataset
        self.train_mode = train_mode
        self._probe_base = None
        self._attn_scaler = None
        self._head: Optional[LogitResidualHead] = None
        self._load()

    def _bundle_path(self) -> Path:
        return (
            PAIR_NEW_PROBE_DIR / _model_key(self.policy_name) / self.dataset
            / f"pair_new_{self.train_mode}.pkl"
        )

    def _load(self) -> None:
        path = self._bundle_path()
        if not path.exists():
            raise FileNotFoundError(
                f"PAIR-NEW probe not found: {path}. "
                f"Run probing/train_pair_new.py first."
            )
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        self._probe_base = bundle["probe_base"]
        self._attn_scaler = bundle["attn_scaler"]
        attn_dim = int(bundle["head_arch"]["attn_dim"])
        head = LogitResidualHead(attn_dim)
        head.load_state_dict(bundle["head_state_dict"])
        head.eval()
        self._head = head
        self._probe_config = bundle.get("config", {})
        logger.info(f"Loaded PAIR-NEW from {path}")

    def reload_probe(self) -> None:
        self._load()

    def compute_rewards(self, trajectory: Trajectory) -> RewardOutput:
        t0 = time.time()
        assistant_turns = [t for t in trajectory.turns if t.role == "assistant"]
        rewards: List[float] = []
        s_base_values: List[float] = []

        for turn in assistant_turns:
            if turn.hidden_states is None or turn.attentions is None:
                rewards.append(0.0)
                s_base_values.append(0.0)
                continue
            s_base = 0.0
            s_final = 0.0
            try:
                h_feat = F.feat_last_token(turn.hidden_states, turn.turn_start, turn.turn_end).reshape(1, -1)
                a_feat = F.feat_multi_layer_attn(turn.attentions, turn.turn_start, turn.turn_end).reshape(1, -1)

                z_base = float(self._probe_base.decision_function(h_feat)[0])
                s_base = float(self._probe_base.predict_proba(h_feat)[0, 1])

                a_scaled = self._attn_scaler.transform(a_feat).astype(np.float32)
                with torch.no_grad():
                    delta_logit = float(self._head(
                        torch.from_numpy(a_scaled),
                        torch.tensor([[s_base]], dtype=torch.float32),
                    ).item())
                s_final = _temp_clip(float(1.0 / (1.0 + np.exp(-(z_base + delta_logit)))))
            except Exception as e:
                logger.warning(f"PAIR-NEW failed on turn: {e}")
                s_final = 0.0
                s_base = 0.0
            s_base_values.append(s_base)
            rewards.append(s_final)
            turn.hidden_states = None
            turn.attentions = None

        self.total_calls += 1
        self.total_latency_sec += time.time() - t0
        return RewardOutput(
            turn_rewards=rewards,
            extras={
                "mean_s_base": float(np.mean(s_base_values)) if s_base_values else 0.0,
                "mean_s_final": float(np.mean(rewards)) if rewards else 0.0,
            },
        )
