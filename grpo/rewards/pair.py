"""PAIR — turn-level reward for GRPO (main method).

Two-stage linear probe trained offline (see `probing/train_pair.py`):

    Stage 1: last_token hidden state → LR → s_bc
    Stage 2: [multi_layer_attn ; s_bc]   → LR → s_final

`s_final` is the per-turn reward signal. Both probes load from
`<PAIR_ROOT>/data/models/methods/PAIR/{model}/{dataset}/pair_{train_mode}.pkl`.

`PAIRMomentumReward` injects a momentum bonus in LOGIT space (paper
Eq. 7) so the final reward stays in (0, 1) and group-relative advantages
don't collapse from probe saturation.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import features as F
from .base import Category, RewardFunction, RewardOutput, Trajectory
from ..paths import PAIR_PROBE_DIR

logger = logging.getLogger("pair_reward")


def _model_key(policy_name: str) -> str:
    p = policy_name.lower()
    if "llama" in p:
        return "llama8b"
    if "qwen" in p:
        return "qwen7b"
    if "mistral" in p:
        return "mistral7b"
    return p


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return float(np.log(p / (1.0 - p)))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _temp_clip(p: float, T: float = 2.0, eps: float = 0.05) -> float:
    """Soften a frozen-probe output to combat reward saturation.

    A frozen LR probe quickly outputs 0.999... when the policy learns to
    push hidden states into its high-confidence region (reward hacking).
    Once every trajectory in a GRPO group hits the ceiling, group_std=0
    and the advantage signal vanishes — policy then drifts only via KL,
    away from the actual task.

    Two compounding mitigations:
      1. Temperature T=2 on the logit halves probe confidence: 0.99 → 0.91,
         0.5 → 0.5. Ordering preserved.
      2. Hard clip to [eps, 1-eps] guarantees a finite logit so there is
         always some advantage signal in the group.
    """
    p_soft = _sigmoid(_logit(p) / T)
    return min(max(p_soft, eps), 1.0 - eps)


class PAIRReward(RewardFunction):
    """Frozen hidden LR (Stage 1) + attention-based LR correction (Stage 2)."""

    name = "pair"
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
        self._probe_correction = None
        self._load()

    def _bundle_path(self) -> Path:
        return PAIR_PROBE_DIR / _model_key(self.policy_name) / self.dataset / f"pair_{self.train_mode}.pkl"

    def _load(self) -> None:
        path = self._bundle_path()
        if not path.exists():
            raise FileNotFoundError(
                f"PAIR probe not found: {path}. Run probing/train_pair.py first."
            )
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        self._probe_base = bundle["probe_base"]
        self._probe_correction = bundle["probe_correction"]
        self._probe_config = bundle.get("config", {})
        logger.info(
            f"Loaded PAIR from {path} "
            f"(hidden={self._probe_config.get('hidden_feature', 'last_token')}, "
            f"C={self._probe_config.get('C', '?')})"
        )

    def reload_probe(self) -> None:
        self._load()

    def compute_rewards(self, trajectory: Trajectory) -> RewardOutput:
        t0 = time.time()
        assistant_turns = [t for t in trajectory.turns if t.role == "assistant"]
        rewards: List[float] = []
        s_bc_values: List[float] = []

        for turn in assistant_turns:
            if turn.hidden_states is None or turn.attentions is None:
                rewards.append(0.0)
                s_bc_values.append(0.0)
                continue
            s_bc = 0.0
            s_final = 0.0
            try:
                h_feat = F.feat_last_token(turn.hidden_states, turn.turn_start, turn.turn_end)
                s_bc = _temp_clip(
                    float(self._probe_base.predict_proba(h_feat.reshape(1, -1))[0, 1])
                )

                a_feat = F.feat_multi_layer_attn(turn.attentions, turn.turn_start, turn.turn_end)
                stage2_input = np.concatenate([a_feat, [s_bc]]).reshape(1, -1)
                s_final = _temp_clip(
                    float(self._probe_correction.predict_proba(stage2_input)[0, 1])
                )
            except Exception as e:
                logger.warning(f"PAIR failed on turn: {e}")
                s_final = 0.0
                s_bc = 0.0
            s_bc_values.append(s_bc)
            rewards.append(s_final)
            # Free large activations once probe inference is done.
            turn.hidden_states = None
            turn.attentions = None

        self.total_calls += 1
        self.total_latency_sec += time.time() - t0
        return RewardOutput(
            turn_rewards=rewards,
            extras={
                "mean_s_bc": float(np.mean(s_bc_values)) if s_bc_values else 0.0,
                "mean_s_final": float(np.mean(rewards)) if rewards else 0.0,
            },
        )


class PAIRMomentumReward(PAIRReward):
    """PAIR + cumulative-momentum bonus, applied in logit space (paper Eq. 7).

        r_t = σ( logit(s̃_final,t) + α · (s̃_final,t − mean(s̃_<t)) )

    α defaults to 5 (paper Table 7). Positive momentum boosts the reward
    relative to the trajectory's running mean, negative is symmetrically
    penalized — restoring the within-group variance GRPO needs for credit
    assignment without ever leaving the (0, 1) range.
    """

    name = "pair_momentum"

    def __init__(self, *args, alpha: float = 5.0, clip_negative: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.clip_negative = clip_negative

    def compute_rewards(self, trajectory: Trajectory) -> RewardOutput:
        base_out = super().compute_rewards(trajectory)
        s = base_out.turn_rewards
        rewards = list(s)
        for i in range(1, len(s)):
            past_mean = float(np.mean(s[:i]))
            delta = s[i] - past_mean
            if self.clip_negative:
                delta = max(0.0, delta)
            bonus_logit = self.alpha * delta
            rewards[i] = _sigmoid(_logit(s[i]) + bonus_logit)
        base_out.turn_rewards = rewards
        base_out.extras["momentum_alpha"] = self.alpha
        base_out.extras["mean_momentum_reward"] = (
            float(np.mean(rewards)) if rewards else 0.0
        )
        return base_out
