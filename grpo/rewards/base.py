"""Base dataclasses and interfaces for PAIR reward functions.

A reward function takes a `Trajectory` and returns per-assistant-turn rewards.
The GRPO loop injects these as token-level rewards on the last token of each
turn and computes group-relative advantages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


Category = Literal["external", "ours"]


@dataclass
class PolicyOutput:
    """Return type of the policy's chat function.

    `text` is always populated. Hidden states, attentions, and token info are
    only populated when `capture_internal=True` is requested (which PAIR
    requires).
    """
    text: str
    # Per-layer hidden states. tuple(num_layers+1,) of (seq, d_hidden) arrays.
    hidden_states: Any = None
    # Per-layer attentions. tuple(num_layers,) of (num_heads, seq, seq) arrays.
    attentions: Any = None
    # Token index range corresponding to the generated assistant turn.
    turn_start: int = 0
    turn_end: int = 0
    # Log-probs of generated tokens (unused by PAIR but kept for parity).
    token_logprobs: Any = None
    num_generated_tokens: int = 0
    # Full prompt+generation token ids — needed for the gradient-carrying
    # re-forward during the policy update.
    full_token_ids: Any = None


@dataclass
class Turn:
    """One assistant / tool / user turn within a trajectory."""
    role: Literal["assistant", "tool", "user"]
    text: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    raw_output: Optional[str] = None
    turn_start: int = 0
    turn_end: int = 0
    hidden_states: Any = None
    attentions: Any = None
    token_logprobs: Any = None
    full_token_ids: Any = None


@dataclass
class Trajectory:
    """Full rollout of one episode."""
    episode_id: str
    dataset: str
    task_query: str
    gt_answer: Any
    turns: List[Turn] = field(default_factory=list)
    final_answer: Optional[str] = None
    success: Optional[float] = None
    token_ids: Optional[List[int]] = None
    policy_name: Optional[str] = None
    gold_dialogs: Optional[List[Dict[str, Any]]] = None


@dataclass
class RewardOutput:
    turn_rewards: List[float]
    extras: Dict[str, Any] = field(default_factory=dict)


class RewardFunction(ABC):
    """Abstract base class for all reward functions."""

    name: str = "abstract"
    category: Category = "external"

    requires_internal: bool = False  # needs hidden_states / attentions

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.total_calls: int = 0
        self.total_latency_sec: float = 0.0

    @abstractmethod
    def compute_rewards(self, trajectory: Trajectory) -> RewardOutput:
        """Return one reward per assistant turn in `trajectory.turns`."""

    def stats(self) -> Dict[str, Any]:
        return {
            "reward_name": self.name,
            "total_calls": self.total_calls,
            "total_latency_sec": self.total_latency_sec,
        }
