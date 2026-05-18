"""Abstract environment interface.

Each environment wraps a dataset + tool executor. The GRPO loop calls
`sample_prompts()` to get a batch of tasks, then `rollout()` to execute
a policy against those tasks with tool execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..rewards.base import Trajectory


@dataclass
class Prompt:
    episode_id: str
    dataset: str
    query: str
    gt_answer: Any
    gold_dialogs: List[Dict[str, Any]]
    tools: List[Dict[str, Any]]
    files: List[Dict[str, Any]]


class Environment(ABC):
    name: str = "abstract"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def load_dataset(self, split: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    def sample_prompts(self, n: int, split: str = "train") -> List[Prompt]:
        ...

    @abstractmethod
    def rollout(
        self,
        policy_chat_fn: Callable[[str, str], str],
        prompt: Prompt,
        max_steps: int = 6,
        record_internal: bool = False,
    ) -> Trajectory:
        ...
