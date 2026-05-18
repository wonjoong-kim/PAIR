"""ToolBench environment — offline replay simulator.

ToolBench's RapidAPI tools (16K+) can't be called live during training:
most are deprecated, rate-limited, or paid. Instead we use the dataset's
pre-recorded outputs as a replay cache. Uncached calls return a generic
error string (optionally fuzzy-matched against same-tool entries when
`TB_ARGS_FUZZY=1`).
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Callable, Dict, List, Optional, Union

from ..paths import DATASET_DIRS
from ..rewards.base import PolicyOutput, Trajectory, Turn
from .base import Environment, Prompt
from ._parsing import build_gt_tool_cache, canonicalize_args, parse_with_modes

logger = logging.getLogger("toolbench_env")

TOOLBENCH_DATA = DATASET_DIRS["toolbench"]


class ToolBenchEnvironment(Environment):
    name = "toolbench"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.splits: Dict[str, Dict[str, Any]] = {}

    def load_dataset(self, split: str) -> Dict[str, Any]:
        if split in self.splits:
            return self.splits[split]
        path = TOOLBENCH_DATA / f"{split}.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        with open(path) as f:
            data = json.load(f)
        self.splits[split] = data
        return data

    def sample_prompts(self, n: int, split: str = "clean_train",
                       seed: Optional[int] = None) -> List[Prompt]:
        data = self.load_dataset(split)
        keys = list(data.keys())
        rng = random.Random(seed)
        rng.shuffle(keys)
        out: List[Prompt] = []
        for k in keys[:n]:
            item = data[k]
            query = None
            for d in item.get("dialogs", []):
                if d.get("role") == "user":
                    query = d.get("content", "")
                    break
            if not query:
                continue
            out.append(Prompt(
                episode_id=k,
                dataset="toolbench",
                query=query,
                gt_answer=item.get("gt_answer"),
                gold_dialogs=item.get("dialogs", []),
                tools=item.get("tools", []),
                files=[],
            ))
        return out

    def rollout(
        self,
        policy_chat_fn: Callable[..., Union[str, PolicyOutput]],
        prompt: Prompt,
        max_steps: int = 6,
        record_internal: bool = False,
        system_prompt: str = "You are an expert API agent.",
        instruction_block: Optional[str] = None,
    ) -> Trajectory:
        cache = build_gt_tool_cache(prompt.gold_dialogs)
        tool_schemas_str = _format_tools(prompt.tools)

        traj = Trajectory(
            episode_id=prompt.episode_id,
            dataset="toolbench",
            task_query=prompt.query,
            gt_answer=prompt.gt_answer,
            gold_dialogs=prompt.gold_dialogs,
        )
        traj.turns.append(Turn(role="user", text=prompt.query))

        history: List[str] = []
        instr = instruction_block or _default_instruction(tool_schemas_str)
        user_prompt_base = prompt.query
        # Cap cumulative history to bound seq length under long rollouts.
        MAX_HISTORY_CHARS = 6000

        for step in range(max_steps):
            user_prompt = user_prompt_base + "\n\n" + instr
            if history:
                joined = "\n".join(history)
                if len(joined) > MAX_HISTORY_CHARS:
                    joined = "[...earlier turns truncated...]\n" + joined[-MAX_HISTORY_CHARS:]
                user_prompt += "\n\n" + joined

            try:
                po = policy_chat_fn(system_prompt, user_prompt, capture_internal=record_internal)
            except TypeError:
                po = policy_chat_fn(system_prompt, user_prompt)

            if isinstance(po, PolicyOutput):
                raw = po.text
                internal = {
                    "hidden_states": po.hidden_states,
                    "attentions": po.attentions,
                    "turn_start": po.turn_start,
                    "turn_end": po.turn_end,
                    "token_logprobs": po.token_logprobs,
                    "full_token_ids": po.full_token_ids,
                }
            else:
                raw = po
                internal = None

            parsed = parse_with_modes(raw)

            def _mk(role, text, **kw):
                t = Turn(role=role, text=text, **kw)
                if internal and role == "assistant":
                    t.hidden_states = internal["hidden_states"]
                    t.attentions = internal["attentions"]
                    t.turn_start = internal["turn_start"]
                    t.turn_end = internal["turn_end"]
                    t.token_logprobs = internal["token_logprobs"]
                    t.full_token_ids = internal["full_token_ids"]
                return t

            if parsed["kind"] == "final":
                traj.turns.append(_mk("assistant", parsed.get("final_answer", ""), raw_output=raw))
                traj.final_answer = parsed.get("final_answer", "")
                break

            if parsed["kind"] == "error":
                traj.turns.append(_mk("assistant", raw, raw_output=raw))
                history.append(f"Thought {step+1}: (parse failed)")
                continue

            action = parsed.get("action", "")
            args = parsed.get("action_input", {})
            thought = parsed.get("thought", "")

            traj.turns.append(_mk(
                "assistant",
                f"Thought: {thought}\nAction: {action}\nAction Input: {json.dumps(args)}",
                tool_name=action,
                tool_args=args if isinstance(args, dict) else None,
                raw_output=raw,
            ))

            key = (action, canonicalize_args(args))
            if key in cache:
                tool_text, _ = cache[key]
            elif os.environ.get("TB_ARGS_FUZZY") == "1":
                # Fuzzy fallback: serve any same-tool response, ranked by
                # token-overlap of the args string.
                same_tool = [(k, v) for k, v in cache.items() if k[0] == action]
                if same_tool:
                    target = canonicalize_args(args)
                    target_toks = _arg_tokens(target)
                    def _sim(cand_key: str) -> float:
                        cand_toks = _arg_tokens(cand_key)
                        if not target_toks and not cand_toks:
                            return 1.0
                        if not target_toks or not cand_toks:
                            return 0.0
                        return len(target_toks & cand_toks) / len(target_toks | cand_toks)
                    same_tool.sort(key=lambda kv: -_sim(kv[0][1]))
                    tool_text = same_tool[0][1][0]
                else:
                    tool_text = f"[Simulator] Tool '{action}' with args {args} has no cached response."
            else:
                tool_text = f"[Simulator] Tool '{action}' with args {args} has no cached response."

            traj.turns.append(Turn(role="tool", text=tool_text, tool_name=action))

            history.append(f"Thought {step+1}: {thought}")
            history.append(f"Action {step+1}: {action}")
            history.append(f"Action Input {step+1}: " +
                           (json.dumps(args, ensure_ascii=False)
                            if isinstance(args, (dict, list)) else str(args)))
            history.append(f"Response {step+1}: {tool_text[:500]}")

        if traj.final_answer is None:
            traj.final_answer = ""
        return traj


def _arg_tokens(s: str) -> set:
    return set(
        s.replace('"', ' ').replace(':', ' ').replace(',', ' ')
         .replace('{', ' ').replace('}', ' ').split()
    )


def _format_tools(tools: List[Dict[str, Any]]) -> str:
    lines = []
    for t in tools or []:
        lines.append(f"- {t.get('name', '?')}: {t.get('description', '')}")
    return "\n".join(lines)


def _default_instruction(tool_schemas_str: str) -> str:
    return (
        "Tools available:\n"
        + tool_schemas_str + "\n\n"
        + "Use this format: Thought: <reasoning> Action: <tool> Action Input: <JSON>. "
        + "For final answer: Thought: <reasoning> Final Answer: <answer>. "
        + "Output one Thought/Action/Action Input triplet or Thought/Final Answer pair per step."
    )
