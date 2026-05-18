"""GTA environment with hybrid tool execution.

- Loads GTA dialogs from `<PAIR_ROOT>/data/gta/<split>.json`.
- Builds a per-episode GT cache so repeated (tool, args) calls return the
  recorded response instantly.
- Falls back to `agentlego` for uncached tool calls when available, and to
  GPT-backed replacements for two deprecated tools (`MathOCR`, `GoogleSearch`).

The environment does NOT own the policy — it takes a callable
`(system_prompt, user_prompt, capture_internal=...) -> str | PolicyOutput`.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..paths import DATASET_DIRS
from ..rewards.base import PolicyOutput, Trajectory, Turn
from .base import Environment, Prompt
from ._parsing import build_gt_tool_cache, canonicalize_args, parse_with_modes

logger = logging.getLogger("gta_env")

GTA_DATA = DATASET_DIRS["gta"]

CACHE_BYPASS_TOOLS: set[str] = set()
FAKE_TOOLS = {"FastOCR", "FastCalculator", "ImageDescriptor", "WebSearch"}
GPT_REPLACED_TOOLS = {"MathOCR", "GoogleSearch"}


# ──────────────────────────────────────────────
# GPT replacements for deprecated tools
# ──────────────────────────────────────────────

def _gpt_tool_mathocr(args: Any, files_root: str = "data") -> Tuple[str, Dict[str, Any]]:
    """MathOCR replacement via gpt-4o vision (reads LaTeX from an image)."""
    import base64
    import os as _os

    img = ""
    if isinstance(args, dict):
        img = str(args.get("image", "") or "")
    elif isinstance(args, str):
        img = args
    if not img:
        msg = "MathOCR requires 'image' argument"
        return msg, {"type": "text", "content": msg}
    if not _os.path.isabs(img) and not img.startswith("data/"):
        img = _os.path.join(files_root, img)
    if not _os.path.exists(img):
        msg = f"MathOCR: image not found: {img}"
        return msg, {"type": "text", "content": msg}

    ext = _os.path.splitext(img)[1].lower()
    mime = "image/jpeg" if ext in {".jpg", ".jpeg"} else ("image/png" if ext == ".png" else "image/jpeg")
    with open(img, "rb") as fp:
        b64 = base64.b64encode(fp.read()).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"

    try:
        import openai
        if not _os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        client = openai.OpenAI(api_key=_os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content":
                 "You are MathOCR. Read the math expression(s) from the image and "
                 "return ONLY the LaTeX expression. No dollar signs, no markdown, "
                 "no explanations."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract LaTeX only (no $)."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        msg = f"MathOCR API error: {e}"
        return msg, {"type": "text", "content": msg}
    return text, {"type": "text", "content": text}


def _gpt_tool_google_search(args: Any) -> Tuple[str, Dict[str, Any]]:
    """GoogleSearch replacement via gpt-4o-mini (synthesized results)."""
    import os as _os

    query = ""
    k: Optional[int] = None
    if isinstance(args, dict):
        query = str(args.get("query", "") or "")
        if args.get("k") is not None:
            try:
                k = int(args["k"])
            except Exception:
                k = None
    elif isinstance(args, str):
        query = args
    if not query:
        msg = "GoogleSearch requires 'query'"
        return msg, {"type": "text", "content": msg}

    try:
        import openai
        if not _os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        client = openai.OpenAI(api_key=_os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content":
                 "You are GoogleSearch. Produce plausible top results as a JSON "
                 "array of {title, url, snippet}. No commentary."},
                {"role": "user", "content":
                 json.dumps({"query": query, "k": k}, ensure_ascii=False)},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        msg = f"GoogleSearch API error: {e}"
        return msg, {"type": "text", "content": msg}
    return text, {"type": "text", "content": text}


class ToolRunner:
    """Tool dispatcher: GT cache → fake stub → GPT replacement → agentlego."""

    def __init__(self, action_names: List[str]):
        self.action_names = action_names
        self._tool_instances: Dict[str, Any] = {}

    def run(self, tool_name: str, args: Any,
            cache: Dict[Tuple[str, str], Tuple[str, Dict[str, Any]]]
            ) -> Tuple[str, Dict[str, Any], str]:
        if tool_name in FAKE_TOOLS:
            msg = "This tool is not available now. Consider other tools."
            return msg, {"type": "text", "content": msg}, "fake"

        if tool_name not in CACHE_BYPASS_TOOLS:
            key = (tool_name, canonicalize_args(args))
            if key in cache:
                text, content = cache[key]
                return text, content, "cache"

        if tool_name == "MathOCR":
            text, content = _gpt_tool_mathocr(args)
            return text, content, "gpt_mathocr"
        if tool_name == "GoogleSearch":
            text, content = _gpt_tool_google_search(args)
            return text, content, "gpt_gsearch"

        try:
            from agentlego import load_tool  # type: ignore
        except Exception:
            err = f"agentlego not available; cannot execute {tool_name} (not in GT cache)"
            return err, {"type": "text", "content": err}, "error"

        try:
            tool = self._tool_instances.get(tool_name)
            if tool is None:
                tool = load_tool(tool_name, device="cuda:0")
                self._tool_instances[tool_name] = tool
        except Exception as e:
            err = f"Failed to load tool {tool_name}: {e}"
            return err, {"type": "text", "content": err}, "error"

        try:
            kwargs = args if isinstance(args, dict) else ({"text": args} if isinstance(args, str) else {})
            out = tool(**kwargs) if kwargs else tool()
        except Exception as e:
            err = f"Tool {tool_name} execution error: {e}"
            return err, {"type": "text", "content": err}, "error"

        text = json.dumps(out, ensure_ascii=False) if isinstance(out, (dict, list)) else str(out)
        return text, {"type": "text", "content": text}, "agentlego"


# ──────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────

class GTAEnvironment(Environment):
    name = "gta"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.splits: Dict[str, Dict[str, Any]] = {}
        self.tool_runner: Optional[ToolRunner] = None
        self.action_names: List[str] = []

    def load_dataset(self, split: str) -> Dict[str, Any]:
        if split in self.splits:
            return self.splits[split]
        path = GTA_DATA / f"{split}.json"
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
                dataset="gta",
                query=query,
                gt_answer=item.get("gt_answer"),
                gold_dialogs=item.get("dialogs", []),
                tools=item.get("tools", []),
                files=item.get("files", []),
            ))
        return out

    def rollout(
        self,
        policy_chat_fn: Callable[..., Union[str, PolicyOutput]],
        prompt: Prompt,
        max_steps: int = 10,
        record_internal: bool = False,
        system_prompt: str = "You are an expert who can utilize external tools.",
        instruction_block: Optional[str] = None,
    ) -> Trajectory:
        if self.tool_runner is None:
            self.tool_runner = ToolRunner(self.action_names)
        cache = build_gt_tool_cache(prompt.gold_dialogs)

        traj = Trajectory(
            episode_id=prompt.episode_id,
            dataset="gta",
            task_query=prompt.query,
            gt_answer=prompt.gt_answer,
            gold_dialogs=prompt.gold_dialogs,
        )
        traj.turns.append(Turn(role="user", text=prompt.query))

        history_chunks: List[str] = []
        user_prompt_base = _build_user_prompt(prompt.query, prompt.files)
        instr = instruction_block or _build_instruction_block(prompt.tools)

        for step in range(max_steps):
            user_prompt = user_prompt_base + "\n\n" + instr
            if history_chunks:
                user_prompt += "\n\n" + "\n".join(history_chunks)

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

            def _mk(role: str, text: str, **kw) -> Turn:
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
                traj.turns.append(_mk("assistant",
                                      parsed.get("final_answer", ""),
                                      raw_output=raw))
                traj.final_answer = parsed.get("final_answer", "")
                break

            if parsed["kind"] == "error":
                traj.turns.append(_mk("assistant", raw, raw_output=raw))
                history_chunks.append(f"Thought {step+1}: (parse failed)")
                continue

            thought = parsed.get("thought", "")
            action = parsed.get("action", "")
            action_input = parsed.get("action_input", {})

            traj.turns.append(_mk(
                "assistant",
                f"Thought: {thought}\nAction: {action}\nAction Input: {json.dumps(action_input)}",
                tool_name=action,
                tool_args=action_input if isinstance(action_input, dict) else None,
                raw_output=raw,
            ))

            tool_text, tool_content, _source = self.tool_runner.run(action, action_input, cache)
            traj.turns.append(Turn(role="tool", text=tool_text, tool_name=action))

            history_chunks.append(f"Thought {step+1}: {thought}")
            history_chunks.append(f"Action {step+1}: {action}")
            if isinstance(action_input, (dict, list)):
                history_chunks.append(
                    f"Action Input {step+1}: " + json.dumps(action_input, ensure_ascii=False)
                )
            else:
                history_chunks.append(f"Action Input {step+1}: {action_input}")
            history_chunks.append(f"Response {step+1}: {tool_text[:500]}")

        if traj.final_answer is None:
            traj.final_answer = ""
        return traj


def _build_user_prompt(query: str, files: List[Dict[str, Any]]) -> str:
    lines = []
    for f in files or []:
        path = f.get("path") or f.get("file") or f.get("url")
        ftype = f.get("type", "file")
        if path:
            lines.append(f"- {ftype}: {path}")
    extra = ("\nAvailable files:\n" + "\n".join(lines)) if lines else ""
    return query.strip() + extra


def _build_tool_description(tools: List[Dict[str, Any]]):
    lines, names = [], []
    for t in tools:
        name = t.get("name", "?")
        names.append(name)
        desc = t.get("description", "")
        inputs = t.get("inputs", [])
        in_sig = ", ".join(
            f"{inp.get('name', '?')}: {inp.get('type', 'text')}"
            for inp in inputs if isinstance(inp, dict)
        )
        lines.append(f"- {name}: {desc}")
        if in_sig:
            lines.append(f"  Inputs: {in_sig}")
    return "\n".join(lines), names


def _build_instruction_block(tools: List[Dict[str, Any]]) -> str:
    tool_desc, action_names = _build_tool_description(tools)
    return (
        f"Tool descriptions:\n{tool_desc}\n\n"
        f"To use a tool, follow this exact format: "
        f"Thought: <your reasoning> then "
        f"Action: <the tool name, must be one of [{', '.join(action_names)}]> then "
        f"Action Input: <valid JSON object with keys matching the tool's schema>.\n"
        f"If no tool is needed and you know the answer, respond with: "
        f"Thought: <your reasoning> then Final Answer: <final answer only, no extra explanation>.\n"
        f"You must output exactly one of: a single Thought/Action/Action Input "
        f"triplet or a single Thought/Final Answer pair — never both, never more.\n"
        f"Tools are external and must be used as specified — do not simulate them.\n"
        f"All information contained in images must be extracted via tools.\n"
        f"Use only the tools explicitly provided.\n"
        f"If you are unsure or lack information, do not halt — call a tool to gather more."
    )
