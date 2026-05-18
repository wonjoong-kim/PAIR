"""Shared output parsing + tool-cache helpers used by both GTA and ToolBench
environments. Kept in a separate module so neither env depends on the other.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────
# GT tool cache
# ──────────────────────────────────────────────

def canonicalize_args(args: Any) -> str:
    if args is None:
        return ""
    if isinstance(args, (dict, list)):
        return json.dumps(args, ensure_ascii=False, sort_keys=True)
    return str(args).strip()


def build_gt_tool_cache(
    dialogs: List[Dict[str, Any]],
) -> Dict[Tuple[str, str], Tuple[str, Dict[str, Any]]]:
    """(tool_name, canonical_args) → (text, gta_tool_content) cache."""
    cache: Dict[Tuple[str, str], Tuple[str, Dict[str, Any]]] = {}
    pending: Dict[str, Any] = {}
    for turn in dialogs or []:
        role = turn.get("role")
        if role == "assistant":
            pending = {}
            for c in (turn.get("tool_calls") or []):
                fn = c.get("function") or {}
                if fn.get("name"):
                    pending[fn["name"]] = fn.get("arguments")
        elif role == "tool":
            name = turn.get("name")
            content = turn.get("content")
            text: Optional[str]
            if isinstance(content, dict) and content.get("type") == "text":
                text = str(content.get("content", ""))
            elif isinstance(content, str):
                text = content
            else:
                text = str(content) if content is not None else None
            if name and text is not None:
                key = (name, canonicalize_args(pending.get(name)))
                if key not in cache:
                    cache[key] = (text, {"type": "text", "content": text})
    return cache


# ──────────────────────────────────────────────
# Regex parser
# ──────────────────────────────────────────────

FINAL_RE = re.compile(r"final\s*answer\s*:\s*(.*?)\s*$",
                      re.IGNORECASE | re.DOTALL)
THOUGHT_RE = re.compile(r"thought\s*:\s*(.*?)(?=\n\s*action\s*:|$)",
                        re.IGNORECASE | re.DOTALL)
ACTION_RE = re.compile(r"action\s*:\s*([\w\-]+)", re.IGNORECASE)
ACTION_INPUT_RE = re.compile(
    r"action\s*input\s*:\s*(.*?)(?=(?:\n\s*thought\s*:|\n\s*action\s*:|$))",
    re.IGNORECASE | re.DOTALL,
)


def parse_first_step(raw: str) -> Dict[str, Any]:
    """Extract one of:
        {"kind": "final", "thought": str, "final_answer": str}
        {"kind": "step",  "thought": str, "action": str, "action_input": ...}
        {"kind": "error", "reason": str}
    """
    if not raw or not raw.strip():
        return {"kind": "error", "reason": "empty output"}

    mfinal = FINAL_RE.search(raw)
    thought = ""
    mt = THOUGHT_RE.search(raw)
    if mt:
        thought = mt.group(1).strip()

    if mfinal:
        final_ans = mfinal.group(1).strip()
        final_ans = re.split(r"\n\s*(thought|action)\s*:", final_ans, flags=re.IGNORECASE)[0].strip()
        return {"kind": "final", "thought": thought, "final_answer": final_ans}

    ma = ACTION_RE.search(raw)
    if not ma:
        return {"kind": "error", "reason": "no final answer or action"}
    action = ma.group(1).strip()

    mai = ACTION_INPUT_RE.search(raw)
    action_input: Any = None
    if mai:
        raw_ai = mai.group(1).strip().strip("`")
        if raw_ai.startswith("{") or raw_ai.startswith("["):
            try:
                action_input = json.loads(raw_ai)
            except Exception:
                action_input = raw_ai
        else:
            action_input = raw_ai

    return {"kind": "step", "thought": thought, "action": action, "action_input": action_input}


def gpt_extract_first_step(raw: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """Fallback parser via GPT (used only when `GPT_PARSE_*` env vars enabled)."""
    if not os.environ.get("OPENAI_API_KEY"):
        return {"kind": "error", "reason": "OPENAI_API_KEY missing"}
    try:
        import openai
    except Exception:
        return {"kind": "error", "reason": "openai not available"}

    sys_prompt = (
        "You are a strict converter. Given ONLY the assistant raw output text, "
        "convert it into one of the following JSON objects and return ONLY the JSON.\n"
        "- If text includes 'Final Answer:' (case-insensitive), extract that as "
        "final_answer and return {kind:'final'}. Keep only the correct answer "
        "without units.\n"
        "- Otherwise, locate the FIRST trio of 'Thought:', 'Action:', "
        "'Action Input:'. 'Action Input' should be JSON when possible; otherwise "
        "a string.\n"
        "- Output schemas (exactly one):\n"
        "  {\"kind\":\"final\",\"thought\":<string>,\"final_answer\":<string>}\n"
        "  {\"kind\":\"step\",\"thought\":<string>,\"action\":<string>,"
        "\"action_input\":<object or string>}\n"
        "No additional text."
    )
    try:
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": raw},
            ],
            temperature=0.0,
            max_tokens=500,
        )
        txt = (resp.choices[0].message.content or "").strip()
        if txt.startswith("```"):
            txt = re.sub(r"^```[a-zA-Z]*\n?", "", txt)
            txt = re.sub(r"```$", "", txt).strip()
        data = json.loads(txt)
        if isinstance(data, dict) and data.get("kind") in {"final", "step"}:
            return data
        return {"kind": "error", "reason": f"bad kind: {data.get('kind')}"}
    except Exception as e:
        return {"kind": "error", "reason": f"gpt fallback failed: {e}"}


def parse_with_modes(raw: str) -> Dict[str, Any]:
    """Dispatch by `GPT_PARSE_PRIMARY` / `GPT_PARSE_FALLBACK` env vars."""
    if os.environ.get("GPT_PARSE_PRIMARY") == "1":
        parsed = gpt_extract_first_step(raw)
        if parsed.get("kind") == "error":
            parsed = parse_first_step(raw)
        return parsed
    if os.environ.get("GPT_PARSE_FALLBACK") == "1":
        parsed = parse_first_step(raw)
        if parsed.get("kind") != "error":
            return parsed
        return gpt_extract_first_step(raw)
    return parse_first_step(raw)
