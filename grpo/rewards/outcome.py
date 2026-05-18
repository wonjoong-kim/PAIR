"""Outcome scoring (used as a success signal alongside the PAIR reward).

For GTA-style ground truth (dict with whitelist/blacklist), uses the
official word-boundary regex matcher; for list-of-references ground
truth, uses max cosine similarity with `all-mpnet-base-v2`.

Loose substring matching is provided for eval-time scoring that mirrors
the reference implementation; strict word-boundary matching is what the
paper reports.
"""

from __future__ import annotations

import re
import time
from typing import Any, List, Optional

from .base import Category, RewardFunction, RewardOutput, Trajectory


_SIM_MODEL = None


def _get_similarity_model():
    global _SIM_MODEL
    if _SIM_MODEL is None:
        from sentence_transformers import SentenceTransformer
        # Force CPU so we don't steal GPU memory from the training policy.
        _SIM_MODEL = SentenceTransformer("all-mpnet-base-v2", device="cpu")
    return _SIM_MODEL


def is_image_compare_gt(gt_answer: Any) -> bool:
    """True if the GT whitelist contains image file paths (image-compare task).

    These episodes are excluded from aggregate metrics because comparing a
    text final-answer to an image file path is meaningless.
    """
    if not isinstance(gt_answer, dict):
        return False
    wl = gt_answer.get("whitelist")
    if not wl:
        return False
    try:
        for group in wl:
            for item in group:
                if isinstance(item, str) and item.lower().endswith((".jpg", ".jpeg", ".png")):
                    return True
    except Exception:
        pass
    return False


def _iscorrect_whitelist(pred: str, whitelist: List[List[str]],
                         blacklist: Optional[List[List[str]]] = None) -> bool:
    """Strict word-boundary matching. Each whitelist group must have one of
    its aliases appear as a whole word; no blacklist alias may appear."""
    count = 0
    for aliases in whitelist:
        pattern = r"\b(?:" + "|".join(re.escape(a) for a in aliases) + r")\b"
        if re.search(pattern, pred, re.IGNORECASE):
            count += 1
    if count != len(whitelist):
        return False
    if not blacklist:
        return True
    pat = r"\b(?:" + "|".join(re.escape(a) for group in blacklist for a in group) + r")\b"
    return not re.search(pat, pred, re.IGNORECASE)


def _iscorrect_whitelist_loose(pred: str, whitelist: List[List[str]],
                               blacklist: Optional[List[List[str]]] = None) -> bool:
    """Loose substring matching. For each whitelist group ALL items must
    appear as case-insensitive substrings; ANY satisfying group → True."""
    p = re.sub(r"\s+", " ", pred.strip()).lower()
    try:
        for group in whitelist:
            ok = True
            for item in group:
                if re.sub(r"\s+", " ", str(item).strip()).lower() not in p:
                    ok = False
                    break
            if ok:
                if blacklist:
                    for bl_group in blacklist:
                        for bl_item in bl_group:
                            if re.sub(r"\s+", " ", str(bl_item).strip()).lower() in p:
                                return False
                return True
    except Exception:
        return False
    return False


def _similarity_score(pred: str, refs: List[str]) -> float:
    from sentence_transformers import util
    model = _get_similarity_model()
    pred_emb = model.encode(pred, convert_to_tensor=True)
    best = 0.0
    for r in refs:
        r_emb = model.encode(str(r), convert_to_tensor=True)
        s = float(max(0.0, float(util.cos_sim(pred_emb, r_emb).item())))
        if s > best:
            best = s
    return best


def evaluate_answer(gt_answer: Any, pred_answer: str, loose: bool = False) -> float:
    """Score a prediction against GTA-style ground truth. Returns [0, 1].

    loose=False (training-time)  — strict word-boundary regex.
    loose=True (eval-time)       — substring matching.
    """
    if gt_answer is None or pred_answer is None:
        return 0.0
    pred_text = str(pred_answer)
    if isinstance(gt_answer, dict) and "whitelist" in gt_answer:
        matcher = _iscorrect_whitelist_loose if loose else _iscorrect_whitelist
        ok = matcher(pred_text, gt_answer["whitelist"], gt_answer.get("blacklist"))
        return 1.0 if ok else 0.0
    if isinstance(gt_answer, list):
        return _similarity_score(pred_text, [str(x) for x in gt_answer])
    return 0.0


class OutcomeReward(RewardFunction):
    """Sparse outcome reward, assigned to the final assistant turn only."""

    name = "outcome"
    category: Category = "external"

    def compute_rewards(self, trajectory: Trajectory) -> RewardOutput:
        t0 = time.time()
        score = evaluate_answer(trajectory.gt_answer, trajectory.final_answer)
        trajectory.success = score

        assistant_turns = [t for t in trajectory.turns if t.role == "assistant"]
        n = len(assistant_turns)
        if n == 0:
            return RewardOutput(turn_rewards=[])
        rewards = [0.0] * n
        rewards[-1] = float(score)

        self.total_calls += 1
        self.total_latency_sec += time.time() - t0
        return RewardOutput(
            turn_rewards=rewards,
            extras={"outcome_score": float(score)},
        )
