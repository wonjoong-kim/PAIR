"""Evaluate a trained PAIR-GRPO policy.

Loads:
    1. Base policy + optional trained LoRA adapter.
    2. Test prompts from the env's `clean_test` split.

Runs deterministic (temperature = 0) rollouts, scores each episode, and
writes a JSON report.

Scoring:
    - GTA: substring matching against the whitelist (mirrors the reference
      evaluator). Image-compare episodes are skipped from aggregates.
    - ToolBench: LLM-as-judge (gpt-4o-mini) is recommended because the
      ground truth is a long-form response and string matching scores ~0%.
      Pass `--llm-judge` to enable; otherwise loose substring matching is
      applied to the raw answer.

Usage:
    python -m grpo.scripts.evaluate --policy qwen7b --env gta \
        --ckpt runs/qwen7b_gta_pair/ckpt_final \
        --output runs/qwen7b_gta_pair/eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from grpo.envs import GTAEnvironment, ToolBenchEnvironment
from grpo.envs.base import Environment
from grpo.paths import RUNS_DIR
from grpo.rewards.outcome import evaluate_answer, is_image_compare_gt
from grpo.training import LoRAPolicy, PolicyConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate")


def build_env(env_name: str) -> Environment:
    if env_name == "gta":
        return GTAEnvironment()
    if env_name == "toolbench":
        return ToolBenchEnvironment()
    raise ValueError(f"unknown env: {env_name}")


def _load_lora_weights(policy: LoRAPolicy, ckpt_dir: str) -> None:
    """Load LoRA adapter weights, handling the `.default` key remap.

    PEFT 0.11 sometimes saves keys as `...lora_A.weight` while the live
    model expects `...lora_A.default.weight` — we patch that up before
    loading.
    """
    import torch
    from safetensors.torch import load_file

    adapter_file = Path(ckpt_dir) / "adapter_model.safetensors"
    bin_file = Path(ckpt_dir) / "adapter_model.bin"
    if adapter_file.exists():
        state = load_file(str(adapter_file))
    elif bin_file.exists():
        state = torch.load(str(bin_file), map_location="cpu")
    else:
        raise FileNotFoundError(f"No adapter weights at {ckpt_dir}")

    model_keys = set(policy.model.state_dict().keys())
    remapped: Dict[str, Any] = {}
    for k, v in state.items():
        if k in model_keys:
            remapped[k] = v
        else:
            new_key = (
                k.replace(".lora_A.weight", ".lora_A.default.weight")
                 .replace(".lora_B.weight", ".lora_B.default.weight")
            )
            remapped[new_key if new_key in model_keys else k] = v

    missing, _ = policy.model.load_state_dict(remapped, strict=False)
    lora_missing = [k for k in missing if "lora" in k]
    if lora_missing:
        logger.warning(f"LoRA keys still missing after remap: {lora_missing[:5]}")


def load_policy(policy_name: str, ckpt: Optional[str],
                max_new_tokens: int, temperature: float) -> LoRAPolicy:
    cfg = PolicyConfig(
        name=policy_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=1.0,
    )
    policy = LoRAPolicy(cfg)
    if ckpt is not None:
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        _load_lora_weights(policy, ckpt)
        logger.info(f"Loaded LoRA adapter from {ckpt}")
    policy.eval_mode()
    return policy


# ──────────────────────────────────────────────
# ToolBench LLM-as-judge (optional)
# ──────────────────────────────────────────────

def _llm_judge_tb(query: str, gt: Any, pred: str, model: str = "gpt-4o-mini") -> float:
    """Binary yes/no judge: 1.0 if `pred` answers `query` consistently with
    `gt`, otherwise 0.0. Falls back to 0.0 on API failure."""
    if not pred or not str(gt).strip():
        return 0.0
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set; LLM judge returning 0.0")
        return 0.0
    try:
        import openai
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content":
                 "You are a strict grader. Given a user query, a reference answer, "
                 "and a model answer, output ONLY 'yes' if the model answer is "
                 "consistent with the reference and resolves the query, otherwise "
                 "'no'. No other text."},
                {"role": "user", "content":
                 f"Query: {query}\nReference: {gt}\nModel answer: {pred}"},
            ],
            temperature=0.0,
            max_tokens=4,
        )
        out = (resp.choices[0].message.content or "").strip().lower()
        return 1.0 if out.startswith("y") else 0.0
    except Exception as e:
        logger.warning(f"LLM judge failed: {e}")
        return 0.0


# ──────────────────────────────────────────────
# Main eval
# ──────────────────────────────────────────────

def evaluate_single(policy: LoRAPolicy, env: Environment, split: str,
                    n_episodes: Optional[int], max_steps: int,
                    seed: int, llm_judge: bool) -> Dict[str, Any]:
    prompts = env.sample_prompts(n=n_episodes or 10**9, split=split, seed=seed)
    if n_episodes is not None:
        prompts = prompts[:n_episodes]
    logger.info(f"Evaluating on {len(prompts)} prompts from {env.name}/{split}")

    results: List[Dict[str, Any]] = []
    t_start = time.time()

    for i, p in enumerate(prompts):
        traj = env.rollout(
            policy_chat_fn=policy.chat,
            prompt=p,
            max_steps=max_steps,
            record_internal=False,
        )
        skipped = is_image_compare_gt(p.gt_answer)
        if skipped:
            score = None
        elif env.name == "toolbench" and llm_judge:
            score = _llm_judge_tb(p.query, p.gt_answer, traj.final_answer or "")
        else:
            score = evaluate_answer(p.gt_answer, traj.final_answer, loose=True)

        n_asst = sum(1 for t in traj.turns if t.role == "assistant")
        results.append({
            "episode_id": p.episode_id,
            "query": p.query[:120],
            "final_answer": (traj.final_answer or "")[:120],
            "score": None if skipped else float(score),
            "skipped_image_compare": skipped,
            "n_assistant_turns": n_asst,
        })
        if (i + 1) % 10 == 0 or (i + 1) == len(prompts):
            scored = [r for r in results if r["score"] is not None]
            mean_so_far = sum(r["score"] for r in scored) / len(scored) if scored else 0.0
            logger.info(
                f"  [{i+1}/{len(prompts)}] mean_score={mean_so_far:.3f} "
                f"(scored={len(scored)}, skipped_ic={len(results)-len(scored)})"
            )

    elapsed = time.time() - t_start
    scored_records = [r for r in results if r["score"] is not None]
    scores = [r["score"] for r in scored_records]
    n_scored = max(1, len(scores))
    return {
        "n_episodes": len(results),
        "n_scored": len(scored_records),
        "n_skipped_image_compare": len(results) - len(scored_records),
        "mean_score": sum(scores) / n_scored,
        "success_rate_at_0.5": sum(1 for s in scores if s >= 0.5) / n_scored,
        "elapsed_seconds": elapsed,
        "per_episode": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, choices=["llama8b", "qwen7b", "mistral7b"])
    parser.add_argument("--env", required=True, choices=["gta", "toolbench"])
    parser.add_argument("--ckpt", default=None, help="LoRA adapter dir (omit → base model).")
    parser.add_argument("--split", default="clean_test")
    parser.add_argument("--n_episodes", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    parser.add_argument("--llm-judge", action="store_true",
                        help="Use gpt-4o-mini as a yes/no judge (recommended for ToolBench).")
    parser.add_argument("--eval_all", action="store_true",
                        help=f"Evaluate every {RUNS_DIR}/<policy>_<env>_*/ckpt_final.")
    args = parser.parse_args()

    env = build_env(args.env)

    if args.eval_all:
        prefix = f"{args.policy}_{args.env}_"
        candidates = sorted(
            d for d in RUNS_DIR.glob(f"{prefix}*")
            if (d / "ckpt_final" / "adapter_config.json").exists()
        )
        logger.info(f"Found {len(candidates)} checkpoints to evaluate")
        policy = load_policy(args.policy, ckpt=None,
                             max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature)
        for c in candidates:
            ckpt_dir = c / "ckpt_final"
            out_path = c / "eval.json"
            if out_path.exists():
                logger.info(f"  SKIP {c.name} (eval.json exists)")
                continue
            logger.info(f"\n=== {c.name} ===")
            _load_lora_weights(policy, str(ckpt_dir))
            report = evaluate_single(
                policy, env, args.split, args.n_episodes,
                args.max_steps, args.seed, args.llm_judge,
            )
            report.update({"ckpt": str(ckpt_dir), "policy": args.policy,
                           "env": args.env, "split": args.split})
            with open(out_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"  → mean={report['mean_score']:.3f}, "
                        f"succ@0.5={report['success_rate_at_0.5']:.3f}, saved {out_path}")
        return

    policy = load_policy(args.policy, args.ckpt,
                         max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature)
    report = evaluate_single(
        policy, env, args.split, args.n_episodes,
        args.max_steps, args.seed, args.llm_judge,
    )
    report.update({"ckpt": args.ckpt, "policy": args.policy,
                   "env": args.env, "split": args.split})

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved {args.output}")

    logger.info("\n=== Final ===")
    logger.info(f"  mean_score:        {report['mean_score']:.4f}")
    logger.info(f"  success @ 0.5:     {report['success_rate_at_0.5']:.4f}")
    logger.info(f"  episodes:          {report['n_episodes']}")
    logger.info(f"  elapsed:           {report['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
