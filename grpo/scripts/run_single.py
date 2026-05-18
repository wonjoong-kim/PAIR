"""Train one (policy, env, PAIR variant) GRPO run.

Usage:
    python -m grpo.scripts.run_single \
        --policy qwen7b --env gta --reward pair \
        --steps 500 --batch_size 8 --group_size 4 \
        --output_dir runs/qwen7b_gta_pair
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from grpo.envs import GTAEnvironment, ToolBenchEnvironment
from grpo.paths import RUNS_DIR
from grpo.rewards import OutcomeReward, PAIRReward
from grpo.rewards.pair import PAIRMomentumReward
from grpo.training import GRPOConfig, GRPOTrainer, LoRAPolicy, PolicyConfig

logger = logging.getLogger("run_single")


REWARD_BUILDERS = {
    "pair":           lambda pol, ds, tm, alpha: PAIRReward(pol, ds, tm),
    "pair_momentum":  lambda pol, ds, tm, alpha: PAIRMomentumReward(pol, ds, tm, alpha=alpha or 5.0),
    "outcome":        lambda *_:                  OutcomeReward(),
}
ALL_REWARD_NAMES = sorted(REWARD_BUILDERS)


def build_env(env_name: str):
    if env_name == "gta":
        return GTAEnvironment()
    if env_name == "toolbench":
        return ToolBenchEnvironment()
    raise ValueError(f"unknown env: {env_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, choices=["llama8b", "qwen7b", "mistral7b"])
    parser.add_argument("--env", required=True, choices=["gta", "toolbench"])
    parser.add_argument("--reward", required=True, choices=ALL_REWARD_NAMES)
    parser.add_argument("--train_mode", default="mixed", choices=["clean_only", "mixed"])
    parser.add_argument("--split", default="clean_train")

    # Defaults below mirror paper Table 7 (Appendix G).
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_steps_per_rollout", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-7)
    parser.add_argument("--kl_beta", type=float, default=0.01)
    parser.add_argument("--lr_schedule", default="constant", choices=["constant", "cosine"])
    parser.add_argument("--lr_warmup_frac", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=None,
                        help="Momentum scaling for pair_momentum (paper Eq. 7). Default 5.0.")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(RUNS_DIR / f"{args.policy}_{args.env}_{args.reward}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(Path(args.output_dir) / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info(f"Run: policy={args.policy} env={args.env} reward={args.reward}")
    logger.info(f"Output: {args.output_dir}")

    pol_cfg = PolicyConfig(name=args.policy, max_new_tokens=args.max_new_tokens)
    policy = LoRAPolicy(pol_cfg)
    env = build_env(args.env)
    reward = REWARD_BUILDERS[args.reward](args.policy, args.env, args.train_mode, args.alpha)

    cfg = GRPOConfig(
        policy_name=args.policy,
        env_name=args.env,
        reward_name=args.reward,
        batch_size=args.batch_size,
        group_size=args.group_size,
        num_training_steps=args.steps,
        max_steps_per_rollout=args.max_steps_per_rollout,
        learning_rate=args.lr,
        kl_beta=args.kl_beta,
        lr_schedule=args.lr_schedule,
        lr_warmup_frac=args.lr_warmup_frac,
        split=args.split,
        train_mode=args.train_mode,
        output_dir=args.output_dir,
        save_every=args.save_every,
        log_every=args.log_every,
    )

    trainer = GRPOTrainer(policy, env, reward, cfg)
    trainer.train()
    logger.info(f"Final reward stats: {reward.stats()}")


if __name__ == "__main__":
    main()
