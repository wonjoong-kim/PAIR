"""Minimal custom GRPO training loop for PAIR.

Pipeline per step:
    1. Sample `batch_size` prompts from the env.
    2. For each prompt, roll out `group_size` trajectories with the policy,
       capturing internal state (PAIR needs hidden states / attentions).
    3. Compute per-turn rewards with PAIR (`compute_rewards`).
    4. Assemble token-level reward vectors and compute group-relative
       advantages (mean = 0 within each group).
    5. Run a REINFORCE-style update on LoRA parameters with optional
       reference-KL penalty.

We avoid VeRL because our rewards need hidden states / attentions from each
rollout — integration with VeRL's worker/actor split is involved. With
LoRA and small batch sizes, a simple in-process loop is sufficient.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as Fnn
from torch.optim import AdamW

from ..envs.base import Environment
from ..rewards.base import RewardFunction, RewardOutput, Trajectory
from ..rewards.outcome import evaluate_answer
from .policy import LoRAPolicy

logger = logging.getLogger("grpo_loop")


@dataclass
class GRPOConfig:
    policy_name: str
    env_name: str                  # "gta" or "toolbench"
    reward_name: str               # for logging
    batch_size: int = 8
    group_size: int = 4
    num_training_steps: int = 500
    max_steps_per_rollout: int = 6
    learning_rate: float = 1e-6
    clip_epsilon: float = 0.2
    kl_beta: float = 0.0                # reference-KL penalty (0 = off)
    lr_schedule: str = "constant"       # "constant" | "cosine"
    lr_warmup_frac: float = 0.05
    split: str = "clean_train"
    train_mode: str = "mixed"           # "clean_only" | "mixed"
    output_dir: str = "runs/default"
    save_every: int = 100
    log_every: int = 10


@dataclass
class StepMetrics:
    step: int
    mean_reward: float            # raw, pre-normalization
    mean_advantage: float         # post group-normalization (≈ 0)
    mean_success: float
    rollout_seconds: float
    reward_seconds: float
    update_seconds: float
    total_tokens: int
    policy_loss: float
    kl_from_init: float = 0.0
    extras: Dict[str, Any] = field(default_factory=dict)


def _group_normalize(rewards: List[float]) -> List[float]:
    arr = np.array(rewards, dtype=np.float32)
    if arr.std() > 1e-8:
        return ((arr - arr.mean()) / arr.std()).tolist()
    return (arr - arr.mean()).tolist()


class GRPOTrainer:
    """Drives rollout → reward → LoRA update."""

    def __init__(
        self,
        policy: LoRAPolicy,
        env: Environment,
        reward_fn: RewardFunction,
        cfg: GRPOConfig,
    ):
        self.policy = policy
        self.env = env
        self.reward_fn = reward_fn
        self.cfg = cfg

        self.optimizer = AdamW(
            [p for p in self.policy.model.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
        )
        self.scheduler = None
        if cfg.lr_schedule == "cosine":
            from transformers import get_cosine_schedule_with_warmup
            n_warm = max(1, int(cfg.lr_warmup_frac * cfg.num_training_steps))
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=n_warm,
                num_training_steps=cfg.num_training_steps,
            )
            logger.info(f"LR schedule: cosine, warmup={n_warm}/{cfg.num_training_steps}")

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        self._metrics_file = Path(cfg.output_dir) / "metrics.jsonl"
        # Don't append to a prior run's log.
        try:
            if self._metrics_file.exists():
                self._metrics_file.unlink()
        except OSError:
            pass

    # ─── Rollout ───────────────────────────────

    def rollout_batch(self, step: int) -> List[Trajectory]:
        prompts = self.env.sample_prompts(
            n=self.cfg.batch_size,
            split=self.cfg.split,
            seed=step,
        )
        trajectories: List[Trajectory] = []
        needs_internal = self.reward_fn.requires_internal

        for p in prompts:
            for _ in range(self.cfg.group_size):
                traj = self.env.rollout(
                    policy_chat_fn=self.policy.chat,
                    prompt=p,
                    max_steps=self.cfg.max_steps_per_rollout,
                    record_internal=needs_internal,
                )
                traj.policy_name = self.cfg.policy_name

                # Streaming reward compute: free hidden states immediately
                # after we've extracted the score, keeping peak memory low.
                if needs_internal:
                    try:
                        traj.success = evaluate_answer(traj.gt_answer, traj.final_answer)
                        o = self.reward_fn.compute_rewards(traj)
                        if o.turn_rewards:
                            o.extras["raw_mean"] = float(np.mean(o.turn_rewards))
                            o.extras["raw_sum"] = float(sum(o.turn_rewards))
                        else:
                            o.extras["raw_mean"] = 0.0
                            o.extras["raw_sum"] = 0.0
                        traj._cached_reward = o
                    except Exception as e:
                        logger.warning(f"Inline reward failed: {e}")
                        traj._cached_reward = None
                trajectories.append(traj)
        return trajectories

    # ─── Reward + advantage ────────────────────

    def compute_rewards_and_advantages(
        self, trajectories: List[Trajectory]
    ) -> List[RewardOutput]:
        for t in trajectories:
            t.success = evaluate_answer(t.gt_answer, t.final_answer)

        outputs: List[RewardOutput] = []
        for t in trajectories:
            cached = getattr(t, "_cached_reward", None)
            if cached is not None:
                outputs.append(cached)
                continue
            try:
                o = self.reward_fn.compute_rewards(t)
            except Exception as e:
                logger.warning(f"Reward fn {self.reward_fn.name} failed: {e}")
                num = sum(1 for x in t.turns if x.role == "assistant")
                o = RewardOutput(turn_rewards=[0.0] * num, extras={"error": str(e)})
            if o.turn_rewards:
                o.extras["raw_mean"] = float(np.mean(o.turn_rewards))
                o.extras["raw_sum"] = float(sum(o.turn_rewards))
            else:
                o.extras["raw_mean"] = 0.0
                o.extras["raw_sum"] = 0.0
            outputs.append(o)

        # Group-relative normalization (groups = rollouts per prompt).
        if self.cfg.group_size > 1:
            G = self.cfg.group_size
            for i in range(0, len(outputs), G):
                group = outputs[i:i + G]
                if len(group) <= 1:
                    continue
                totals = [sum(o.turn_rewards) for o in group]
                normed = _group_normalize(totals)
                for o, adv in zip(group, normed):
                    if o.turn_rewards:
                        raw = np.array(o.turn_rewards, dtype=np.float32)
                        o.turn_rewards = (raw - raw.mean() + adv).tolist()
        return outputs

    # ─── LoRA update (REINFORCE) ───────────────

    def update_policy(
        self,
        trajectories: List[Trajectory],
        reward_outputs: List[RewardOutput],
    ) -> float:
        """REINFORCE update: -advantage · log p(generated | prompt).

        Backward is called per turn so the autograd graph is released
        immediately — keeps peak activation memory ~constant regardless
        of how many trajectories we have.
        """
        tasks = []
        for traj, out in zip(trajectories, reward_outputs):
            assistant_turns = [t for t in traj.turns if t.role == "assistant"]
            for turn, advantage in zip(assistant_turns, out.turn_rewards):
                if turn.full_token_ids is None:
                    continue
                tok_ids = np.asarray(turn.full_token_ids, dtype=np.int64)
                if tok_ids.size == 0:
                    continue
                gen_len = int(turn.turn_end - turn.turn_start)
                if gen_len <= 0:
                    continue
                tasks.append((tok_ids, int(turn.turn_start), gen_len, float(advantage)))

        n_turns = len(tasks)
        if n_turns == 0:
            return 0.0

        self.policy.train_mode()
        self.optimizer.zero_grad()

        loss_total = 0.0
        kl_total = 0.0
        grad_max_len = min(self.policy.cfg.max_context_len, 2560)
        skipped_oom = 0
        kl_beta = float(self.cfg.kl_beta)
        use_kl = kl_beta > 0.0

        for tok_ids, prompt_len, gen_len, advantage in tasks:
            if tok_ids.shape[0] > grad_max_len:
                trim = tok_ids.shape[0] - grad_max_len
                tok_ids = tok_ids[trim:]
                prompt_len = max(0, prompt_len - trim)
            if gen_len > grad_max_len - 1:
                gen_len = grad_max_len - 1 - prompt_len
                if gen_len <= 0:
                    continue

            ids = torch.from_numpy(tok_ids).unsqueeze(0).to(self.policy.device)
            try:
                outputs = self.policy.model(ids)
                logits = outputs.logits[0].float()
                start = max(1, prompt_len)
                end = min(start + gen_len, logits.shape[0])
                if end <= start:
                    del outputs, logits, ids
                    continue

                target_ids = ids[0, start:end]
                pred_logits = logits[start - 1: end - 1]
                log_probs = Fnn.log_softmax(pred_logits, dim=-1)
                gathered = log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
                mean_lp = gathered.mean()

                kl_term = None
                if use_kl:
                    with torch.no_grad():
                        with self.policy.model.disable_adapter():
                            ref_outputs = self.policy.model(ids)
                            ref_pred = ref_outputs.logits[0].float()[start - 1: end - 1]
                            ref_lp = Fnn.log_softmax(ref_pred, dim=-1).gather(
                                1, target_ids.unsqueeze(-1)
                            ).squeeze(-1)
                            del ref_outputs, ref_pred
                    kl_term = (gathered - ref_lp).mean()
                    kl_total += float(kl_term.detach().cpu().item())

                obj = advantage * mean_lp
                if kl_term is not None:
                    obj = obj - kl_beta * kl_term
                per_turn_loss = -obj / n_turns
                per_turn_loss.backward()
                loss_total += float(per_turn_loss.detach().cpu().item())

                del outputs, logits, pred_logits, log_probs, gathered, mean_lp
                del per_turn_loss, ids, target_ids
                if kl_term is not None:
                    del kl_term
            except torch.cuda.OutOfMemoryError:
                skipped_oom += 1
                torch.cuda.empty_cache()
                continue

            if self.policy.device == "cuda":
                torch.cuda.empty_cache()

        if skipped_oom > 0:
            logger.warning(f"update_policy: skipped {skipped_oom}/{n_turns} turns due to CUDA OOM")

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.policy.model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.policy.eval_mode()
        self._last_kl = kl_total / max(1, n_turns) if use_kl else 0.0
        return loss_total

    # ─── Main loop ─────────────────────────────

    def train(self) -> None:
        logger.info(f"Starting GRPO training: {self.cfg}")
        self._last_step = -1
        try:
            self._train_loop()
        except BaseException as e:
            logger.error(f"Training aborted at step {self._last_step+1}: {type(e).__name__}: {e}")
            try:
                self.policy.save_lora(str(Path(self.cfg.output_dir) / "ckpt_crash"))
                logger.info("Saved ckpt_crash before exit.")
            except Exception as save_err:
                logger.warning(f"ckpt_crash save failed: {save_err}")
            raise

    def _train_loop(self) -> None:
        for step in range(self.cfg.num_training_steps):
            self._last_step = step
            try:
                self._train_step(step)
            except torch.cuda.OutOfMemoryError as e:
                logger.warning(f"step={step} CUDA OOM at top level; skipping. {e}")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                logger.warning(f"step={step} failed ({type(e).__name__}: {e}); skipping")
                torch.cuda.empty_cache()
                continue
        self.policy.save_lora(str(Path(self.cfg.output_dir) / "ckpt_final"))
        logger.info("Training complete.")

    def _train_step(self, step: int) -> None:
        t0 = time.time()
        trajectories = self.rollout_batch(step)
        rollout_sec = time.time() - t0

        t0 = time.time()
        reward_outputs = self.compute_rewards_and_advantages(trajectories)
        reward_sec = time.time() - t0

        t0 = time.time()
        loss = self.update_policy(trajectories, reward_outputs)
        update_sec = time.time() - t0

        raw_means = [o.extras.get("raw_mean", 0.0) for o in reward_outputs]
        mean_reward = float(np.mean(raw_means)) if raw_means else 0.0
        adv_means = [
            float(np.mean(o.turn_rewards)) if o.turn_rewards else 0.0
            for o in reward_outputs
        ]
        mean_advantage = float(np.mean(adv_means)) if adv_means else 0.0
        mean_success = float(np.mean([t.success or 0.0 for t in trajectories]))
        total_tokens = sum(
            (tu.turn_end - tu.turn_start)
            for t in trajectories for tu in t.turns
            if tu.role == "assistant" and tu.full_token_ids is not None
        )

        # Free trajectory tensors aggressively.
        for traj in trajectories:
            for turn in traj.turns:
                turn.hidden_states = None
                turn.attentions = None
                turn.full_token_ids = None
        del trajectories, reward_outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        m = StepMetrics(
            step=step,
            mean_reward=mean_reward,
            mean_advantage=mean_advantage,
            mean_success=mean_success,
            rollout_seconds=rollout_sec,
            reward_seconds=reward_sec,
            update_seconds=update_sec,
            total_tokens=total_tokens,
            policy_loss=loss,
            kl_from_init=getattr(self, "_last_kl", 0.0),
        )
        self._log_step(m)

        if (step + 1) % self.cfg.save_every == 0:
            self.policy.save_lora(
                str(Path(self.cfg.output_dir) / f"ckpt_step{step + 1}")
            )

    def _log_step(self, m: StepMetrics) -> None:
        if m.step % self.cfg.log_every == 0:
            logger.info(
                f"step={m.step} "
                f"reward={m.mean_reward:.3f} "
                f"adv={m.mean_advantage:+.3f} "
                f"success={m.mean_success:.3f} "
                f"t_rollout={m.rollout_seconds:.1f}s "
                f"t_reward={m.reward_seconds:.1f}s "
                f"t_update={m.update_seconds:.1f}s "
                f"loss={m.policy_loss:+.4f}"
            )
        with open(self._metrics_file, "a") as fp:
            fp.write(json.dumps({
                "step": m.step,
                "mean_reward": m.mean_reward,
                "mean_advantage": m.mean_advantage,
                "mean_success": m.mean_success,
                "rollout_seconds": m.rollout_seconds,
                "reward_seconds": m.reward_seconds,
                "update_seconds": m.update_seconds,
                "total_tokens": m.total_tokens,
                "policy_loss": m.policy_loss,
                "kl_from_init": m.kl_from_init,
            }) + "\n")
