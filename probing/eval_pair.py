"""Evaluate a trained PAIR probe on test splits.

Loads pair_{train_mode}.pkl (Stage 1 + Stage 2), scores each test turn, and
prints classification metrics:

    AUROC, AUPRC, ECE (15 bins), Brier score, pairwise within-episode
    ranking accuracy (clean > contaminated).

Usage:
    python -m probing.eval_pair --model qwen7b --dataset gta --train-mode mixed
    python -m probing.eval_pair --model qwen7b --dataset toolbench --pair-new
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import FEATURES_DIR, PAIR_DIR, PAIR_NEW_DIR, RESULTS_DIR
from utils import (
    expected_calibration_error,
    pairwise_ranking_accuracy,
    setup_logging,
)

logger = setup_logging("eval_pair")

EVAL_SPLITS = ["matched_clean_test", "matched_contaminated_test"]


def load_features(model, dataset, split, feature_type) -> np.ndarray:
    path = FEATURES_DIR / model / dataset / split / f"features_{feature_type}.npz"
    X = np.load(path, allow_pickle=True)["X"].astype(np.float64)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def load_labels(model, dataset, split) -> np.ndarray:
    return np.load(FEATURES_DIR / model / dataset / split / "labels.npy").astype(np.int32)


def load_episode_ids(model, dataset, split) -> np.ndarray:
    path = FEATURES_DIR / model / dataset / split / "features_last_token.npz"
    return np.load(path, allow_pickle=True)["episode_ids"]


def score_pair(bundle, Xh, Xa) -> np.ndarray:
    """Stage 1 → s_base, Stage 2 → s_final."""
    s_base = bundle["probe_base"].predict_proba(Xh)[:, 1]
    X_stage2 = np.column_stack([Xa, s_base])
    return bundle["probe_correction"].predict_proba(X_stage2)[:, 1]


def score_pair_new(bundle, Xh, Xa) -> np.ndarray:
    """PAIR-NEW logit-residual: σ(z_base + delta_logit(Xa, s_base))."""
    from train_pair_new import LogitResidualHead

    probe_base = bundle["probe_base"]
    attn_scaler = bundle["attn_scaler"]
    head = LogitResidualHead(int(bundle["head_arch"]["attn_dim"]))
    head.load_state_dict(bundle["head_state_dict"])
    head.eval()

    z_base = probe_base.decision_function(Xh)
    s_base = probe_base.predict_proba(Xh)[:, 1]
    Xa_s = attn_scaler.transform(Xa)
    with torch.no_grad():
        delta = head(
            torch.from_numpy(Xa_s.astype(np.float32)),
            torch.from_numpy(s_base.astype(np.float32)).unsqueeze(1),
        ).numpy()
    return 1.0 / (1.0 + np.exp(-(z_base + delta)))


def evaluate_one(model: str, dataset: str, train_mode: str,
                 pair_new: bool, rows: List[Dict]) -> None:
    bundle_path = (PAIR_NEW_DIR if pair_new else PAIR_DIR) / model / dataset / (
        f"pair_new_{train_mode}.pkl" if pair_new else f"pair_{train_mode}.pkl"
    )
    if not bundle_path.exists():
        logger.warning(f"  SKIP — bundle not found: {bundle_path}")
        return
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    cfg = bundle.get("config", {})
    hidden_feat = cfg.get("hidden_feature", "last_token")
    attn_feat = cfg.get("attn_feature", "multi_layer_attn")

    logger.info(f"\n=== {('PAIR-NEW' if pair_new else 'PAIR')} | "
                f"{model}/{dataset}/{train_mode} ===")
    logger.info(f"  hidden={hidden_feat}  attn={attn_feat}")

    for split in EVAL_SPLITS:
        try:
            Xh = load_features(model, dataset, split, hidden_feat)
            Xa = load_features(model, dataset, split, attn_feat)
            y = load_labels(model, dataset, split)
            eids = load_episode_ids(model, dataset, split)
        except FileNotFoundError as e:
            logger.warning(f"  SKIP {split}: {e}")
            continue

        p = score_pair_new(bundle, Xh, Xa) if pair_new else score_pair(bundle, Xh, Xa)

        m = {
            "AUROC": float(roc_auc_score(y, p)),
            "AUPRC": float(average_precision_score(y, p)),
            "ECE": expected_calibration_error(y, p),
            "Brier": float(brier_score_loss(y, p)),
            "PairAcc": pairwise_ranking_accuracy(y, p, eids),
        }
        logger.info(f"  [{split}] AUROC={m['AUROC']:.4f} AUPRC={m['AUPRC']:.4f} "
                    f"ECE={m['ECE']:.4f} Brier={m['Brier']:.4f} "
                    f"PairAcc={m['PairAcc']:.4f}")
        rows.append({
            "model": model, "dataset": dataset, "train_mode": train_mode,
            "method": "pair_new" if pair_new else "pair",
            "test_split": split, **m,
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen7b", choices=["llama8b", "qwen7b", "mistral7b"])
    parser.add_argument("--dataset", default="gta", choices=["gta", "toolbench"])
    parser.add_argument("--train-mode", default="mixed", choices=["clean_only", "mixed"])
    parser.add_argument("--pair-new", action="store_true",
                        help="Evaluate PAIR-NEW (logit-residual) instead of PAIR.")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate PAIR and PAIR-NEW for both train modes.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows: List[Dict] = []
    if args.all:
        for tm in ("clean_only", "mixed"):
            for is_new in (False, True):
                evaluate_one(args.model, args.dataset, tm, is_new, rows)
    else:
        evaluate_one(args.model, args.dataset, args.train_mode, args.pair_new, rows)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        logger.info(f"\nResults → {out}")


if __name__ == "__main__":
    main()
