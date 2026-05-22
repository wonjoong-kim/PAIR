"""Train PAIR — two-stage linear probe.

PAIR classifies each assistant turn as clean/contaminated using a
two-stage logistic-regression cascade:

    Stage 1 (hidden):       w_1^T h_t + b_1                → s_bc = σ(·)
    Stage 2 (correction):   w_2^T [a_t ; s_bc] + b_2     → s_final = σ(·)

Canonical configuration (matches the paper):

    hidden feature    = last_token
    attention feature = multi_layer_attn
    penalty           = L2
    C                 = 0.01  (strong regularization)
    train data        = matched_clean_train (and matched_contaminated_train if mixed)

Output:
    <PAIR_ROOT>/data/models/methods/PAIR/{model}/{dataset}/pair_{train_mode}.pkl

The bundle stores both stages plus the training config, so a single
unpickle is enough to reload at inference time (see grpo/rewards/pair.py).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import FEATURES_DIR, PAIR_DIR
from utils import set_seed, setup_logging

logger = setup_logging("train_pair")

SEED = 42
HIDDEN_FEATURE = "last_token"
ATTN_FEATURE = "multi_layer_attn"
C_VALUE = 0.01

TRAIN_MODES = {
    "clean_only": ["matched_clean_train"],
    "mixed": ["matched_clean_train", "matched_contaminated_train"],
}


def load_features(model, dataset, split, feature_type) -> np.ndarray:
    path = FEATURES_DIR / model / dataset / split / f"features_{feature_type}.npz"
    X = np.load(path, allow_pickle=True)["X"].astype(np.float64)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def load_labels(model, dataset, split) -> np.ndarray:
    return np.load(FEATURES_DIR / model / dataset / split / "labels.npy").astype(np.int32)


def load_balanced_indices(model, dataset) -> np.ndarray:
    """Indices that down-sample the majority class so both labels are balanced.
    Optional — falls back to all indices if the file is missing."""
    path = FEATURES_DIR / model / dataset / "matched_clean_train" / "balanced_indices.npy"
    if path.exists():
        return np.load(path)
    # Fallback: use all indices.
    return np.arange(len(load_labels(model, dataset, "matched_clean_train")))


def make_lr() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            solver="saga",
            max_iter=2000,
            random_state=SEED,
            C=C_VALUE,
        )),
    ])


def train_one(model: str, dataset: str, train_mode: str) -> Path:
    train_splits = TRAIN_MODES[train_mode]
    bal_idx = load_balanced_indices(model, dataset)

    Xh_parts, Xa_parts, y_parts = [], [], []
    for s in train_splits:
        Xh_parts.append(load_features(model, dataset, s, HIDDEN_FEATURE)[bal_idx])
        Xa_parts.append(load_features(model, dataset, s, ATTN_FEATURE)[bal_idx])
        y_parts.append(load_labels(model, dataset, s)[bal_idx])

    Xh = np.concatenate(Xh_parts)
    Xa = np.concatenate(Xa_parts)
    y = np.concatenate(y_parts)
    logger.info(
        f"{model}/{dataset}/{train_mode}: "
        f"n={len(y)} (pos={int(y.sum())}, neg={int((y==0).sum())}) "
        f"hidden_dim={Xh.shape[1]} attn_dim={Xa.shape[1]}"
    )

    # Stage 1: hidden → s_bc
    probe_base = make_lr().fit(Xh, y)
    s_bc = probe_base.predict_proba(Xh)[:, 1]

    # Stage 2: [attention, s_bc] → s_final
    X_stage2 = np.column_stack([Xa, s_bc])
    probe_correction = make_lr().fit(X_stage2, y)

    out_path = PAIR_DIR / model / dataset / f"pair_{train_mode}.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "probe_base": probe_base,
            "probe_correction": probe_correction,
            "config": {
                "hidden_feature": HIDDEN_FEATURE,
                "attn_feature": ATTN_FEATURE,
                "penalty": "l2",
                "C": C_VALUE,
                "seed": SEED,
                "train_splits": train_splits,
            },
        }, f)
    logger.info(f"  saved → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen7b"],
                        choices=["llama8b", "qwen7b", "mistral7b"])
    parser.add_argument("--datasets", nargs="+", default=["gta", "toolbench"])
    parser.add_argument("--train-modes", nargs="+", default=list(TRAIN_MODES.keys()),
                        choices=list(TRAIN_MODES.keys()))
    args = parser.parse_args()

    set_seed(SEED)
    n_done = 0
    for model in args.models:
        for dataset in args.datasets:
            for train_mode in args.train_modes:
                try:
                    train_one(model, dataset, train_mode)
                    n_done += 1
                except FileNotFoundError as e:
                    logger.warning(f"  SKIP {model}/{dataset}/{train_mode}: {e}")
    logger.info(f"\n=== Done: {n_done} PAIR probe(s) trained ===")


if __name__ == "__main__":
    main()
