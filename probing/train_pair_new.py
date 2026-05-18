"""Train PAIR-NEW — logit-space residual variant of PAIR.

PAIR (score-space concat, two sigmoids):
    s_base  = σ(w_1^T h + b_1)
    s_final = σ(w_2^T [a ; s_base] + b_2)

PAIR-NEW (logit-space residual, single sigmoid on the sum):
    z_base      = w_1^T h + b_1            (Stage 1 logit, frozen)
    s_base      = σ(z_base)
    delta_logit = w_2^T [a_scaled ; s_base] + b_2
    s_final     = σ(z_base + delta_logit)

Properties:
    * delta_logit = 0  ⇒  s_final = s_base exactly (a perfect Stage-1 prefix
      is preserved by construction, not by training).
    * Single sigmoid on the final output (no double squashing).

Stage 1 is loaded as-is from `train_pair.py`'s output bundle, so the two
methods share the same Stage 1 — only Stage 2 differs.

Output:
    <PAIR_ROOT>/data/models/methods/PAIR_NEW/{model}/{dataset}/pair_new_{train_mode}.pkl
    <PAIR_ROOT>/data/results/methods/pair_new_{model}_{dataset}.csv  (eval results)
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import FEATURES_DIR, PAIR_DIR, PAIR_NEW_DIR, RESULTS_DIR
from utils import set_seed, setup_logging

logger = setup_logging("train_pair_new")

SEED = 42

HIDDEN_FEATURE = "last_token"
ATTN_FEATURE = "multi_layer_attn"

TRAIN_MODES = {
    "clean_only": ["matched_clean_train"],
    "mixed": ["matched_clean_train", "matched_contaminated_train"],
}
EVAL_SPLITS = ["matched_clean_test", "matched_contaminated_test"]

# Best per-dataset recipes (see paper §X.Y). Override via CLI when sweeping.
DEFAULTS = {
    "gta":       dict(optimizer="adamw", schedule="constant", lr=1e-2,  epochs=50,   weight_decay=1e-3,  momentum=0.9, l2_lambda=0.0),
    "toolbench": dict(optimizer="sgd",   schedule="cosine",   lr=3e-2,  epochs=1000, weight_decay=0.0,   momentum=0.9, l2_lambda=0.0),
}


class LogitResidualHead(nn.Module):
    def __init__(self, attn_dim: int):
        super().__init__()
        self.linear = nn.Linear(attn_dim + 1, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, a: torch.Tensor, s_base: torch.Tensor) -> torch.Tensor:
        x = torch.cat([a, s_base], dim=1)
        return self.linear(x).squeeze(-1)


def load_features(model, dataset, split, feature_type) -> np.ndarray:
    path = FEATURES_DIR / model / dataset / split / f"features_{feature_type}.npz"
    X = np.load(path, allow_pickle=True)["X"].astype(np.float64)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def load_labels(model, dataset, split) -> np.ndarray:
    return np.load(FEATURES_DIR / model / dataset / split / "labels.npy").astype(np.int32)


def load_balanced_indices(model, dataset) -> np.ndarray:
    path = FEATURES_DIR / model / dataset / "matched_clean_train" / "balanced_indices.npy"
    if path.exists():
        return np.load(path)
    return np.arange(len(load_labels(model, dataset, "matched_clean_train")))


def load_stage1_probe(model, dataset, train_mode):
    """Reuse the Stage 1 probe from `train_pair.py`'s output bundle."""
    path = PAIR_DIR / model / dataset / f"pair_{train_mode}.pkl"
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["probe_base"], bundle.get("probe_correction")


def expected_calibration_error(y_true, p, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - p[mask].mean())
    return float(ece)


def metrics(y, p) -> Dict[str, float]:
    return {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "ECE": expected_calibration_error(np.asarray(y), np.asarray(p)),
        "Brier": float(brier_score_loss(y, p)),
    }


def build_stage2(z_base_train, s_base_train, Xa_train_scaled, y_train, attn_dim,
                 optimizer="adamw", schedule="constant",
                 epochs=50, lr=1e-2, weight_decay=1e-3,
                 momentum=0.9, l2_lambda=0.0) -> LogitResidualHead:
    z_t = torch.from_numpy(z_base_train.astype(np.float32))
    s_t = torch.from_numpy(s_base_train.astype(np.float32)).unsqueeze(1)
    Xa_t = torch.from_numpy(Xa_train_scaled.astype(np.float32))
    y_t = torch.from_numpy(y_train.astype(np.float32))

    head = LogitResidualHead(attn_dim)
    bce = nn.BCEWithLogitsLoss()

    if optimizer == "lbfgs":
        optim = torch.optim.LBFGS(head.parameters(), max_iter=epochs,
                                  history_size=20, line_search_fn="strong_wolfe")
        last = {"loss": None}
        def closure():
            optim.zero_grad()
            delta = head(Xa_t, s_t)
            loss = bce(z_t.detach() + delta, y_t)
            if l2_lambda > 0:
                loss = loss + l2_lambda * (head.linear.weight ** 2).sum()
            loss.backward()
            last["loss"] = float(loss.item())
            return loss
        head.train()
        optim.step(closure)
        logger.info(f"    final loss={last['loss']:.4f} (lbfgs, max_iter={epochs})")
        head.eval()
        return head

    if optimizer == "adamw":
        optim = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        optim = torch.optim.SGD(head.parameters(), lr=lr, momentum=momentum,
                                nesterov=momentum > 0, weight_decay=weight_decay)
    else:
        raise ValueError(f"unknown optimizer: {optimizer}")

    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
             if schedule == "cosine" else None)

    head.train()
    for epoch in range(epochs):
        optim.zero_grad()
        delta = head(Xa_t, s_t)
        loss = bce(z_t.detach() + delta, y_t)
        if l2_lambda > 0:
            loss = loss + l2_lambda * (head.linear.weight ** 2).sum()
        loss.backward()
        optim.step()
        if sched is not None:
            sched.step()
        if epochs >= 100 and (epoch + 1) % max(1, epochs // 4) == 0:
            logger.info(f"    epoch {epoch+1:>4d}: loss={loss.item():.4f}")
    logger.info(f"    final loss={loss.item():.4f} ({optimizer}, schedule={schedule}, "
                f"{epochs} epochs, lr={lr}, wd={weight_decay})")
    head.eval()
    return head


def predict_pair_new(probe_base, attn_scaler, head, Xh, Xa):
    z_base = probe_base.decision_function(Xh)
    s_base = probe_base.predict_proba(Xh)[:, 1]
    Xa_s = attn_scaler.transform(Xa)
    with torch.no_grad():
        delta = head(
            torch.from_numpy(Xa_s.astype(np.float32)),
            torch.from_numpy(s_base.astype(np.float32)).unsqueeze(1),
        ).numpy()
    return 1.0 / (1.0 + np.exp(-(z_base + delta))), s_base


def predict_pair(probe_base, probe_correction, Xh, Xa):
    s_base = probe_base.predict_proba(Xh)[:, 1]
    X_stage2 = np.column_stack([Xa, s_base])
    return probe_correction.predict_proba(X_stage2)[:, 1], s_base


def run_one(model: str, dataset: str, train_mode: str, hp: dict, rows: list) -> None:
    logger.info(f"\n=== {model}/{dataset}/{train_mode} ===")
    bal = load_balanced_indices(model, dataset)
    train_splits = TRAIN_MODES[train_mode]

    Xh_parts, Xa_parts, y_parts = [], [], []
    for s in train_splits:
        Xh_parts.append(load_features(model, dataset, s, HIDDEN_FEATURE)[bal])
        Xa_parts.append(load_features(model, dataset, s, ATTN_FEATURE)[bal])
        y_parts.append(load_labels(model, dataset, s)[bal])
    Xh = np.concatenate(Xh_parts)
    Xa = np.concatenate(Xa_parts)
    y = np.concatenate(y_parts)
    logger.info(f"  train n={len(y)} (pos={int(y.sum())}, neg={int((y==0).sum())}), "
                f"hidden={Xh.shape[1]}d, attn={Xa.shape[1]}d")

    probe_base, probe_correction_old = load_stage1_probe(model, dataset, train_mode)
    z_base_train = probe_base.decision_function(Xh)
    s_base_train = probe_base.predict_proba(Xh)[:, 1]

    attn_scaler = StandardScaler().fit(Xa)
    Xa_train_s = attn_scaler.transform(Xa)

    set_seed(SEED)
    head = build_stage2(
        z_base_train, s_base_train, Xa_train_s, y,
        attn_dim=Xa.shape[1], **hp,
    )

    out_path = PAIR_NEW_DIR / model / dataset / f"pair_new_{train_mode}.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "probe_base": probe_base,
            "attn_scaler": attn_scaler,
            "head_state_dict": head.state_dict(),
            "head_arch": {"attn_dim": int(Xa.shape[1])},
            "config": {
                "hidden_feature": HIDDEN_FEATURE,
                "attn_feature": ATTN_FEATURE,
                "stage2": "logit_residual_linear",
                "stage1_inherited_from": str(PAIR_DIR / model / dataset / f"pair_{train_mode}.pkl"),
                "seed": SEED,
                **hp,
            },
        }, f)
    logger.info(f"  saved → {out_path}")

    for split in EVAL_SPLITS:
        Xh_eval = load_features(model, dataset, split, HIDDEN_FEATURE)
        Xa_eval = load_features(model, dataset, split, ATTN_FEATURE)
        y_eval = load_labels(model, dataset, split)
        p_new, s_base_eval = predict_pair_new(probe_base, attn_scaler, head, Xh_eval, Xa_eval)
        m_new = metrics(y_eval, p_new)
        m_base = metrics(y_eval, s_base_eval)
        m_pair = (metrics(y_eval,
                          predict_pair(probe_base, probe_correction_old, Xh_eval, Xa_eval)[0])
                  if probe_correction_old is not None
                  else {k: float("nan") for k in m_new})

        logger.info(f"  [{split}]")
        logger.info(f"    Stage-1 only       {fmt(m_base)}")
        logger.info(f"    PAIR               {fmt(m_pair)}")
        logger.info(f"    PAIR-NEW           {fmt(m_new)}")

        for variant, m in [("stage1_only", m_base), ("pair", m_pair), ("pair_new", m_new)]:
            rows.append({"model": model, "dataset": dataset, "train_mode": train_mode,
                         "variant": variant, "test_split": split, **m})


def fmt(m: Dict[str, float]) -> str:
    return f"AUROC={m['AUROC']:.4f} AUPRC={m['AUPRC']:.4f} ECE={m['ECE']:.4f} Brier={m['Brier']:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen7b", choices=["llama8b", "qwen7b", "mistral7b"])
    parser.add_argument("--datasets", nargs="+", default=["gta", "toolbench"])
    parser.add_argument("--train-modes", nargs="+", default=list(TRAIN_MODES.keys()),
                        choices=list(TRAIN_MODES.keys()))
    parser.add_argument("--optimizer", choices=["adamw", "sgd", "lbfgs"], default=None)
    parser.add_argument("--schedule", choices=["constant", "cosine"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--l2-lambda", type=float, default=None)
    args = parser.parse_args()

    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        # Pick dataset default; let CLI override per-field.
        hp = dict(DEFAULTS.get(dataset, DEFAULTS["gta"]))
        for key in ("optimizer", "schedule", "epochs", "lr", "weight_decay",
                    "momentum", "l2_lambda"):
            cli_val = getattr(args, key, None)
            if cli_val is not None:
                hp[key] = cli_val
        logger.info(f"[{args.model}/{dataset}] Stage-2 config: {hp}")

        rows = []
        for train_mode in args.train_modes:
            run_one(args.model, dataset, train_mode, hp, rows)

        out_csv = RESULTS_DIR / f"pair_new_{args.model}_{dataset}.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        logger.info(f"[{dataset}] results → {out_csv}")


if __name__ == "__main__":
    main()
