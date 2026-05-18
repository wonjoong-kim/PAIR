"""Extract hidden-state and attention features for offline PAIR probe training.

Given a dataset of multi-turn agentic dialogs with per-turn binary labels
(1 = clean / correct, 0 = contaminated), this script runs the base LM over
each assistant turn's prefix, captures hidden states + attention, and saves
six feature types:

  1. last_token       — last layer, last token hidden state            (hidden_dim,)
  2. mean_pooled      — last layer, mean over current-turn tokens      (hidden_dim,)
  3. multi_layer      — last 4 layers, last token concatenated         (4 * hidden_dim,)
  4. raw_attention    — last-layer per-head stats                      (4 * num_heads,)
  5. lookback_ratio   — last 4 layers per-head prefix ratio            (4 * num_heads,)
  6. hidden_attn      — last_token + raw_attention                     (hidden_dim + 4*num_heads,)

These six feature tensors plus the multi-layer-attention extractor
(`extract_multi_layer_attn.py`) cover everything the PAIR pipeline needs.

Outputs to `<PAIR_ROOT>/data/features/{model}/{dataset}/{split}/features_<type>.npz`.

Usage:
    python -m probing.extract_features --model qwen7b --dataset gta
    python -m probing.extract_features --model llama8b --dataset toolbench --split clean_train
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import DATASET_DIRS, FEATURES_DIR, model_path
from utils import (
    build_chat_messages,
    get_assistant_turn_indices,
    load_json,
    merge_consecutive_roles,
    set_seed,
    setup_logging,
)

logger = setup_logging("extract_features")

RANDOM_SEED = 42
MAX_SEQ_LEN = 4096
NUM_LAST_LAYERS = 4

FEATURE_TYPES = [
    "last_token", "mean_pooled", "multi_layer",
    "raw_attention", "lookback_ratio", "hidden_attn",
]

ALL_SPLITS = [
    "clean_train", "contaminated_train", "clean_test", "contaminated_test",
    "matched_clean_train", "matched_contaminated_train",
    "matched_clean_test", "matched_contaminated_test",
]


def load_model_and_tokenizer(path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer, device


def find_turn_token_boundaries(tokenizer, dialogs, target_turn_idx, max_len):
    messages_full = merge_consecutive_roles(
        build_chat_messages(dialogs, up_to_idx=target_turn_idx)
    )
    messages_prefix = (
        merge_consecutive_roles(
            build_chat_messages(dialogs, up_to_idx=target_turn_idx - 1)
        )
        if target_turn_idx > 0 else []
    )

    try:
        if hasattr(tokenizer, "apply_chat_template"):
            text_full = tokenizer.apply_chat_template(
                messages_full, tokenize=False, add_generation_prompt=False
            )
            text_prefix = (
                tokenizer.apply_chat_template(
                    messages_prefix, tokenize=False, add_generation_prompt=False
                )
                if messages_prefix else ""
            )
        else:
            text_full = "\n".join(m["content"] for m in messages_full)
            text_prefix = "\n".join(m["content"] for m in messages_prefix) if messages_prefix else ""

        tokens_full = tokenizer.encode(text_full, add_special_tokens=True)
        tokens_prefix = tokenizer.encode(text_prefix, add_special_tokens=True) if text_prefix else []

        if len(tokens_full) > max_len:
            tokens_full = tokens_full[-max_len:]
            if len(tokens_prefix) > max_len:
                tokens_prefix = tokens_prefix[-max_len:]

        turn_start = len(tokens_prefix) if tokens_prefix else 0
        turn_end = len(tokens_full)
        if turn_start >= turn_end:
            turn_start = max(0, turn_end - 1)
        return tokens_full, turn_start, turn_end
    except Exception as e:
        logger.warning(f"Tokenization error: {e}")
        return None, None, None


def compute_raw_attention(attentions, turn_start, turn_end, layer_idx=-1):
    attn = attentions[layer_idx][0].float()
    num_heads = attn.shape[0]
    if turn_start >= turn_end:
        turn_end = turn_start + 1
    turn_attn = attn[:, turn_start:turn_end, :]

    max_attn = turn_attn.max(dim=-1).values.mean(dim=-1)
    std_attn = turn_attn.std(dim=-1).mean(dim=-1)
    total = turn_attn.sum(dim=-1).clamp(min=1e-10)
    if turn_start > 0:
        prefix_ratio = (turn_attn[:, :, :turn_start].sum(dim=-1) / total).mean(dim=-1)
    else:
        prefix_ratio = torch.zeros(num_heads, device=attn.device)
    self_ratio = (turn_attn[:, :, turn_start:turn_end].sum(dim=-1) / total).mean(dim=-1)
    return torch.cat([max_attn, std_attn, prefix_ratio, self_ratio]).cpu().numpy().astype(np.float32)


def compute_lookback_ratio(attentions, turn_start, turn_end, num_last_layers=NUM_LAST_LAYERS):
    n_layers = len(attentions)
    if turn_start >= turn_end:
        turn_end = turn_start + 1
    feats: List[torch.Tensor] = []
    for li in range(max(0, n_layers - num_last_layers), n_layers):
        attn = attentions[li][0].float()
        num_heads = attn.shape[0]
        turn_attn = attn[:, turn_start:turn_end, :]
        total = turn_attn.sum(dim=-1).clamp(min=1e-10)
        if turn_start > 0:
            lb = (turn_attn[:, :, :turn_start].sum(dim=-1) / total).mean(dim=-1)
        else:
            lb = torch.zeros(num_heads, device=attn.device)
        feats.append(lb)
    return torch.cat(feats).cpu().numpy().astype(np.float32)


@torch.no_grad()
def extract_turn_features(model, tokenizer, device, dialogs, turn_idx, max_len):
    input_ids, turn_start, turn_end = find_turn_token_boundaries(
        tokenizer, dialogs, turn_idx, max_len
    )
    if input_ids is None:
        return None

    tensor = torch.tensor([input_ids], device=device)
    try:
        out = model(tensor, output_hidden_states=True, output_attentions=True)
    except Exception as e:
        logger.warning(f"Forward pass error: {e}")
        return None

    hs = out.hidden_states
    attentions = out.attentions
    last_idx = len(input_ids) - 1
    last_hs = hs[-1][0]

    feats = {}
    feats["last_token"] = last_hs[last_idx].cpu().numpy().astype(np.float32)
    feats["mean_pooled"] = last_hs[turn_start:turn_end].mean(dim=0).cpu().numpy().astype(np.float32)

    multi: List[np.ndarray] = []
    for li in range(max(0, len(hs) - NUM_LAST_LAYERS), len(hs)):
        multi.append(hs[li][0][last_idx].cpu().numpy().astype(np.float32))
    feats["multi_layer"] = np.concatenate(multi)

    feats["raw_attention"] = compute_raw_attention(attentions, turn_start, turn_end)
    feats["lookback_ratio"] = compute_lookback_ratio(attentions, turn_start, turn_end)
    feats["hidden_attn"] = np.concatenate([feats["last_token"], feats["raw_attention"]])

    del out, hs, attentions, tensor
    if device == "cuda":
        torch.cuda.empty_cache()
    return feats


def process_split(model, tokenizer, device, episodes_or_records, is_contaminated, max_len):
    """Process either episode-level dialogs (clean_train/contaminated_train)
    or matched per-turn records (matched_* splits)."""
    features_by_type = {ft: [] for ft in FEATURE_TYPES}
    labels: List[int] = []
    episode_ids: List[str] = []

    # Matched splits: list-of-records with {dialogs, turn_idx, label, episode_id}.
    if isinstance(episodes_or_records, list):
        for record in tqdm(episodes_or_records, desc="Records", leave=False):
            feats = extract_turn_features(
                model, tokenizer, device,
                record["dialogs"], record["turn_idx"], max_len,
            )
            if feats is None:
                continue
            for ft in FEATURE_TYPES:
                features_by_type[ft].append(feats[ft])
            labels.append(record["label"])
            episode_ids.append(record["episode_id"])
            gc.collect()
        return features_by_type, labels, episode_ids

    # Episode dict: {eid -> {dialogs, contaminated_turn_indices, ...}}.
    for eid, episode in tqdm(episodes_or_records.items(), desc="Episodes", leave=False):
        dialogs = episode["dialogs"]
        contaminated = set()
        if is_contaminated:
            raw = episode.get("contaminated_turn_indices")
            if isinstance(raw, list):
                contaminated = set(raw)
            else:
                old = episode.get("contaminated_turn_idx", -1)
                if old >= 0:
                    contaminated = {old}

        for turn_idx in get_assistant_turn_indices(dialogs):
            feats = extract_turn_features(model, tokenizer, device, dialogs, turn_idx, max_len)
            if feats is None:
                continue
            for ft in FEATURE_TYPES:
                features_by_type[ft].append(feats[ft])
            labels.append(0 if (is_contaminated and turn_idx in contaminated) else 1)
            episode_ids.append(eid)
        gc.collect()
    return features_by_type, labels, episode_ids


def save_split(features_by_type, labels, episode_ids, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_arr = np.array(labels, dtype=np.int32)
    eids_arr = np.array(episode_ids)

    for ft in FEATURE_TYPES:
        if not features_by_type[ft]:
            continue
        X = np.stack(features_by_type[ft])
        path = output_dir / f"features_{ft}.npz"
        np.savez_compressed(path, X=X, y=labels_arr, episode_ids=eids_arr)
        logger.info(f"  Saved {path.name} shape={X.shape} "
                    f"pos={int((labels_arr==1).sum())} neg={int((labels_arr==0).sum())}")
    np.save(output_dir / "labels.npy", labels_arr)


def is_split_complete(output_dir: Path) -> bool:
    if not output_dir.exists():
        return False
    return all((output_dir / f"features_{ft}.npz").exists() for ft in FEATURE_TYPES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["llama8b", "qwen7b", "mistral7b"])
    parser.add_argument("--dataset", required=True, choices=list(DATASET_DIRS.keys()))
    parser.add_argument("--split", default=None, choices=ALL_SPLITS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    data_dir = DATASET_DIRS[args.dataset]
    splits = [args.split] if args.split else ALL_SPLITS

    available = [s for s in splits if (data_dir / f"{s}.json").exists()]
    if not available:
        logger.error(f"No data found in {data_dir} for splits {splits}")
        return

    todo = [s for s in available
            if args.force or not is_split_complete(FEATURES_DIR / args.model / args.dataset / s)]
    if not todo:
        logger.info("All requested splits already complete.")
        return

    model, tokenizer, device = load_model_and_tokenizer(model_path(args.model))

    for split in todo:
        data_path = data_dir / f"{split}.json"
        data = load_json(data_path)
        is_contaminated = "contaminated" in split
        logger.info(f"\n[{args.model}/{args.dataset}/{split}] is_contaminated={is_contaminated}")
        features, labels, eids = process_split(
            model, tokenizer, device, data, is_contaminated, MAX_SEQ_LEN
        )
        if not labels:
            logger.warning("  No features extracted; skipping save.")
            continue
        out_dir = FEATURES_DIR / args.model / args.dataset / split
        save_split(features, labels, eids, out_dir)

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
