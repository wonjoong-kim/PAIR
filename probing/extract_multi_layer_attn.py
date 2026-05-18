"""Extract multi-layer attention features used by PAIR Stage 2.

Computes 4 per-head stats (max, std, prefix_ratio, self_ratio) over the
target assistant turn's attention distributions, separately for every
transformer layer (not just the last one). Output dim = 4 × num_heads × num_layers.

Reuses the same boundary-detection logic as `extract_features.py` but only
runs a single forward pass per turn and only stores the single
`features_multi_layer_attn.npz`.

Usage:
    python -m probing.extract_multi_layer_attn --model qwen7b --dataset gta
    python -m probing.extract_multi_layer_attn --model llama8b --dataset toolbench --split matched_clean_train
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import List, Optional

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

logger = setup_logging("extract_multi_layer_attn")

RANDOM_SEED = 42
MAX_SEQ_LEN = 4096

ALL_SPLITS = [
    "matched_clean_train", "matched_contaminated_train",
    "matched_clean_test", "matched_contaminated_test",
    "clean_train", "contaminated_train", "clean_test", "contaminated_test",
]


def load_model_and_tokenizer(path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer, device


def find_turn_boundaries(tokenizer, dialogs, target_turn_idx, max_len):
    messages_full = merge_consecutive_roles(
        build_chat_messages(dialogs, up_to_idx=target_turn_idx)
    )
    messages_prefix = (
        merge_consecutive_roles(
            build_chat_messages(dialogs, up_to_idx=target_turn_idx - 1)
        )
        if target_turn_idx > 0 else []
    )

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


def compute_multi_layer_attn(attentions, turn_start, turn_end):
    """4 stats per head, for every layer. Returns (4 * num_heads * num_layers,)."""
    if turn_start >= turn_end:
        turn_end = turn_start + 1
    feats: List[np.ndarray] = []
    for layer_idx in range(len(attentions)):
        attn = attentions[layer_idx][0].float()
        num_heads = attn.shape[0]
        turn_attn = attn[:, turn_start:turn_end, :]

        max_attn = turn_attn.max(dim=-1).values.mean(dim=-1)
        std_attn = turn_attn.std(dim=-1).mean(dim=-1)
        total = turn_attn.sum(dim=-1).clamp(min=1e-10)
        if turn_start > 0:
            prefix_ratio = (turn_attn[:, :, :turn_start].sum(dim=-1) / total).mean(dim=-1)
        else:
            prefix_ratio = torch.zeros(num_heads, device=attn.device)
        self_ratio = (turn_attn[:, :, turn_start:turn_end].sum(dim=-1) / total).mean(dim=-1)

        feats.append(
            torch.cat([max_attn, std_attn, prefix_ratio, self_ratio])
            .cpu().numpy().astype(np.float32)
        )
    return np.concatenate(feats)


@torch.no_grad()
def extract_feature(model, tokenizer, device, dialogs, turn_idx, max_len):
    try:
        input_ids, turn_start, turn_end = find_turn_boundaries(
            tokenizer, dialogs, turn_idx, max_len
        )
    except Exception as e:
        logger.warning(f"Tokenization error: {e}")
        return None

    tensor = torch.tensor([input_ids], device=device)
    try:
        out = model(tensor, output_hidden_states=False, output_attentions=True)
    except Exception as e:
        logger.warning(f"Forward pass error: {e}")
        return None

    feat = compute_multi_layer_attn(out.attentions, turn_start, turn_end)
    del out, tensor
    if device == "cuda":
        torch.cuda.empty_cache()
    return feat


def process_records(model, tokenizer, device, records_or_episodes, is_contaminated, max_len):
    feats, labels, eids = [], [], []
    if isinstance(records_or_episodes, list):
        for r in tqdm(records_or_episodes, leave=False):
            f = extract_feature(model, tokenizer, device, r["dialogs"], r["turn_idx"], max_len)
            if f is None:
                continue
            feats.append(f)
            labels.append(r["label"])
            eids.append(r["episode_id"])
            gc.collect()
        return feats, labels, eids

    for eid, ep in tqdm(records_or_episodes.items(), leave=False):
        dialogs = ep["dialogs"]
        contaminated = set()
        if is_contaminated:
            raw = ep.get("contaminated_turn_indices")
            if isinstance(raw, list):
                contaminated = set(raw)
            else:
                old = ep.get("contaminated_turn_idx", -1)
                if old >= 0:
                    contaminated = {old}
        for turn_idx in get_assistant_turn_indices(dialogs):
            f = extract_feature(model, tokenizer, device, dialogs, turn_idx, max_len)
            if f is None:
                continue
            feats.append(f)
            labels.append(0 if (is_contaminated and turn_idx in contaminated) else 1)
            eids.append(eid)
        gc.collect()
    return feats, labels, eids


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
            if args.force or not (FEATURES_DIR / args.model / args.dataset / s
                                  / "features_multi_layer_attn.npz").exists()]
    if not todo:
        logger.info("All splits already done.")
        return

    model, tokenizer, device = load_model_and_tokenizer(model_path(args.model))

    for split in todo:
        data_path = data_dir / f"{split}.json"
        records = load_json(data_path)
        is_contaminated = "contaminated" in split
        logger.info(f"\n[{args.model}/{args.dataset}/{split}] is_contaminated={is_contaminated}")
        feats, labels, eids = process_records(
            model, tokenizer, device, records, is_contaminated, MAX_SEQ_LEN
        )
        if not feats:
            logger.warning("  No features extracted; skipping.")
            continue
        X = np.stack(feats)
        y = np.array(labels, dtype=np.int32)
        out_dir = FEATURES_DIR / args.model / args.dataset / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "features_multi_layer_attn.npz"
        np.savez_compressed(out_path, X=X, y=y, episode_ids=np.array(eids))
        logger.info(f"  Saved {out_path} shape={X.shape}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
