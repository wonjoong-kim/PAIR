#!/usr/bin/env bash
# Extract all feature types for GTA across the three reference policies.
# Outputs to $PAIR_ROOT/data/features/{model}/gta/{split}/*.npz

set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=("${MODELS[@]:-qwen7b llama8b mistral7b}")
for m in $MODELS; do
    echo "=== $m / gta ==="
    python -m probing.extract_features        --model "$m" --dataset gta
    python -m probing.extract_multi_layer_attn --model "$m" --dataset gta
done
