#!/usr/bin/env bash
# Extract all feature types for ToolBench across the three reference policies.
# Outputs to $PAIR_ROOT/data/features/{model}/toolbench/{split}/*.npz

set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=("${MODELS[@]:-qwen7b llama8b mistral7b}")
for m in $MODELS; do
    echo "=== $m / toolbench ==="
    python -m probing.extract_features        --model "$m" --dataset toolbench
    python -m probing.extract_multi_layer_attn --model "$m" --dataset toolbench
done
