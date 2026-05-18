#!/usr/bin/env bash
# Train and evaluate PAIR probes for both datasets, both train modes.
# Assumes features have been extracted via extract_{gta,toolbench}.sh.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-qwen7b}"

echo "=== PAIR (Stage 1 + Stage 2 LR) ==="
python -m probing.train_pair \
    --models "$MODEL" \
    --datasets gta toolbench \
    --train-modes clean_only mixed

echo "=== Evaluation ==="
for ds in gta toolbench; do
    python -m probing.eval_pair --model "$MODEL" --dataset "$ds" --all \
        --output "data/results/eval_${MODEL}_${ds}.csv"
done
