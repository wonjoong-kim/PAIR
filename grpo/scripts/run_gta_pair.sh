#!/usr/bin/env bash
# Train PAIR on GTA with the canonical recipe.
# Outputs to $PAIR_ROOT/runs/<policy>_gta_<reward>/.

set -euo pipefail
cd "$(dirname "$0")/../.."

POLICY="${POLICY:-qwen7b}"
REWARD="${REWARD:-pair_momentum}"  # pair | pair_momentum | outcome   (paper headline uses pair_momentum)
TRAIN_MODE="${TRAIN_MODE:-mixed}"
STEPS="${STEPS:-500}"
BATCH="${BATCH:-1}"
GROUP="${GROUP:-4}"
LR="${LR:-3e-7}"

python -m grpo.scripts.run_single \
    --policy "$POLICY" --env gta --reward "$REWARD" \
    --train_mode "$TRAIN_MODE" \
    --steps "$STEPS" --batch_size "$BATCH" --group_size "$GROUP" \
    --lr "$LR" \
    --output_dir "runs/${POLICY}_gta_${REWARD}"
