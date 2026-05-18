#!/usr/bin/env bash
# Train PAIR on ToolBench with the canonical recipe.
# Outputs to $PAIR_ROOT/runs/<policy>_toolbench_<reward>/.

set -euo pipefail
cd "$(dirname "$0")/../.."

POLICY="${POLICY:-qwen7b}"
REWARD="${REWARD:-pair_momentum}"  # pair | pair_momentum | outcome   (paper headline uses pair_momentum)
TRAIN_MODE="${TRAIN_MODE:-mixed}"
STEPS="${STEPS:-500}"
BATCH="${BATCH:-1}"
GROUP="${GROUP:-4}"
LR="${LR:-3e-7}"

# Fuzzy tool-arg matching helps when the policy diverges slightly from the
# replay cache's recorded arguments — fall back to any same-tool response.
export TB_ARGS_FUZZY="${TB_ARGS_FUZZY:-1}"

python -m grpo.scripts.run_single \
    --policy "$POLICY" --env toolbench --reward "$REWARD" \
    --train_mode "$TRAIN_MODE" \
    --steps "$STEPS" --batch_size "$BATCH" --group_size "$GROUP" \
    --lr "$LR" \
    --output_dir "runs/${POLICY}_toolbench_${REWARD}"
