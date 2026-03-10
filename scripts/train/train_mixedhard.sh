#!/bin/bash
# Train MixedHard (ObsDelay=3, ActDelay=1)
# Usage: ./scripts/train_mixedhard.sh [METHOD_TAG] [SEED] [RUN_ID_SUFFIX]

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

METHOD_TAG=${1:-"baseline_mixedhard"}
SEED=${2:-0}
SUFFIX=${3:-""}

# Define root output directories
OUT_ROOT="runs"
FIG_ROOT="figures"

# Generate Run ID (or let Python do it, but we need it for redirection)
# To ensure Python and Bash agree, we can pass it explicitly.
# Format: <method>__<env>__seed<seed>__<timestamp>
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ENV_ID="FetchPickAndPlace-v4"
CLEAN_ENV="PnP"

if [ -z "$SUFFIX" ]; then
    RUN_ID="${METHOD_TAG}__${CLEAN_ENV}__seed${SEED}__${TIMESTAMP}"
else
    RUN_ID="${METHOD_TAG}__${CLEAN_ENV}__seed${SEED}__${TIMESTAMP}_${SUFFIX}"
fi

# Create directory explicitly in bash to allow redirection before python starts
mkdir -p "$OUT_ROOT/$RUN_ID"

LOG_FILE="$OUT_ROOT/$RUN_ID/stdout.log"

echo "Starting training..."
echo "Run ID: $RUN_ID"
echo "Logging to: $LOG_FILE"

# Run training
# Note: We use -u for unbuffered output so it shows up in log immediately
python -u train.py \
    --env $ENV_ID \
    --seed $SEED \
    --out-root $OUT_ROOT \
    --fig-root $FIG_ROOT \
    --run-id $RUN_ID \
    --method-tag $METHOD_TAG \
    --total-timesteps 2000000 \
    --n-envs 16 \
    --vec-env subproc \
    --eval-freq 100000 \
    --eval-freq-paper 100000 \
    --n-eval-paper 50 \
    --paper-eval-settings "Nominal,ObsOnly3,ActOnly1,MixedHard" \
    --obs-delay-k 3 \
    --action-delay-k 1 \
    "${@:4}" \
    > "$LOG_FILE" 2>&1

echo "Training finished. Check results in $OUT_ROOT/$RUN_ID"


