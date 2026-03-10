#!/bin/bash
# Ours: Adversarial Training Demo (Residual Memory + Budgeted Adversary)

# 0. Path & Dir Setup
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# 1. Environment
export ENV_NAME="mastering-rl"
export ENV_ID="FetchPickAndPlace-v4"
export LOG_DIR="logs/data_e2"

# 2. Training Config
export TOTAL_STEPS=1000000
export RUN_NAME="ours_adversary_demo"

# 3. Residual Config (ENABLED)
export USE_RESIDUAL_MEM=1
# Alpha schedule
export ALPHA_START="0.0"
export ALPHA_MID="0.2"
export ALPHA_FINAL="1.0"
export RESIDUAL_MEM_TYPE="gru"

# 4. Adversary Config
# Replaces --use-disturbance-curriculum with --use-adversary
# We set hard disturbance limits here, but adversary chooses actual values per episode.
export ADV_K_OBS_MAX="3"
export ADV_K_ACT_MAX="1"

echo "=== Starting Training: $RUN_NAME ==="
echo "Adversarial Training: ObsDelayMax=$ADV_K_OBS_MAX, ActDelayMax=$ADV_K_ACT_MAX"

python -u train.py \
    --env "$ENV_ID" \
    --total-timesteps "$TOTAL_STEPS" \
    --run-name "$RUN_NAME" \
    --use-residual-mem \
    --residual-mem-type "$RESIDUAL_MEM_TYPE" \
    --alpha-start "$ALPHA_START" \
    --alpha-mid "$ALPHA_MID" \
    --alpha-final "$ALPHA_FINAL" \
    --pretrained "logs/$ENV_ID/SAC_250204_065306/best_model.zip" \
    --no-load-replay-buffer \
    --use-adversary \
    --adv-lambda 0.5 \
    --adv-w-act 2.0 \
    --adv-update-every 10 \
    --adv-stage-steps 200000 400000 \
    --adv-k-obs-max "$ADV_K_OBS_MAX" \
    --adv-k-act-max "$ADV_K_ACT_MAX" \
    --adv-lr 0.05 \
    --n-envs 8 \
    --vec-env "subproc" \
    --rm-batch-size 512 \
    --rm-train-freq 64 \
    --rm-gradient-steps 6 \
    --log_dir "$LOG_DIR" \
    --verbose 1


