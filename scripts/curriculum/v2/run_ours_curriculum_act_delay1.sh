#!/bin/bash
# Ours: Action Delay k=1 (Residual Memory, WITH Curriculum)

# 0. Path & Dir Setup
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# 1. Environment
export ENV_NAME="mastering-rl"
export ENV_ID="FetchPickAndPlace-v4"
export LOG_DIR="logs/data_e2"

# 2. Training Config
export TOTAL_STEPS=1000000
export RUN_NAME="ours_curriculum_act_delay_1"

# 3. Disturbance Config (Max values for Curriculum)
# Stages: none -> easy -> med -> hard
# Hard stage will reach these values:
export OBS_DROPOUT_LEVEL="none"
export OBS_DELAY_K="0"
export ACTION_DELAY_K="1"
export OBS_BIAS_B="0.0"

# 4. Residual Config (ENABLED)
export USE_RESIDUAL_MEM=1
# Alpha schedule
export ALPHA_START="0.0"
export ALPHA_MID="0.2"
export ALPHA_FINAL="1.0"
export RESIDUAL_MEM_TYPE="gru"

echo "=== Starting Training: $RUN_NAME ==="
echo "Disturbance Curriculum: Target Action Delay k=$ACTION_DELAY_K"

python -u train.py \
    --env "$ENV_ID" \
    --total-timesteps "$TOTAL_STEPS" \
    --run-name "$RUN_NAME" \
    --obs-dropout-level "$OBS_DROPOUT_LEVEL" \
    --obs-delay-k "$OBS_DELAY_K" \
    --action-delay-k "$ACTION_DELAY_K" \
    --obs-bias-b "$OBS_BIAS_B" \
    --use-residual-mem \
    --residual-mem-type "$RESIDUAL_MEM_TYPE" \
    --alpha-start "$ALPHA_START" \
    --alpha-mid "$ALPHA_MID" \
    --alpha-final "$ALPHA_FINAL" \
    --pretrained "logs/$ENV_ID/SAC_250204_065306/best_model.zip" \
    --no-load-replay-buffer \
    --use-disturbance-curriculum \
    --n-envs 8 \
    --vec-env "subproc" \
    --rm-batch-size 512 \
    --rm-train-freq 64 \
    --rm-gradient-steps 6 \
    --rm-gradient-steps 6 \
    --log_dir "$LOG_DIR" \
    --verbose 1


