#!/bin/bash
# Compare Models: Baseline vs Ours (Nominal) vs Ours (Adversary)

# Ensure correct python environment
PYTHON="conda run -n mastering-rl --no-capture-output python"

LOG_ROOT="/home/poison/桌面/SAC/Mastering-RL/logs/final/FetchPickAndPlace-v4"

# List of all valid models
MODELS=(
    # "SAC_baseline_act_obs__seed0_260126_235554"
    # "SAC_baseline_act1__seed0_260127_180143"
    "baseline_obs_delay_3_260124_003824"
    "baseline_obs_delay_3"
    "SAC_ours_nominal__seed0_260127_025358"
    "SAC_ours_adv_act1__seed0_260126_051920"
    "SAC_ours_adv_obs3__seed0_260126_051520"
    "SAC_ours_adv_all__seed0_260126_195807"
)

# Join models with comma
MODEL_LIST=$(IFS=,; echo "${MODELS[*]}")

echo "=========================================="
echo "Comparing ${#MODELS[@]} Models using scripts/eval/evaluate_residual.py"
echo "Conditions: Nominal, Obs(3), Act(1), Mixed"
echo "=========================================="

# Run unified evaluation
$PYTHON -u scripts/eval/evaluate_residual.py \
    --log-root "$LOG_ROOT" \
    --dirs "$MODEL_LIST" \
    "$@"


