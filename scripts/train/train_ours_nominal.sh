#!/bin/bash
# Ours (Residual Memory) - Nominal (No Disturbances)
# This script trains the residual agent on the clean environment to verify it doesn't degrade performance.

# Ensure this script works no matter where you run it from
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# 0. Path Config
LOG_ROOT="/home/poison/桌面/SAC/Mastering-RL/logs/final"

echo 'Starting Training: ours_nominal__seed0'
python -u train.py \
    --env FetchPickAndPlace-v4 \
    --total-timesteps 1000000 \
    --n-envs 16 \
    --vec-env subproc \
    --eval-freq 100000 \
    --eval-freq-paper 100000 \
    --n-eval-paper 50 \
    --rm-batch-size 512 \
    --rm-train-freq 64 \
    --rm-gradient-steps 6 \
    --pretrained logs/FetchPickAndPlace-v4/SAC_250204_065306/best_model.zip \
    --no-load-replay-buffer \
    --seed 0 \
    --run-name ours_nominal__seed0 \
    --log_dir $LOG_ROOT \
    --use-residual-mem \
    --residual-mem-type gru \
    --alpha-start 0.0 \
    --alpha-final 1.0


