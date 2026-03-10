#!/bin/bash
# Baseline (Standard SAC) + Combined Delays (Fixed: Obs=3, Act=1)
# This script trains a standard SAC agent (fine-tuning) on the hard combined delay setting.

# Ensure this script works no matter where you run it from
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# 0. Path Config
LOG_ROOT="/home/poison/桌面/SAC/Mastering-RL/logs/final"

echo 'Starting Training: baseline_act_obs__seed0'
python -u train.py \
    --env FetchPickAndPlace-v4 \
    --total-timesteps 5000000 \
    --n-envs 16 \
    --vec-env subproc \
    --eval-freq 100000 \
    --eval-freq-paper 100000 \
    --n-eval-paper 50 \
    --pretrained logs/FetchPickAndPlace-v4/SAC_250204_065306/best_model.zip \
    --no-load-replay-buffer \
    --seed 0 \
    --run-name baseline_act_obs__seed0 \
    --log_dir $LOG_ROOT \
    --obs-delay-k 3 \
    --action-delay-k 1


