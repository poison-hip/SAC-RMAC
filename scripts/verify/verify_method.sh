#!/bin/bash
# Generated Verification Script

LOG_ROOT="/home/poison/桌面/SAC/Mastering-RL/logs/final"

# 1. Ours + Adversary (Restricted to Obs Delay only)
# Adversary will learn to push k_obs up to 3.
# Added --adv-stage-steps 1000 2000 to SKIP warm-up and allow k=3 immediately.
echo 'Starting Verification: ours_adv_obs3__seed0'
python -u train.py --env FetchPickAndPlace-v4 --total-timesteps 1000000 --n-envs 16 --vec-env subproc --eval-freq 100000 --eval-freq-paper 100000 --n-eval-paper 50 --rm-batch-size 512 --rm-train-freq 64 --rm-gradient-steps 6 --pretrained logs/FetchPickAndPlace-v4/SAC_250204_065306/best_model.zip --no-load-replay-buffer --seed 0 --run-name ours_adv_obs3__seed0 --log_dir $LOG_ROOT --use-residual-mem --residual-mem-type gru --alpha-start 0.0 --alpha-final 1.0 --use-adversary --adv-k-obs-max 3 --adv-k-act-max 0 --adv-stage-steps 1000 2000

# 2. Ours + Adversary (Restricted to Act Delay only)
# Adversary will learn to push k_act up to 1.
echo 'Starting Verification: ours_adv_act1__seed0'
python -u train.py --env FetchPickAndPlace-v4 --total-timesteps 1000000 --n-envs 16 --vec-env subproc --eval-freq 100000 --eval-freq-paper 100000 --n-eval-paper 50 --rm-batch-size 512 --rm-train-freq 64 --rm-gradient-steps 6 --pretrained logs/FetchPickAndPlace-v4/SAC_250204_065306/best_model.zip --no-load-replay-buffer --seed 0 --run-name ours_adv_act1__seed0 --log_dir $LOG_ROOT --use-residual-mem --residual-mem-type gru --alpha-start 0.0 --alpha-final 1.0 --use-adversary --adv-k-obs-max 0 --adv-k-act-max 1 --adv-stage-steps 1000 2000


























