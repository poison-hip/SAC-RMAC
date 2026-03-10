#!/usr/bin/env bash
# Baseline (Nominal) training script
# - "原版"：标准 SAC 训练，不启用 residual-mem / adversary / disturbance curriculum
# - 适合你开多个终端：每个终端改一下 SEED/RUN_NAME 直接跑
set -euo pipefail

# --------- user config ----------
ENV_ID="${ENV_ID:-FetchPickAndPlace-v4}"
SEED="${SEED:-0}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-5000000}"

# throughput (adjust to your machine)
N_ENVS="${N_ENVS:-8}"
VEC_ENV="${VEC_ENV:-subproc}"   # subproc|dummy

# output naming
RUN_NAME="${RUN_NAME:-baseline_nominal__seed${SEED}}"
OUT_ROOT="${OUT_ROOT:-}"        # default: <repo>/runs_baseline_nominal

# If you prefer forcing CPU, uncomment:
# export CUDA_VISIBLE_DEVICES=""
# --------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"   # .../Mastering-RL

if [[ -z "${OUT_ROOT}" ]]; then
  OUT_ROOT="${REPO_ROOT}/runs_baseline_nominal"
fi

echo "[run] repo_root=${REPO_ROOT}"
echo "[run] env=${ENV_ID} seed=${SEED} total_timesteps=${TOTAL_TIMESTEPS}"
echo "[run] n_envs=${N_ENVS} vec_env=${VEC_ENV}"
echo "[run] out_root=${OUT_ROOT}"
echo "[run] run_name=${RUN_NAME}"
echo

cd "${REPO_ROOT}"

# Note:
# - 不传 --use-residual-mem / --use-adversary / --use-disturbance-curriculum => 都是关闭
# - 显式把各种扰动参数设为 none/0，确保是纯 nominal
python -u train.py \
  --env "${ENV_ID}" \
  --seed "${SEED}" \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --run-name "${RUN_NAME}" \
  --out-root "${OUT_ROOT}" \
  --method-tag "baseline_nominal" \
  --n-envs "${N_ENVS}" \
  --vec-env "${VEC_ENV}" \
  --eval-freq-paper 100000 \
  --n-eval-paper 5 \
  --obs-dropout-level none \
  --action-delay-k 0 \
  --obs-delay-k 0 \
  --obs-bias-b 0.0 \
  --history-k 0


