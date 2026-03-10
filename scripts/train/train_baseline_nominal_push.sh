#!/usr/bin/env bash
# Baseline nominal training for FetchPush-v4.
set -euo pipefail

ENV_ID="${ENV_ID:-FetchPush-v4}"
SEED="${SEED:-0}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2000000}"
N_ENVS="${N_ENVS:-8}"
VEC_ENV="${VEC_ENV:-subproc}" # subproc|dummy
PRETRAINED="${PRETRAINED:-}"  # optional: path to a SB3 .zip to warm-start from

# Must match train.py's run_id env abbreviation logic.
CLEAN_ENV="${ENV_ID/Fetch/}"
CLEAN_ENV="${CLEAN_ENV/PickAndPlace/PnP}"
CLEAN_ENV="${CLEAN_ENV/-v4/}"

RUN_NAME="${RUN_NAME:-baseline_nominal_push__seed${SEED}}"
METHOD_TAG="${METHOD_TAG:-baseline_nominal_push}"
OUT_ROOT="${OUT_ROOT:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -z "${OUT_ROOT}" ]]; then
  OUT_ROOT="${REPO_ROOT}/runs_push"
fi

mkdir -p "${OUT_ROOT}"

echo "[run] repo_root=${REPO_ROOT}"
echo "[run] env=${ENV_ID} seed=${SEED} total_timesteps=${TOTAL_TIMESTEPS}"
echo "[run] n_envs=${N_ENVS} vec_env=${VEC_ENV}"
echo "[run] out_root=${OUT_ROOT}"
echo "[run] run_name=${RUN_NAME}"
if [[ -n "${PRETRAINED}" ]]; then
  echo "[run] pretrained=${PRETRAINED}"
fi
echo

cd "${REPO_ROOT}"

PRETRAIN_ARGS=()
if [[ -n "${PRETRAINED}" ]]; then
  PRETRAIN_ARGS=(--pretrained "${PRETRAINED}" --no-load-replay-buffer)
fi

python -u train.py \
  --env "${ENV_ID}" \
  --seed "${SEED}" \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --run-name "${RUN_NAME}" \
  --out-root "${OUT_ROOT}" \
  --method-tag "${METHOD_TAG}" \
  --n-envs "${N_ENVS}" \
  --vec-env "${VEC_ENV}" \
  --eval-freq-paper 100000 \
  --n-eval-paper 5 \
  "${PRETRAIN_ARGS[@]}" \
  --obs-dropout-level none \
  --action-delay-k 0 \
  --obs-delay-k 0 \
  --obs-bias-b 0.0 \
  --history-k 0

