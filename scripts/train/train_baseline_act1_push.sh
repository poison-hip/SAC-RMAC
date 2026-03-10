#!/usr/bin/env bash
# Baseline Push fine-tuning under Action Delay (k=1).
set -euo pipefail

ENV_ID="${ENV_ID:-FetchPush-v4}"
SEED="${SEED:-0}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
N_ENVS="${N_ENVS:-8}"
VEC_ENV="${VEC_ENV:-subproc}" # subproc|dummy

# Must match train.py's run_id env abbreviation logic.
CLEAN_ENV="${ENV_ID/Fetch/}"
CLEAN_ENV="${CLEAN_ENV/PickAndPlace/PnP}"
CLEAN_ENV="${CLEAN_ENV/-v4/}"

RUN_NAME="${RUN_NAME:-baseline_act1_push__seed${SEED}}"
METHOD_TAG="${METHOD_TAG:-baseline_act1_push}"
OUT_ROOT="${OUT_ROOT:-}"
BASELINE_OUT_ROOT="${BASELINE_OUT_ROOT:-}"
PRETRAINED="${PRETRAINED:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -z "${OUT_ROOT}" ]]; then
  OUT_ROOT="${REPO_ROOT}/runs_push"
fi
if [[ -z "${BASELINE_OUT_ROOT}" ]]; then
  BASELINE_OUT_ROOT="${REPO_ROOT}/runs_push"
fi

mkdir -p "${OUT_ROOT}"

if [[ -z "${PRETRAINED}" ]]; then
  PATTERN="${BASELINE_OUT_ROOT}/baseline_nominal_push__${CLEAN_ENV}__seed${SEED}__*"
  LATEST_RUN="$(ls -dt ${PATTERN} 2>/dev/null | head -n 1 || true)"
  if [[ -n "${LATEST_RUN}" && -f "${LATEST_RUN}/checkpoints/last_model.zip" ]]; then
    PRETRAINED="${LATEST_RUN}/checkpoints/last_model.zip"
  fi
fi

if [[ -z "${PRETRAINED}" || ! -f "${PRETRAINED}" ]]; then
  echo "[error] Could not resolve PRETRAINED baseline Push model."
  echo "[hint] Set PRETRAINED=/abs/path/to/last_model.zip"
  exit 1
fi

echo "[run] repo_root=${REPO_ROOT}"
echo "[run] env=${ENV_ID} seed=${SEED} total_timesteps=${TOTAL_TIMESTEPS}"
echo "[run] n_envs=${N_ENVS} vec_env=${VEC_ENV}"
echo "[run] out_root=${OUT_ROOT}"
echo "[run] run_name=${RUN_NAME}"
echo "[run] pretrained=${PRETRAINED}"
echo

cd "${REPO_ROOT}"

python -u train.py \
  --env "${ENV_ID}" \
  --seed "${SEED}" \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --run-name "${RUN_NAME}" \
  --out-root "${OUT_ROOT}" \
  --method-tag "${METHOD_TAG}" \
  --n-envs "${N_ENVS}" \
  --vec-env "${VEC_ENV}" \
  --pretrained "${PRETRAINED}" \
  --no-load-replay-buffer \
  --eval-freq-paper 100000 \
  --n-eval-paper 5 \
  --obs-dropout-level none \
  --action-delay-k 1 \
  --obs-delay-k 0 \
  --obs-bias-b 0.0 \
  --history-k 0


