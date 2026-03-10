#!/usr/bin/env bash
# Evaluate baseline Push model under 4 fixed delay scenarios.
set -euo pipefail

ENV_ID="${ENV_ID:-FetchPush-v4}"
SEED="${SEED:-0}"
EPISODES="${EPISODES:-200}"
RUN_DIR="${RUN_DIR:-}"
MODEL_PATH="${MODEL_PATH:-}"
BASELINE_OUT_ROOT="${BASELINE_OUT_ROOT:-}"

# Must match train.py's run_id env abbreviation logic.
CLEAN_ENV="${ENV_ID/Fetch/}"
CLEAN_ENV="${CLEAN_ENV/PickAndPlace/PnP}"
CLEAN_ENV="${CLEAN_ENV/-v4/}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${BASELINE_OUT_ROOT}" ]]; then
  BASELINE_OUT_ROOT="${REPO_ROOT}/runs_push"
fi

if [[ -z "${RUN_DIR}" ]]; then
  PATTERN="${BASELINE_OUT_ROOT}/baseline_nominal_push__${CLEAN_ENV}__seed${SEED}__*"
  RUN_DIR="$(ls -dt ${PATTERN} 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "[error] Could not resolve baseline Push run dir."
  echo "[hint] Set RUN_DIR=/abs/path/to/run_dir"
  exit 1
fi

if [[ -z "${MODEL_PATH}" ]]; then
  MODEL_PATH="${RUN_DIR}/checkpoints/last_model.zip"
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "[error] MODEL_PATH not found: ${MODEL_PATH}"
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

SCENARIO_CSV="${TMP_DIR}/scenario_metrics.csv"
echo "scenario,k_act,k_obs,success_rate_percent,success_rate" > "${SCENARIO_CSV}"

run_one() {
  local name="$1"
  local k_act="$2"
  local k_obs="$3"
  local log_file="${TMP_DIR}/${name}.log"

  echo "[eval] ${name} (k_act=${k_act}, k_obs=${k_obs})"
  python -u scripts/eval/eval.py \
    --model SAC \
    --env "${ENV_ID}" \
    --episodes "${EPISODES}" \
    --seed "${SEED}" \
    --no-render \
    --model-path "${MODEL_PATH}" \
    --action-delay-k "${k_act}" \
    --obs-delay-k "${k_obs}" \
    > "${log_file}" 2>&1

  local sr_pct
  sr_pct="$(grep -E "Success rate:" "${log_file}" | tail -n 1 | sed -E 's/.*Success rate:[[:space:]]*([0-9.]+)%.*/\1/')"
  if [[ -z "${sr_pct}" ]]; then
    echo "[error] failed to parse success rate from ${log_file}"
    cat "${log_file}"
    exit 1
  fi
  local sr
  sr="$(python - <<PY
v = float("${sr_pct}") / 100.0
print(f"{v:.6f}")
PY
)"
  echo "${name},${k_act},${k_obs},${sr_pct},${sr}" >> "${SCENARIO_CSV}"
}

run_one "Nominal" 0 0
run_one "ObsOnly3" 0 3
run_one "ActOnly1" 1 0
run_one "MixedHard" 1 3

OUT_DIR="${RUN_DIR}/docs"
mkdir -p "${OUT_DIR}"
OUT_JSON="${OUT_DIR}/eval_4scenarios_ep${EPISODES}.json"

python - <<PY
import csv
import json

rows = []
with open("${SCENARIO_CSV}", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

out = {
    "method": "baseline",
    "env": "${ENV_ID}",
    "seed": int("${SEED}"),
    "episodes_per_scenario": int("${EPISODES}"),
    "run_dir": "${RUN_DIR}",
    "model_path": "${MODEL_PATH}",
    "scenarios": {},
}

for r in rows:
    out["scenarios"][r["scenario"]] = {
        "k_act": int(r["k_act"]),
        "k_obs": int(r["k_obs"]),
        "success_rate": float(r["success_rate"]),
        "success_rate_percent": float(r["success_rate_percent"]),
    }

with open("${OUT_JSON}", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY

echo "[done] wrote ${OUT_JSON}"

