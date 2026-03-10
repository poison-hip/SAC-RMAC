
import os
import subprocess
import json
import re
import sys

# Define pairs to evaluate
pairs = [
    {"name": "ObsDelay k=2", "baseline": "baseline_obs_delay_2", "ours": "ours_obs_delay_2", "args": ["--obs-delay-k", "2", "--action-delay-k", "0", "--obs-dropout-level", "none", "--obs-bias-b", "0.0"]},
    {"name": "ObsDelay k=3", "baseline": "baseline_obs_delay_3", "ours": "ours_obs_delay_3", "args": ["--obs-delay-k", "3", "--action-delay-k", "0", "--obs-dropout-level", "none", "--obs-bias-b", "0.0"]},
    {"name": "Dropout Easy", "baseline": "baseline_dropout_easy", "ours": "ours_dropout_easy", "args": ["--obs-dropout-level", "easy", "--obs-dropout-mode", "hold-last", "--action-delay-k", "0", "--obs-delay-k", "0", "--obs-bias-b", "0.0"]},
    {"name": "Dropout Med",  "baseline": "baseline_dropout_med",  "ours": "ours_dropout_med",  "args": ["--obs-dropout-level", "med", "--obs-dropout-mode", "hold-last", "--action-delay-k", "0", "--obs-delay-k", "0", "--obs-bias-b", "0.0"]},
    {"name": "Bias b=0.01",  "baseline": "baseline_bias_0p01",    "ours": "ours_bias_0p01",    "args": ["--obs-bias-b", "0.01", "--action-delay-k", "0", "--obs-delay-k", "0", "--obs-dropout-level", "none"]},
    {"name": "Bias b=0.05",  "baseline": "baseline_bias_0p05",    "ours": "ours_bias_0p05",    "args": ["--obs-bias-b", "0.05", "--action-delay-k", "0", "--obs-delay-k", "0", "--obs-dropout-level", "none"]},
    {"name": "Industrial",   "baseline": "baseline_industrial_all_on", "ours": "ours_industrial_all_on", "args": ["--obs-dropout-level", "hard", "--obs-dropout-mode", "hold-last", "--obs-delay-k", "2", "--action-delay-k", "1", "--obs-bias-b", "0.02"]},
]

ENV_ID = "FetchPickAndPlace-v4"
LOG_DIR = f"logs/{ENV_ID}"
EPISODES = 1000

results = []

def get_latest_run(pattern):
    try:
        # Sort by modification time, newest first
        dirs = [d for d in os.listdir(LOG_DIR) if pattern in d]
        if not dirs: return None
        # Sort by full path mod time
        dirs.sort(key=lambda x: os.path.getmtime(os.path.join(LOG_DIR, x)), reverse=True)
        return dirs[0]
    except Exception:
        return None

for p in pairs:
    row = {"name": p["name"], "baseline_sr": "N/A", "ours_sr": "N/A"}
    
    # 1. EVALUATE BASELINE (Re-run locally to capture output)
    b_run = get_latest_run(p["baseline"])
    if b_run:
        b_path = os.path.join(LOG_DIR, b_run, "best_model.zip")
        if not os.path.exists(b_path):
             b_path = os.path.join(LOG_DIR, b_run, f"{ENV_ID}.zip")
        
        if os.path.exists(b_path):
            print(f"Evaluating Baseline: {p['name']} ({b_run})...", flush=True)
            cmd = ["python", "-u", "scripts/eval/eval.py", "--env", ENV_ID, "--model-path", b_path, "--episodes", str(EPISODES), "--no-render"] + p["args"]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True)
                # Parse "Success rate: 30.0%"
                m = re.search(r"Success rate:\s*([\d\.]+)%", res.stdout)
                if m:
                    row["baseline_sr"] = f"{float(m.group(1)):.1f}%"
                else:
                    row["baseline_sr"] = "Error"
            except Exception as e:
                row["baseline_sr"] = "Exc"
        else:
            row["baseline_sr"] = "No Model"
    else:
        row["baseline_sr"] = "Missing"

    # 2. READ OURS (From JSON)
    o_run = get_latest_run(p["ours"])
    if o_run:
        json_path = os.path.join(LOG_DIR, o_run, "docs", "eval_residual_mem.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    sr = data.get("success_rate", 0) * 100
                    row["ours_sr"] = f"{sr:.1f}%"
            except:
                row["ours_sr"] = "Read Err"
        else:
            row["ours_sr"] = "No JSON"
    else:
        row["ours_sr"] = "Missing"

    results.append(row)

print("\n\n=== Final Results Summary ===")
print("| Experiment | Baseline Success | Ours Success |")
print("|---|---|---|")
for r in results:
    print(f"| {r['name']} | {r['baseline_sr']} | {r['ours_sr']} |")
