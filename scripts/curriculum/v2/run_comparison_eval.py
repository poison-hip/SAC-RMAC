#!/usr/bin/env python3
import os
import subprocess
import re
import sys

# Paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGS_E2 = os.path.join(REPO_ROOT, "logs/data_e2/FetchPickAndPlace-v4")
LOGS_V1 = os.path.join(REPO_ROOT, "logs/FetchPickAndPlace-v4")
PRETRAINED = os.path.join(LOGS_V1, "SAC_250204_065306/best_model.zip")

# Scenarios
# Format: Name, DisturbanceArgs, OursDirName, BaselineZipPath (None=Pretrained)
SCENARIOS = [
    {
        "name": "Obs Delay 3",
        "dist_args": ["--obs-delay-k", "3", "--action-delay-k", "0"],
        "ours_dir": "SAC_ours_curriculum_obs_delay_3_260125_050157",
        "baseline_zip": os.path.join(LOGS_V1, "baseline_obs_delay_3/best_model.zip")
    },
    {
        "name": "Action Delay 1",
        "dist_args": ["--action-delay-k", "1", "--obs-delay-k", "0"],
        "ours_dir": "SAC_ours_curriculum_act_delay_1_260125_050253",
        "baseline_zip": os.path.join(LOGS_V1, "baseline_act_delay_1_260122_192335/best_model.zip")
    },
    {
        "name": "Bias 0.05",
        "dist_args": ["--obs-bias-b", "0.05"],
        "ours_dir": "SAC_ours_curriculum_bias_0p05_260125_225318",
        # Fallback to Pretrained as no specific baseline found
        "baseline_zip": PRETRAINED 
    },
    {
        "name": "Dropout Med",
        "dist_args": ["--obs-dropout-level", "med"],
        "ours_dir": "SAC_ours_curriculum_dropout_med_260125_225301",
        "baseline_zip": os.path.join(LOGS_V1, "baseline_dropout_med/best_model.zip")
    },
    {
        "name": "Mixed (No Act)",
        "dist_args": ["--obs-delay-k", "3", "--obs-bias-b", "0.05", "--obs-dropout-level", "hard", "--action-delay-k", "0"],
        "ours_dir": "SAC_ours_curriculum_mixed_no_act_260125_065648",
        # Use industrial all on baseline
        "baseline_zip": os.path.join(LOGS_V1, "SAC_baseline_industrial_all_on_260124_174529/best_model.zip")
    }
]

EPISODES = 1000

def parse_success_rate(stdout_str):
    # Look for "Success rate: 85.0%" or similar
    match = re.search(r"Success rate:\\s*([\\d\\.]+)%", stdout_str)
    if match:
        return float(match.group(1))
    # Fallback for eval_residual_mem output: "[eval] success_rate=0.8500"
    match = re.search(r"success_rate=([\\d\\.]+)", stdout_str)
    if match:
        return float(match.group(1)) * 100.0
    return 0.0

def run_cmd(cmd_list):
    print(" ".join(cmd_list))
    try:
        # Run from REPO_ROOT
        res = subprocess.run(cmd_list, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        return res.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(e.stdout)
        print(e.stderr)
        return ""

def main():
    print(f"=== Starting Comparative Evaluation ({EPISODES} episodes each) ===")
    print(f"REPO_ROOT: {REPO_ROOT}")
    
    results = []

    for sc in SCENARIOS:
        print(f"\\n>>> Scenario: {sc['name']}")
        
        # 1. Evaluate Ours
        ours_path = os.path.join(LOGS_E2, sc[\"ours_dir\"])
        cmd_ours = [
            \"python\", \"scripts/eval/eval_residual_mem.py\",
            \"--run-dir\", ours_path,
            \"--episodes\", str(EPISODES),
            \"--no-render\" # Assuming we want speed, user didn't ask for video
        ] + sc[\"dist_args\"]
        
        print(f\"  Evaluating Ours ({os.path.basename(ours_path)})...\")
        out_ours = run_cmd(cmd_ours)
        sr_ours = parse_success_rate(out_ours)
        print(f\"  -> Ours Result: {sr_ours:.2f}%\")

        # 2. Evaluate Baseline
        base_path = sc[\"baseline_zip\"]
        cmd_base = [
            \"python\", \"scripts/eval/eval.py\",
            \"--model-path\", base_path,
            \"--episodes\", str(EPISODES),
            \"--no-render\"
        ] + sc[\"dist_args\"]
        
        print(f\"  Evaluating Baseline ({os.path.basename(os.path.dirname(base_path) if 'best_model.zip' in base_path else base_path)})...\")
        out_base = run_cmd(cmd_base)
        sr_base = parse_success_rate(out_base)
        print(f\"  -> Baseline Result: {sr_base:.2f}%\")

        results.append({
            \"scenario\": sc[\"name\"],
            \"ours\": sr_ours,
            \"baseline\": sr_base,
            \"diff\": sr_ours - sr_base
        })

    # Summary
    print(\"\\n\\n\" + \"=\"*60)
    print(f\"{'Scenario':<20} | {'Ours':<10} | {'Baseline':<10} | {'Diff':<10}\")
    print(\"-\" * 60)
    for r in results:
        print(f\"{r['scenario']:<20} | {r['ours']:.2f}%     | {r['baseline']:.2f}%     | {r['diff']:+.2f}%\")
    print(\"=\"*60)

if __name__ == \"__main__\":
    main()


