import os

# Experiment Matrix
SEEDS = [0, 1, 2, 3, 4]
METHODS = {
    "baseline_direct": {
        "args": "--use-residual-mem --residual-mem-type gru --alpha-start 1.0 --alpha-final 1.0", # Adjust if baseline needs simple SAC or residual
        # Note: Baseline Direct usually means training directly on the task without curriculum/adversary?
        # Or standard SAC? If "baseline_direct" means simple SAC on nominal, or simple SAC on max disturbance?
        # Usually baseline for robustness is "Train on Max Disturbance" or "Train on Nominal".
        # Let's assume "Train on Max Disturbance" (Static 3/1) to match "ObsOnly3" capability?
        # Or maybe User implies "Nominal" as baseline? 
        # Often "baseline_direct" = Domain Randomization (DR) or training on target distribution.
        # Let's assume DR or fixed max disturbance.
        # "scenario_tag" implies "train_dist".
        # Let's define it as "Train on Nominal" for now, or user can edit.
        "desc": "Standard Training (Nominal)"
    },
    "ours_advB": {
        "args": "--use-residual-mem --residual-mem-type gru --use-adversary --adv-lambda 0.5 --adv-stage-steps 200000 400000",
        "desc": "Ours with Budgeted Adversary (Version B)"
    }
}

COMMON_ARGS = (
    "--env FetchPickAndPlace-v4 "
    "--total-timesteps 2000000 "
    "--n-envs 16 "
    "--vec-env subproc "
    "--eval-freq 100000 " # SB3 eval
    "--eval-freq-paper 100000 " # Paper fixed eval
    "--n-eval-paper 50 "
    "--rm-batch-size 512 "
    "--rm-train-freq 64 "
    "--rm-gradient-steps 6 "
    "--force-wrappers " # Ensure wrappers exist? (Handled by logic in train.py usually)
)

OUTPUT_FILE = "run_experiments.sh"

def main():
    with open(OUTPUT_FILE, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Generated Experiment Matrix\n\n")
        f.write("LOG_ROOT=\"/home/poison/桌面/SAC/Mastering-RL/logs/final\"\n\n")
        
        for method_name, config in METHODS.items():
            for seed in SEEDS:
                run_id = f"{method_name}__seed{seed}"
                cmd = (
                    f"python -u train.py "
                    f"{COMMON_ARGS} "
                    f"--seed {seed} "
                    f"--run-name {run_id} "
                    f"--log_dir $LOG_ROOT "
                    f"{config['args']}"
                )
                f.write(f"echo 'Starting {run_id}'\n")
                f.write(f"{cmd}\n\n")
                
    print(f"Generated {OUTPUT_FILE}")

if __name__ == "__main__":
    main()


























