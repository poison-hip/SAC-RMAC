import os

# "Verified" Hard Cases
SEEDS = [0] # Minimal seed for verification
METHODS = {
    # 1. Ours trained specifically on Fixed Obs Delay = 3
    "ours_fixed_obs3": {
        "args": "--use-residual-mem --residual-mem-type gru --alpha-start 1.0 --alpha-final 1.0 --obs-delay-k 3 --action-delay-k 0",
        "desc": "Ours (Residual) on Fixed Obs Delay 3"
    },
    # 2. Ours trained specifically on Fixed Act Delay = 1
    "ours_fixed_act1": {
        "args": "--use-residual-mem --residual-mem-type gru --alpha-start 1.0 --alpha-final 1.0 --obs-delay-k 0 --action-delay-k 1",
        "desc": "Ours (Residual) on Fixed Act Delay 1"
    }
}

COMMON_ARGS = (
    "--env FetchPickAndPlace-v4 "
    "--total-timesteps 1000000 "
    "--n-envs 16 "
    "--vec-env subproc "
    "--eval-freq 100000 "
    "--eval-freq-paper 100000 "
    "--n-eval-paper 50 "
    "--rm-batch-size 512 "
    "--rm-train-freq 64 "
    "--rm-gradient-steps 6 "
    "--pretrained logs/FetchPickAndPlace-v4/SAC_250204_065306/best_model.zip "
    "--no-load-replay-buffer "
)

OUTPUT_FILE = "verify_method.sh"

def main():
    with open(OUTPUT_FILE, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Generated Verification Script\n\n")
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
                f.write(f"echo 'Starting Verification: {run_id}'\n")
                f.write(f"{cmd}\n\n")
                
    print(f"Generated {OUTPUT_FILE}")

if __name__ == "__main__":
    main()


























