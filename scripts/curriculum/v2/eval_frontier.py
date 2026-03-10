import argparse
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC, DDPG, TD3
import torch as th

# Add parent directory to path to import modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from perturbation_wrappers import build_perturbed_env
from residual_mem_trainer import train_pretrained_sac_with_residual_mem # for Reference if needed
from residual_mem_modules import ResidualMemPolicy # Needed for loading if using custom objects?
# Note: SB3 loads custom policies if class is available in namespace or passed in custom_objects

def parse_args():
    parser = argparse.ArgumentParser("Eval Frontier")
    parser.add_argument("--model-path", type=str, required=True, help="Path to best_model.zip or similar")
    parser.add_argument("--env-id", type=str, default="FetchPickAndPlace-v4")
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--algo", type=str, default="SAC", choices=["SAC", "DDPG", "TD3"])
    # Residual args to reconstruct environment correctly if needed
    parser.add_argument("--use-residual-mem", action="store_true")
    # We assume 'gru' if not specified, but let's allow it
    parser.add_argument("--residual-mem-type", type=str, default="gru")
    # For frontier, we usually sweep k_obs and k_act.
    # We might need other disturbance params to remain "nominal" or match training? 
    # Usually frontier implies "robustness to ONE type of disturbance varying".
    return parser.parse_args()

def evaluate_point(model, env, k_obs, k_act, n_episodes):
    # Set disturbance
    # We can use env.set_disturbance via wrapper
    # We need to verify if the env structure supports it.
    
    # Check if we have the DisturbanceManagerWrapper
    if hasattr(env, "set_disturbance"):
         env.set_disturbance(k_obs=k_obs, k_act=k_act)
    else:
        # Try finding it in wrapper hierarchy
        # (Usually SB3 VecEnv hides it, but here we might use a single Gym env for simplicity)
        pass

    successes = []
    returns = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_ret = 0
        state = None # For recurrent policies
        
        while not (done or truncated):
            # Deterministic for eval
            action, state = model.predict(obs, state=state, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_ret += reward
            
        successes.append(float(info.get("is_success", 0.0)))
        returns.append(ep_ret)
        
    return np.mean(successes), np.mean(returns), np.mean(len(successes))

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Environment
    gym.register_envs(gymnasium_robotics)
    env = gym.make(args.env_id)
    
    # Wrap with our perturbation wrapper (enabled for control)
    # We MUST enable disturbance control to allow setting k_obs/k_act
    env = build_perturbed_env(
        env,
        dropout_level="none", # default
        force_wrappers=True,
        enable_disturbance_control=True
    )
    
    # 2. Load Model
    print(f"Loading model from {args.model_path}...")
    custom_objects = {}
    if args.use_residual_mem:
        # Assuming ResidualMemPolicy is needed; ensure it is importable.
        pass

    model_cls = {"SAC": SAC, "TD3": TD3, "DDPG": DDPG}[args.algo]
    model = model_cls.load(args.model_path, env=env, custom_objects=custom_objects)
    
    # 3. Define Frontier Grid
    k_obs_vals = [0, 1, 2, 3]
    k_act_vals = [0, 1, 2]
    
    results = []
    
    print("\nRunning Frontier Scan...")
    
    row_count = 0
    total_rows = len(k_obs_vals) * len(k_act_vals)
    
    for k_obs in k_obs_vals:
        for k_act in k_act_vals:
            row_count += 1
            print(f"[{row_count}/{total_rows}] Evaluating k_obs={k_obs}, k_act={k_act} ... ", end="")
            
            # Update Envs
            if hasattr(env, "set_disturbance"):
                env.set_disturbance(k_obs=k_obs, k_act=k_act)
            
            sr, ret, _ = evaluate_point(model, env, k_obs, k_act, args.n_eval)
            print(f"SR={sr:.2f}, Ret={ret:.1f}")
            
            results.append({
                "k_obs": k_obs,
                "k_act": k_act,
                "success_rate": sr,
                "return_mean": ret
            })
            
    env.close()
    
    # 4. Save CSV
    csv_path = os.path.join(args.output_dir, "frontier.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k_obs", "k_act", "success_rate", "return_mean"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Saved results to {csv_path}")
    
    # 5. Plotting
    matrix = np.zeros((len(k_obs_vals), len(k_act_vals)))
    for r in results:
        i = k_obs_vals.index(r["k_obs"])
        j = k_act_vals.index(r["k_act"])
        matrix[i, j] = r["success_rate"]
        
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix.T, origin='lower', cmap='viridis', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(label='Success Rate')
    plt.xlabel('Observation Delay (k_obs)')
    plt.ylabel('Action Delay (k_act)')
    plt.xticks(range(len(k_obs_vals)), k_obs_vals)
    plt.yticks(range(len(k_act_vals)), k_act_vals)
    plt.title(f'Robustness Heatmap: {args.algo}')
    plt.savefig(os.path.join(args.output_dir, "heatmap.png"))
    plt.close()
    
    # 1D Curve: SR vs k_obs (at fixed k_act=1)
    if 1 in k_act_vals:
        subset = [r for r in results if r["k_act"] == 1]
        subset.sort(key=lambda x: x["k_obs"])
        x = [r["k_obs"] for r in subset]
        y = [r["success_rate"] for r in subset]
        
        plt.figure()
        plt.plot(x, y, marker='o')
        plt.ylim(-0.05, 1.05)
        plt.xlabel("k_obs")
        plt.ylabel("Success Rate")
        plt.title("Robustness to Obs Delay (Fixed k_act=1)")
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "frontier_kobs.png"))
        plt.close()

if __name__ == "__main__":
    main()


























