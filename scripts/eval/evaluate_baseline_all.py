
import argparse
import sys
import yaml
import json
import torch as th
import numpy as np
import gymnasium as gym
import gymnasium_robotics
try:
    import panda_gym
except ImportError:
    pass
import os
from stable_baselines3 import SAC
from perturbation_wrappers import build_perturbed_env, parse_indices

# Best effort action scaling
def _unscale_action(a_scaled: th.Tensor, action_bias: th.Tensor, action_scale: th.Tensor) -> th.Tensor:
    return action_bias + action_scale * a_scaled

def _scale_action(a_env: th.Tensor, action_bias: th.Tensor, action_scale: th.Tensor) -> th.Tensor:
    denom = th.where(action_scale.abs() < 1e-8, th.ones_like(action_scale), action_scale)
    return (a_env - action_bias) / denom

# Updated Scenarios matching "evaluate_all_complete.py" exactly
PAPER_SETTINGS = [
    # Standard Delay Scenarios
    {'name': 'Nominal', 'k_obs': 0, 'k_act': 0, 'dropout_p': 0, 'dropout_level': 'none'},
    {'name': 'ObsOnly3', 'k_obs': 3, 'k_act': 0, 'dropout_p': 0, 'dropout_level': 'none'},
    {'name': 'ActOnly1', 'k_obs': 0, 'k_act': 1, 'dropout_p': 0, 'dropout_level': 'none'},
    {'name': 'MixedHard', 'k_obs': 3, 'k_act': 1, 'dropout_p': 0, 'dropout_level': 'none'},
    
    # Dropout Scenarios
    {'name': 'DropoutEasy', 'k_obs': 0, 'k_act': 0, 'dropout_p': 0.3, 'dropout_level': 'easy'},
    {'name': 'DropoutMed', 'k_obs': 0, 'k_act': 0, 'dropout_p': 0.5, 'dropout_level': 'medium'},
    {'name': 'DropoutHard', 'k_obs': 0, 'k_act': 0, 'dropout_p': 0.5, 'dropout_level': 'hard'},
    
    # Combined Delay + Dropout
    {'name': 'Delay+DropoutEasy', 'k_obs': 3, 'k_act': 1, 'dropout_p': 0.3, 'dropout_level': 'easy'},
    {'name': 'Delay+DropoutMed', 'k_obs': 3, 'k_act': 1, 'dropout_p': 0.5, 'dropout_level': 'medium'},
    # Replicating the "anomaly" from previous scripts where Hard didn't have delays
    {'name': 'Delay+DropoutHard', 'k_obs': 0, 'k_act': 0, 'dropout_p': 0.5, 'dropout_level': 'hard'}, 
]

def eval_baseline_dir(run_dir, seed_override=None, n_episodes=1000):
    config_path = f"{run_dir}/config.json"
    
    # Check if files exist
    if not os.path.exists(config_path):
        print(f"[Warn] Missing config in {run_dir}, skipping.")
        return None

    with open(config_path, 'r') as f:
        config = json.load(f)

    env_id = config.get("env", "PandaReachDense-v3")
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # 1. Base SAC
    base_model_path = f"{run_dir}/best_model.zip"
    if not os.path.exists(base_model_path):
         # Try creating alias if different naming?
         # User said "best_model.zip" in chat, but script might save "best_model.zip"
         # But maybe `baseline_pretrain__.../best_model.zip`
         # Let's check for `model.zip` or `best_model.zip`
         if os.path.exists(f"{run_dir}/model.zip"):
             base_model_path = f"{run_dir}/model.zip"
         else:
             print(f"[Warn] Missing best_model.zip in {run_dir}, checking subdirs...")
             # Sometimes it's inside `checkpoints/`? But list_dir showed it at root
             print(f"[Warn] Still missing, skipping.")
             return None

    dummy_env = gym.make(env_id)
    base_model = SAC.load(base_model_path, env=dummy_env, device=device)
    base_policy = base_model.policy
    base_policy.set_training_mode(False)
    
    dummy_env.close()

    # Evaluation Loop per setting
    results = {}
    
    # Determine seed
    if seed_override is not None:
        seed = int(seed_override)
    else:
        seed = int(config.get("seed", 42))

    for scenario in PAPER_SETTINGS:
        setting_name = scenario["name"]
        k_obs = scenario["k_obs"]
        k_act = scenario["k_act"]
        dropout_p = scenario["dropout_p"]
        dropout_level = scenario["dropout_level"]
        
        env = gym.make(env_id)
        
        env = build_perturbed_env(
            env,
            dropout_p=dropout_p,
            dropout_level=dropout_level,
            action_delay_k=k_act,
            obs_delay_k=k_obs,
            bias_b=0.0,   
            force_wrappers=True
        )
        
        successes = []
        rewards = []
        
        for i in range(n_episodes):
            obs, info = env.reset(seed=seed + 2000 + i) # Align with ours eval seeds
            done = False
            truncated = False
            ep_reward = 0
            
            while not (done or truncated):
                # Standard SB3 predict handles numpy dicts automatically
                action, _ = base_model.predict(obs, deterministic=True)
                
                obs, r, done, truncated, info = env.step(action)
                ep_reward += r
            
            success = float(info.get("is_success", 0.0))
            successes.append(success)
            rewards.append(ep_reward)
        
        env.close()
        sr = np.mean(successes)
        mr = np.mean(rewards)
        results[setting_name] = {"sr": sr, "mr": mr}

    return results

def eval_all_baseline(args):
    root_dir = args.root_dir
    n_episodes = args.n_episodes
    
    gym.register_envs(gymnasium_robotics)
    
    # Find subdirs
    subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    subdirs.sort()
    
    all_results = []
    
    print(f"Found {len(subdirs)} runs in {root_dir}")
    print(f"Evaluating {n_episodes} episodes per scenario for each run.")
    
    for d in subdirs:
        # Filter for baseline runs only if needed? User pointed to `runs/baseline` so assume all are valid.
        try:
            seed_str = d.split("seed")[-1].split("__")[0]
        except:
            seed_str = "unknown"
            
        print(f"--- Evaluating Seed {seed_str} at {os.path.basename(d)} ---")
        res = eval_baseline_dir(d, n_episodes=n_episodes)
        if res:
            res["seed"] = seed_str
            all_results.append(res)
            # Print brief summary row
            print(f"Seed {seed_str} | Nominal: {res['Nominal']['sr']*100:.1f}% | Obs3: {res['ObsOnly3']['sr']*100:.1f}% | MixedHard: {res['MixedHard']['sr']*100:.1f}%")
        else:
            print(f"[Warn] Failed/Skipped run {d}")
    
    # Compute and Print Final Table
    if not all_results:
        print("No results collected.")
        return

    n = len(all_results)
    avg_stats = {}
    
    scenario_names = [s['name'] for s in PAPER_SETTINGS]
    
    for name in scenario_names:
        srs = [r[name]["sr"] for r in all_results]
        mrs = [r[name]["mr"] for r in all_results]
        avg_stats[name] = {
            "sr_mean": np.mean(srs),
            "sr_std": np.std(srs),
            "mr_mean": np.mean(mrs),
        }

    print("\n==========================================================================================")
    print(f"FINAL AGGREGATED BASELINE RESULTS (N={n} Seeds)")
    print("==========================================================================================")
    print(f"{'Scenario':<20} | {'SR Mean':<8} | {'SR Std':<8} | {'Return':<8}")
    print("-" * 80)
    for name in scenario_names:
        stats = avg_stats[name]
        print(f"{name:<20} | {stats['sr_mean']*100:5.1f}%  | {stats['sr_std']*100:5.1f}%  | {stats['mr_mean']:6.2f}")
    print("==========================================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, required=True, help="Path to directory containing baseline runs")
    parser.add_argument("--n-episodes", type=int, default=1000)
    args = parser.parse_args()
    
    eval_all_baseline(args)
