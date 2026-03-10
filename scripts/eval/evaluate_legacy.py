
from __future__ import annotations

# Make repo root importable when running as: python scripts/eval/evaluate_legacy.py
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import os
import argparse
import numpy as np
import torch as th
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
from perturbation_wrappers import build_perturbed_env, parse_indices
from residual_mem_modules import FourierFeatures, ResidualMemPolicy

# Register envs
gym.register_envs(gymnasium_robotics)

def evaluate_residual_policy(env, base_model, residual_policy, device, obj_idx, fourier, n_episodes=20):
    successes = []
    # Action scaling
    low = th.as_tensor(env.action_space.low, device=device).float()
    high = th.as_tensor(env.action_space.high, device=device).float()
    action_scale = 0.5 * (high - low)
    action_bias = 0.5 * (high + low)
    
    def _unscale_action(a_scaled):
        return action_bias + action_scale * a_scaled

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        
        # Init recurrent state
        h = residual_policy.init_hidden(1, device=device)
        prev_action = np.zeros(env.action_space.shape[0], dtype=np.float32)
        
        while not (done or truncated):
            # 1. Base Policy
            obs_th = {k: th.as_tensor(v, device=device).float().unsqueeze(0) for k, v in obs.items()}
            with th.no_grad():
                a0_scaled = base_model.policy.actor(obs_th, deterministic=True)
                a0_env = _unscale_action(a0_scaled).cpu().numpy()[0]
            
            # 2. Residual Input
            o_obs = obs["observation"]
            o_ag = obs.get("achieved_goal", np.zeros(0))
            o_dg = obs.get("desired_goal", np.zeros(0))
            
            # Fourier features for object part
            obj_dim = len(obj_idx)
            keep_idx = [i for i in range(o_obs.shape[0]) if i not in obj_idx]
            
            obj_part = o_obs[obj_idx]
            rest_part = o_obs[keep_idx]
            
            obj_t = th.as_tensor(obj_part, device=device).float().unsqueeze(0) # (1, obj_dim)
            obj_ff = fourier(obj_t)
            
            rest_t = th.as_tensor(rest_part, device=device).float().unsqueeze(0)
            ag_t = th.as_tensor(o_ag, device=device).float().unsqueeze(0)
            dg_t = th.as_tensor(o_dg, device=device).float().unsqueeze(0)
            ap_t = th.as_tensor(prev_action, device=device).float().unsqueeze(0)
            m_t = th.ones((1, obj_dim), device=device).float()
            
            x = th.cat([obj_ff, rest_t, ag_t, dg_t, ap_t, m_t], dim=-1).unsqueeze(1) # (1,1,input)
            
            with th.no_grad():
                delta, _, h_out = residual_policy.forward_sequence(x, h, deterministic=True)
            h = h_out
            
            delta_val = delta.cpu().numpy()[0, 0] # (act_dim)
            delta_env = delta_val * action_scale.cpu().numpy()
            
            action = a0_env + 1.0 * delta_env
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            obs, reward, done, truncated, info = env.step(action)
            prev_action = action
            
        successes.append(float(info.get("is_success", 0.0)))
        
    return np.mean(successes)

def evaluate_baseline_policy(env, model, n_episodes=20):
    successes = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
        successes.append(float(info.get("is_success", 0.0)))
    return np.mean(successes)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=str, required=True)
    parser.add_argument("--dirs", type=str, required=True, help="Comma separated paths relative to log-root")
    args = parser.parse_args()
    
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    env_id = "FetchPickAndPlace-v4"
    dummy_env = gym.make(env_id) # For HER loading
    
    dirs = args.dirs.split(",")
    results = {} 
    
    # Scenarios to test
    scenarios = {
        "Nominal": {"k_obs": 0, "k_act": 0},
        "Act_Only (k=1)": {"k_obs": 0, "k_act": 1},
        "Obs_Only (k=3)": {"k_obs": 3, "k_act": 0},
    }

    print(f"\nEvaluating {len(dirs)} models...")
    print(f"{'Model':<40} | {'Nominal':<10} | {'Act(1)':<10} | {'Obs(3)':<10}")
    print("-" * 80)
    
    for d in dirs:
        full_path = os.path.join(args.log_root, d)
        
        # Detect Type
        is_residual = False
        if os.path.exists(os.path.join(full_path, "residual_mem.pt")):
            is_residual = True
            
        # Load Model
        model_name = d.split("/")[-1]
        scores = {}
        
        try:
            if is_residual:
                # Load Residual
                base_path = os.path.join(full_path, "base_sac.zip")
                if not os.path.exists(base_path): raise FileNotFoundError("base_sac.zip")
                base_model = SAC.load(base_path, env=dummy_env, device=device)
                
                ckpt = th.load(os.path.join(full_path, "residual_mem.pt"), map_location=device)
                s_dict = ckpt["state_dict"]
                obj_idx = ckpt.get("obj_idx", [3,4,5])
                
                # Reconstruct Res
                obs_dim = dummy_env.observation_space["observation"].shape[0]
                obj_dim = len(obj_idx)
                rest_dim = obs_dim - obj_dim
                fourier = FourierFeatures(obj_dim, num_bands=8, max_freq=10.0).to(device)
                input_dim = fourier.out_dim + rest_dim + 3 + 3 + 4 + obj_dim
                try:
                    residual = ResidualMemPolicy(
                        input_dim=input_dim, action_dim=4, hidden_size=128, mem_type="gru"
                    ).to(device)
                    residual.load_state_dict(s_dict)
                except RuntimeError as e:
                    if "size mismatch" in str(e):
                        print("  [Info] HIDDEN 128 failed, trying 256...")
                        residual = ResidualMemPolicy(
                            input_dim=input_dim, action_dim=4, hidden_size=256, mem_type="gru"
                        ).to(device)
                        residual.load_state_dict(s_dict)
                    else:
                        raise e
                residual.eval()
                
                # Evaluation Function
                eval_fn = lambda e: evaluate_residual_policy(e, base_model, residual, device, obj_idx, fourier)
                
            else:
                # Load Baseline
                # Try best_model.zip, then FetchPickAndPlace-v4.zip
                p = os.path.join(full_path, "best_model.zip")
                if not os.path.exists(p):
                    p = os.path.join(full_path, "FetchPickAndPlace-v4.zip")
                if not os.path.exists(p): raise FileNotFoundError("No model zip found")
                
                model = SAC.load(p, env=dummy_env, device=device)
                eval_fn = lambda e: evaluate_baseline_policy(e, model)
            
            # Run Scenarios
            for s_name, s_cfg in scenarios.items():
                eval_env = gym.make(env_id)
                eval_env = build_perturbed_env(
                    eval_env,
                    obs_delay_k=s_cfg["k_obs"],
                    action_delay_k=s_cfg["k_act"],
                    enable_disturbance_control=True
                )
                scores[s_name] = eval_fn(eval_env)
                
            # Print Row
            print(f"{model_name[:38]:<40} | {scores['Nominal']*100:5.1f}      | {scores['Act_Only (k=1)']*100:5.1f}      | {scores['Obs_Only (k=3)']*100:5.1f}")
            
        except Exception as e:
            print(f"{model_name[:38]:<40} | Error: {str(e)}")

if __name__ == "__main__":
    main()



