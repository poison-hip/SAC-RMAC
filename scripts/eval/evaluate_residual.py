
from __future__ import annotations

# Make repo root importable when running as: python scripts/eval/evaluate_residual.py
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
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from perturbation_wrappers import build_perturbed_env, parse_indices
from residual_mem_modules import FourierFeatures, ResidualMemPolicy

# Register envs
gym.register_envs(gymnasium_robotics)

def make_env(env_id, rank, seed, disturbance_kwargs):
    def _init():
        e = gym.make(env_id)
        e = build_perturbed_env(e, **disturbance_kwargs)
        e.reset(seed=seed + rank)
        e.action_space.seed(seed + rank)
        return e
    return _init

def evaluate_residual_policy_vec(venv, base_model, residual_policy, device, obj_idx, fourier, n_episodes):
    n_envs = venv.num_envs
    # Per-env counters
    completed_episodes = 0
    success_count = 0
    
    # We need to track how many episodes each env has finished if we want exact N episodes.
    # But simpler: run until total completed >= n_episodes.
    
    # Action scaling
    low = th.as_tensor(venv.action_space.low[0], device=device).float() # Assume same space
    high = th.as_tensor(venv.action_space.high[0], device=device).float()
    action_scale = 0.5 * (high - low)
    action_bias = 0.5 * (high + low)
    
    def _unscale_action(a_scaled):
        return action_bias + action_scale * a_scaled

    obs = venv.reset()
    
    # Init recurrent state: (1, n_envs, hidden)
    h = residual_policy.init_hidden(n_envs, device=device)
    prev_action = np.zeros((n_envs, venv.action_space.shape[0]), dtype=np.float32)
    
    while completed_episodes < n_episodes:
        # 1. Base Policy
        a0_env, _ = base_model.predict(obs, deterministic=True) # (n_envs, act_dim)
        
        # 2. Residual Input
        o_obs = obs["observation"]
        o_ag = obs.get("achieved_goal", np.zeros((n_envs, 0)))
        o_dg = obs.get("desired_goal", np.zeros((n_envs, 0)))
        
        # Extract object parts for all envs
        obj_part = o_obs[:, obj_idx] # (n_envs, obj_dim)
        
        # Rest part
        all_indices = np.arange(o_obs.shape[1])
        rest_mask = np.isin(all_indices, obj_idx, invert=True)
        rest_part = o_obs[:, rest_mask] # (n_envs, rest_dim)
        
        # To Tensor
        obj_t = th.as_tensor(obj_part, device=device).float() # (n_envs, obj_dim)
        obj_ff = fourier(obj_t) # (n_envs, out_dim)
        
        rest_t = th.as_tensor(rest_part, device=device).float()
        ag_t = th.as_tensor(o_ag, device=device).float()
        dg_t = th.as_tensor(o_dg, device=device).float()
        ap_t = th.as_tensor(prev_action, device=device).float()
        m_t = th.ones((n_envs, len(obj_idx)), device=device).float() # mask
        
        # Cat (Batch, InputDim)
        x = th.cat([obj_ff, rest_t, ag_t, dg_t, ap_t, m_t], dim=-1)
        
        # Unsqueeze time dim: (Batch, 1, InputDim)
        x_seq = x.unsqueeze(1)
        
        # 3. Residual Forward
        with th.no_grad():
            delta, _, h_out = residual_policy.forward_sequence(x_seq, h, deterministic=True)
        h = h_out
        
        delta_val = delta.squeeze(1) # (Batch, ActDim) in [-1, 1]
        
        # Scale Delta
        delta_env = delta_val * action_scale # Element-wise broadcasting
        
        # Combine
        delta_env_np = delta_env.cpu().numpy()
        
        action = a0_env + 1.0 * delta_env_np
        action = np.clip(action, venv.action_space.low[0], venv.action_space.high[0])
        
        # Step
        obs, rewards, dones, infos = venv.step(action)
        prev_action = action.copy()
        
        # Reset hidden states and prev_action for done envs
        for i, done in enumerate(dones):
            if done:
                # Reset H for this env
                h[:, i, :] = 0.0 # (1, Batch, Hidden)
                prev_action[i] = 0.0
                
                # Check success
                if completed_episodes < n_episodes:
                    is_success = infos[i].get("is_success", 0.0)
                    success_count += float(is_success)
                    completed_episodes += 1
                    
        # Print progress occasionally
        if completed_episodes % 200 == 0 and completed_episodes > 0:
            print(f"\r  ... {completed_episodes}/{n_episodes}", end="", flush=True)

    print(f"\r  ... {completed_episodes}/{n_episodes} Done.")
    return success_count / completed_episodes

def evaluate_standard_vec(venv, model, n_episodes):
    completed_episodes = 0
    success_count = 0
    
    obs = venv.reset()
    
    while completed_episodes < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = venv.step(action)
        
        for i, done in enumerate(dones):
            if done:
                if completed_episodes < n_episodes:
                    is_success = infos[i].get("is_success", 0.0)
                    success_count += float(is_success)
                    completed_episodes += 1
                    
        if completed_episodes % 200 == 0 and completed_episodes > 0:
             print(f"\r  ... {completed_episodes}/{n_episodes}", end="", flush=True)
             
    print(f"\r  ... {completed_episodes}/{n_episodes} Done.")
    return success_count / completed_episodes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=str, required=True)
    parser.add_argument("--dirs", type=str, required=True, help="Comma separated folder names in log-root")
    parser.add_argument("--n-episodes", type=int, default=50, help="Number of episodes per setting")
    parser.add_argument("--n-envs", type=int, default=16, help="Number of parallel environments")
    args = parser.parse_args()
    
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    env_id = "FetchPickAndPlace-v4"
    dummy_env = gym.make(env_id) # For loading/dimensions
    
    dirs = args.dirs.split(",")
    results = {} 
    
    for d in dirs:
        full_path = os.path.join(args.log_root, d)
        print(f"\n--- Loading: {d} ---")
        
        # Check residual
        res_path = os.path.join(full_path, "residual_mem.pt")
        is_residual = os.path.exists(res_path)
        
        scores = {}
        
        # Prepare Model
        if is_residual:
            base_path = os.path.join(full_path, "base_sac.zip")
            if not os.path.exists(base_path):
                print(f"[Warn] base_sac.zip not found. Skipping.")
                continue
            try:
                base_model = SAC.load(base_path, env=dummy_env, device=device) # Required for HerReplayBuffer
            except Exception as e:
                print(f"[Error] Load failed: {e}")
                continue
                
            checkpoint = th.load(res_path, map_location=device)
            obj_idx = checkpoint.get("obj_idx", [3,4,5])
            state_dict = checkpoint["state_dict"]
            
            # Dimensions
            obs_dim = dummy_env.observation_space["observation"].shape[0]
            obj_dim = len(obj_idx)
            rest_dim = obs_dim - obj_dim
            # fourier
            fourier = FourierFeatures(obj_dim, num_bands=8, max_freq=10.0).to(device)
            input_dim = fourier.out_dim + rest_dim + 3 + 3 + 4 + obj_dim # ag, dg, act, mask
            
            residual = ResidualMemPolicy(input_dim, 4, 128, "gru").to(device)
            try:
                residual.load_state_dict(state_dict)
            except RuntimeError:
                residual = ResidualMemPolicy(input_dim, 4, 256, "gru").to(device)
                residual.load_state_dict(state_dict)
            residual.eval()
            
            eval_func = lambda v: evaluate_residual_policy_vec(v, base_model, residual, device, obj_idx, fourier, args.n_episodes)
            
        else:
            model_path = os.path.join(full_path, "best_model.zip")
            if not os.path.exists(model_path): model_path = os.path.join(full_path, "FetchPickAndPlace-v4.zip")
            if not os.path.exists(model_path): 
                print("No model found. Skipping.")
                continue
                
            print(f"Standard SAC: {model_path}")
            model = SAC.load(model_path, env=dummy_env, device=device)
            eval_func = lambda v: evaluate_standard_vec(v, model, args.n_episodes)

        # Run Scenarios
        scenarios = {
            "Nominal": {"k_obs": 0, "k_act": 0},
            "Obs_Only (k=3)": {"k_obs": 3, "k_act": 0},
            "Act_Only (k=1)": {"k_obs": 0, "k_act": 1},
            "Mixed (k=3,a=1)": {"k_obs": 3, "k_act": 1},
        }
        
        for s_name, s_cfg in scenarios.items():
            # Build VecEnv
            dist_kwargs = {
                "obs_delay_k": s_cfg["k_obs"],
                "action_delay_k": s_cfg["k_act"],
                "enable_disturbance_control": True
            }
            
            env_fns = [make_env(env_id, i, 1000, dist_kwargs) for i in range(args.n_envs)]
            venv = SubprocVecEnv(env_fns)
            
            try:
                score = eval_func(venv)
                scores[s_name] = score
                print(f"  {s_name}: {score*100:.1f}%")
            finally:
                venv.close()
                
        results[d] = scores

    print("\n\n====== FINAL SUMMARY (Success Rate %) ======")
    print(f"{'Model':<40} | {'Nominal':<10} | {'Obs(3)':<10} | {'Act(1)':<10} | {'Mixed':<10}")
    print("-" * 90)
    for d, sc in results.items():
        print(f"{d[:38]:<40} | {sc.get('Nominal',0)*100:5.1f}      | {sc.get('Obs_Only (k=3)',0)*100:5.1f}      | {sc.get('Act_Only (k=1)',0)*100:5.1f}      | {sc.get('Mixed (k=3,a=1)',0)*100:5.1f}")

if __name__ == "__main__":
    main()



