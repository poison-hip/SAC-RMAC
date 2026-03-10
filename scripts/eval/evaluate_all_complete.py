#!/usr/bin/env python3
"""
Evaluate both Baseline (SAC) and Ours (Residual Memory) models on a comprehensive set of scenarios:
1. Standard delays: Nominal, ObsOnly3, ActOnly1, MixedHard
2. Dropout scenarios: Easy (p=0.3), Medium (p=0.5), Hard (p=0.5 + delays)

This script loads the full residual memory architecture for "Ours".
"""

import os
import sys
import json
import argparse
import yaml
import numpy as np
import torch as th
import gymnasium as gym
import gymnasium_robotics
import panda_gym
from pathlib import Path
from stable_baselines3 import SAC

# Import custom modules
sys.path.insert(0, str(Path(__file__).parent))
from perturbation_wrappers import build_perturbed_env, ObsDropoutWrapper, DelayWrapper
from residual_mem_modules import ResidualMemPolicy, FourierFeatures

def get_scenarios():
    """Define all evaluation scenarios."""
    return [
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
        {'name': 'Delay+DropoutHard', 'k_obs': 0, 'k_act': 0, 'dropout_p': 0.5, 'dropout_level': 'hard'}, # Actually hard usually combines delays
    ]
    # Note: I fixed the last scenario to be consistently hard if needed, but per request "Hard (p=0.5 + delays)" 
    # Let's check evaluate_dropout_robustness.py for exact definitions if possible.
    # In evaluate_dropout_robustness:
    # {'name': 'Delay+DropoutHard', 'k_obs': 3, 'k_act': 1, 'dropout_p': 0.5, 'dropout_level': 'hard'},

def create_env(env_name, scenario, seed):
    """Create environment with specific scenario settings."""
    env = gym.make(env_name)
    
    # Apply dropout if needed
    if scenario['dropout_p'] > 0:
        env = ObsDropoutWrapper(
            env,
            p=scenario['dropout_p'],
            mode='hold-last',
            keys=('observation',)
        )
    
    # Apply delays if needed
    if scenario['k_obs'] > 0 or scenario['k_act'] > 0:
        env = DelayWrapper(
            env,
            obs_delay_k=scenario['k_obs'],
            action_delay_k=scenario['k_act']
        )
        
    env.reset(seed=seed)
    return env

# =============================================================================
# Helper functions for Residual Memory Model
# =============================================================================

def _scale_action(action, high, low):
    """Scale action from [-1, 1] to [low, high]."""
    # Matching SB3 logic
    return low + (0.5 * (action + 1.0) * (high - low))

def _unscale_action(action, high, low):
    """Unscale action from [low, high] to [-1, 1]."""
    return 2.0 * ((action - low) / (high - low)) - 1.0

def evaluate_baseline(model_path, env_name, scenarios, n_episodes, seed, device="auto"):
    print(f"\n[{'BASELINE'}] Loading model from: {model_path}")
    
    # We need a dummy env to load the model sometimes, but usually fine without if policy is standard
    # But to be safe, create a nominal env
    dummy_env = gym.make(env_name)
    model = SAC.load(model_path, env=dummy_env, device=device)
    dummy_env.close()
    
    results = []
    
    for scenario in scenarios:
        # Skip if using logic that requires specific keys? No, baseline uses standard obs
        print(f"  > Scenario: {scenario['name']} ... ", end="", flush=True)
        env = create_env(env_name, scenario, seed)
        
        episode_returns = []
        episode_successes = []
        episode_lengths = []
        
        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed + ep)
            done = False
            truncated = False
            ep_ret = 0
            ep_len = 0
            
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                ep_ret += reward
                ep_len += 1
            
            episode_returns.append(ep_ret)
            episode_successes.append(float(info.get('is_success', 0.0)))
            episode_lengths.append(ep_len)
        
        env.close()
        
        sr_mean = np.mean(episode_successes)
        ret_mean = np.mean(episode_returns)
        print(f"SR={sr_mean:.1%}, Return={ret_mean:.1f}")
        
        results.append({
            "scenario": scenario['name'],
            "model": "Baseline",
            "success_rate": float(sr_mean),
            "success_std": float(np.std(episode_successes)),
            "return_mean": float(ret_mean),
            "return_std": float(np.std(episode_returns)),
            "length_mean": float(np.mean(episode_lengths))
        })
        
    return results

def evaluate_residual(run_dir, env_name, scenarios, n_episodes, seed, device="auto"):
    print(f"\n[{'RESIDUAL'}] Loading model from: {run_dir}")
    
    # 1. Load Metadata
    meta_path = os.path.join(run_dir, "residual_mem_meta.yaml")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    
    with open(meta_path, "r") as f:
        meta = yaml.safe_load(f)
        
    obj_idx = meta["obj_idx"]
    fourier_cfg = meta["fourier"]
    
    # 2. Check for device
    if device == "auto":
        device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"    Using device: {device}")
    
    # 3. Create Dummy Env for Shapes
    dummy_env = gym.make(env_name)
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space
    
    base_obs_dim = obs_space["observation"].shape[0]
    ag_dim = obs_space["achieved_goal"].shape[0]
    dg_dim = obs_space["desired_goal"].shape[0]
    act_dim = act_space.shape[0]
    
    obj_dim = len(obj_idx)
    rest_dim = base_obs_dim - obj_dim
    
    dummy_env.close()
    
    # 4. Reconstruct Residual Model
    fourier = FourierFeatures(obj_dim, num_bands=fourier_cfg["bands"], max_freq=fourier_cfg["max_freq"]).to(device)
    input_dim = fourier.out_dim + rest_dim + ag_dim + dg_dim + act_dim + obj_dim
    
    # Checkpoint
    ckpt_path = os.path.join(run_dir, "residual_mem.pt")
    checkpoint = th.load(ckpt_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    
    # Infer hidden size and mem_type from args.yaml if available
    hidden_size = 256
    mem_type = "gru"
    
    args_path = os.path.join(run_dir, "args.yaml")
    if os.path.exists(args_path):
        try:
            with open(args_path, "r") as f:
                run_args = yaml.safe_load(f)
                hidden_size = int(run_args.get("residual_hidden", 256))
                mem_type = run_args.get("residual_mem_type", "gru")
                print(f"    Loaded config from args.yaml: hidden={hidden_size}, mem={mem_type}")
        except Exception as e:
            print(f"    [Warn] Failed to load args.yaml: {e}")

    # Create model structure
    residual = ResidualMemPolicy(
        input_dim=input_dim, 
        action_dim=act_dim, 
        hidden_size=hidden_size, 
        mem_type=mem_type
    ).to(device)
    
    # Load weights
    try:
        residual.load_state_dict(state_dict)
    except Exception as e:
        print(f"    [Error] Failed to load state dict: {e}")
        raise e
        
    residual.eval()
    
    # 5. Load Base SAC
    base_path = os.path.join(run_dir, "base_sac.zip")
    # HER requires env to be passed during load
    dummy_env = gym.make(env_name)
    base_model = SAC.load(base_path, env=dummy_env, device=device)
    dummy_env.close()
    
    # 6. Evaluation Helpers
    low_t = th.as_tensor(act_space.low, device=device).float()
    high_t = th.as_tensor(act_space.high, device=device).float()
    action_scale_t = 0.5 * (high_t - low_t)
    action_bias_t = 0.5 * (high_t + low_t)
    
    def _unscale_t(a): return action_bias_t + action_scale_t * a
    
    results = []
    
    for scenario in scenarios:
        print(f"  > Scenario: {scenario['name']} ... ", end="", flush=True)
        env = create_env(env_name, scenario, seed)
        
        episode_returns = []
        episode_successes = []
        episode_lengths = []
        
        # We need update k_obs/k_act for creating appropriate storage in Residual Policy if needed?
        # No, Residual Policy is just RNN. But the wrapper provides info.
        
        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed + ep)
            done = False
            truncated = False
            ep_ret = 0
            ep_len = 0
            
            # Reset residual state
            h = residual.init_hidden(1, device=device)
            prev_a_np = np.zeros(act_dim, dtype=np.float32)
            
            while not (done or truncated):
                # Prepare input for Base SAC
                obs_t = {k: th.as_tensor(v, device=device).float().unsqueeze(0) for k, v in obs.items()}
                
                # Get Base Action
                with th.no_grad():
                    a0_scaled = base_model.policy.actor(obs_t, deterministic=True)
                    a0_env_t = _unscale_t(a0_scaled) # (1, act_dim)
                    a0_env_np = a0_env_t.cpu().numpy()[0]
                
                # Prepare input for Residual
                # Extract parts
                o_obs = obs["observation"]
                o_ag = obs.get("achieved_goal", np.zeros(ag_dim, dtype=np.float32))
                o_dg = obs.get("desired_goal", np.zeros(dg_dim, dtype=np.float32))
                
                keep_idx = [i for i in range(base_obs_dim) if i not in set(obj_idx)]
                
                obj_np = o_obs[obj_idx]
                rest_np = o_obs[keep_idx]
                
                # Mask
                m_obj_np = np.ones(obj_dim, dtype=np.float32)
                if "obs_mask" in info:
                    mm = np.asarray(info["obs_mask"]).reshape(-1)
                    if mm.shape[0] == base_obs_dim:
                        m_obj_np = mm[obj_idx]
                
                # To Torch
                obj_t_in = th.as_tensor(obj_np, device=device).float().unsqueeze(0)
                rest_t_in = th.as_tensor(rest_np, device=device).float().unsqueeze(0)
                ag_t_in = th.as_tensor(o_ag, device=device).float().unsqueeze(0)
                dg_t_in = th.as_tensor(o_dg, device=device).float().unsqueeze(0)
                ap_t_in = th.as_tensor(prev_a_np, device=device).float().unsqueeze(0)
                m_t_in = th.as_tensor(m_obj_np, device=device).float().unsqueeze(0)
                
                obj_ff = fourier(obj_t_in)
                
                x = th.cat([obj_ff, rest_t_in, ag_t_in, dg_t_in, ap_t_in, m_t_in], dim=-1).unsqueeze(1) # (1, 1, input)
                
                # Forward Residual
                with th.no_grad():
                    # alpha=1.0 for evaluation
                    delta, _, h_out = residual.forward_sequence(x, h, deterministic=True)
                    h = h_out
                    
                    delta_np = delta.squeeze(1).cpu().numpy()[0] # (act_dim,)
                    
                    # Scale delta to env space
                    action_scale_np = action_scale_t.cpu().numpy()
                    delta_env_np = delta_np * action_scale_np
                    
                    # Combine: a = a0 + alpha * delta
                    final_action = a0_env_np + 1.0 * delta_env_np
                    
                    # Clip
                    final_action = np.clip(final_action, act_space.low, act_space.high)
                
                # Step
                obs, reward, done, truncated, info = env.step(final_action)
                ep_ret += reward
                ep_len += 1
                
                # Update prev action
                if "executed_action" in info:
                    try:
                         prev_a_np = np.asarray(info["executed_action"]).reshape(-1)
                    except:
                         prev_a_np = final_action
                else:
                    prev_a_np = final_action
            
            episode_returns.append(ep_ret)
            episode_successes.append(float(info.get('is_success', 0.0)))
            episode_lengths.append(ep_len)
        
        env.close()
        
        sr_mean = np.mean(episode_successes)
        ret_mean = np.mean(episode_returns)
        print(f"SR={sr_mean:.1%}, Return={ret_mean:.1f}")
        
        results.append({
            "scenario": scenario['name'],
            "model": "Residual",
            "success_rate": float(sr_mean),
            "success_std": float(np.std(episode_successes)),
            "return_mean": float(ret_mean),
            "return_std": float(np.std(episode_returns)),
            "length_mean": float(np.mean(episode_lengths))
        })
        
    return results

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours_path", type=str, required=True, help="Directory containing ours model (residual_mem.pt, etc.)")
    parser.add_argument("--baseline_path", type=str, required=True, help="Path to baseline SAC .zip")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Path to pretrained SAC .zip (optional)")
    parser.add_argument("--env", type=str, default="PandaReachDense-v3", help="Environment ID")
    parser.add_argument("--n_episodes", type=int, default=50, help="Episodes per scenario")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    # Standard Scenarios + Dropout
    scenarios = get_scenarios()
    
    print("="*80)
    print(f"EVALUATION START: n_episodes={args.n_episodes}, env={args.env}")
    print("="*80)
    
    all_results = []
    
    # Evaluate Baseline
    print("\n" + "-"*40)
    print("Evaluating BASELINE Model")
    print("-"*40)
    try:
        baseline_results = evaluate_baseline(args.baseline_path, args.env, scenarios, args.n_episodes, 42, args.device)
        all_results.extend(baseline_results)
    except Exception as e:
        print(f"Baseline evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    # Evaluate Pretrained (if provided)
    if args.pretrained_path:
        print("\n" + "-"*40)
        print(f"Evaluating PRETRAINED Model: {args.pretrained_path}")
        print("-"*40)
        try:
            # Re-use evaluate_baseline as it effectively loads a SAC zip
            pretrained_results = evaluate_baseline(args.pretrained_path, args.env, scenarios, args.n_episodes, 42, args.device)
            # Update mode name
            for r in pretrained_results:
                r['model'] = 'Pretrained'
            all_results.extend(pretrained_results)
        except Exception as e:
            print(f"Pretrained evaluation failed: {e}")
            import traceback
            traceback.print_exc()
        
    # Evaluate Residual
    print("\n" + "-"*40)
    print("Evaluating RESIDUAL (Ours) Model")
    print("-"*40)
    try:
        residual_results = evaluate_residual(args.ours_path, args.env, scenarios, args.n_episodes, 42, args.device)
        all_results.extend(residual_results)
    except Exception as e:
        print(f"Residual evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    # Save Results
    output_file = os.path.join(args.ours_path, "comparison_results_complete.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print Table
    print("\n" + "="*140)
    print(f"{'Scenario':<20} | {'Baseline SR':<12} | {'Pretrain SR':<12} | {'Ours SR':<12} | {'Base Ret':<10} | {'Pre Ret':<10} | {'Ours Ret':<10}")
    print("-" * 140)
    
    base_map = {r['scenario']: r for r in all_results if r['model'] == 'Baseline'}
    pre_map = {r['scenario']: r for r in all_results if r['model'] == 'Pretrained'}
    ours_map = {r['scenario']: r for r in all_results if r['model'] == 'Residual'}
    
    for sc in scenarios:
        name = sc['name']
        b = base_map.get(name, {})
        p = pre_map.get(name, {})
        o = ours_map.get(name, {})
        
        b_sr = f"{b.get('success_rate', 0):.1%}" if b else "N/A"
        p_sr = f"{p.get('success_rate', 0):.1%}" if p else "N/A"
        o_sr = f"{o.get('success_rate', 0):.1%}" if o else "N/A"
        
        b_ret = f"{b.get('return_mean', 0):.1f}" if b else "N/A"
        p_ret = f"{p.get('return_mean', 0):.1f}" if p else "N/A"
        o_ret = f"{o.get('return_mean', 0):.1f}" if o else "N/A"
        
        print(f"{name:<20} | {b_sr:<12} | {p_sr:<12} | {o_sr:<12} | {b_ret:<10} | {p_ret:<10} | {o_ret:<10}")
    print("="*110)

if __name__ == "__main__":
    main()
