
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
from residual_mem_modules import FourierFeatures, ResidualMemPolicy, build_residual_input

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

def eval_reach_dir(run_dir, seed_override=None, n_episodes=1000):
    meta_path = f"{run_dir}/residual_mem_meta.yaml"
    config_path = f"{run_dir}/config.json"
    
    # Check if files exist
    if not os.path.exists(meta_path) or not os.path.exists(config_path):
        print(f"[Warn] Missing meta/config in {run_dir}, skipping.")
        return None

    with open(meta_path, 'r') as f:
        meta = yaml.safe_load(f)
    with open(config_path, 'r') as f:
        config = json.load(f)

    env_id = config.get("env", "PandaReachDense-v3")
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # 1. Base SAC
    base_model_path = f"{run_dir}/base_sac.zip"
    if not os.path.exists(base_model_path):
         print(f"[Warn] Missing base_sac.zip in {run_dir}, skipping.")
         return None

    dummy_env = gym.make(env_id)
    base_model = SAC.load(base_model_path, env=dummy_env, device=device)
    base_policy = base_model.policy
    base_policy.set_training_mode(False)
    
    # 2. Residual
    residual_path = f"{run_dir}/residual_mem.pt"
    if not os.path.exists(residual_path):
         print(f"[Warn] Missing residual_mem.pt in {run_dir}, skipping.")
         return None
    
    # Reconstruct dims
    obs_space = dummy_env.observation_space
    base_obs_dim = int(obs_space["observation"].shape[0])
    ag_dim = int(obs_space["achieved_goal"].shape[0]) if "achieved_goal" in obs_space.spaces else 0
    dg_dim = int(obs_space["desired_goal"].shape[0]) if "desired_goal" in obs_space.spaces else 0
    act_dim = int(dummy_env.action_space.shape[0])
    
    obj_idx = meta["obj_idx"]
    if obj_idx is None:
         # Default [3,4,5] used in training if None
         obj_idx = [3, 4, 5]
    
    obj_dim = len(obj_idx)
    rest_dim = base_obs_dim - obj_dim
    
    fourier_bands = meta["fourier"]["bands"]
    fourier_max_freq = meta["fourier"]["max_freq"]
    
    fourier = FourierFeatures(obj_dim, num_bands=fourier_bands, max_freq=fourier_max_freq).to(device)
    input_dim = int(fourier.out_dim + rest_dim + ag_dim + dg_dim + act_dim + obj_dim)
    
    hidden_size = int(config.get("residual_hidden", 128))
    mem_type = config.get("residual_mem_type", "gru")
    
    residual = ResidualMemPolicy(
        input_dim=input_dim,
        action_dim=act_dim,
        hidden_size=hidden_size,
        mem_type=mem_type
    ).to(device)
    
    checkpoint = th.load(residual_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        residual.load_state_dict(checkpoint["state_dict"])
    else:
        residual.load_state_dict(checkpoint)
    residual.eval()
    
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
        
        # Consistent seeding for evaluation
        # Use fixed seed for comparison across models? 
        # User said "1000 episode 和之前的测试用的环境要一样", meaning likely the same seed sequence.
        # Standard practice: reset(seed=seed_base + i)
        
        env = build_perturbed_env(
            env,
            dropout_p=dropout_p,
            dropout_level=dropout_level,
            action_delay_k=k_act,
            obs_delay_k=k_obs,
            bias_b=0.0,   
            force_wrappers=True
        )
        
        # Action scaling
        low = th.as_tensor(env.action_space.low, device=device).float()
        high = th.as_tensor(env.action_space.high, device=device).float()
        action_scale = 0.5 * (high - low)
        action_bias = 0.5 * (high + low)
        
        successes = []
        rewards = []
        
        for i in range(n_episodes):
            obs, info = env.reset(seed=seed + 2000 + i) # Offset 2000 to avoid training overlap
            done = False
            truncated = False
            ep_reward = 0
            
            h = residual.init_hidden(1, device=device)
            prev_action = np.zeros(act_dim, dtype=np.float32)
            
            while not (done or truncated):
                # Base Action
                # o_tensor = th.as_tensor(obs["observation"], device=device).float().unsqueeze(0)
                # Proper dict construction for HER-based policies
                obs_t = {k: th.as_tensor(v, device=device).float().unsqueeze(0) for k, v in obs.items() if k in ["observation", "achieved_goal", "desired_goal"]}
                
                with th.no_grad():
                    # Base policy expects dict if trained on Dict space
                    a0_scaled = base_policy.actor(obs_t, deterministic=True)
                    a0 = _unscale_action(a0_scaled, action_bias, action_scale).cpu().numpy()[0]
                
                # Residual Input
                o_obs = obs["observation"]
                o_ag = obs.get("achieved_goal", np.zeros(0))
                o_dg = obs.get("desired_goal", np.zeros(0))
                
                # Mask handling:
                # If dropout happened, constructing 'mask' input for residual is crucial.
                # The 'info' dict from wrapper should contain 'obs_mask'.
                # Reference: evaluate_all_complete.py logic
                
                # Default "visible"
                mask_obj_np = np.ones(obj_dim, dtype=np.float32) 
                
                if "obs_mask" in info:
                     # obs_mask is full dim (base_obs_dim)
                     mm = np.asarray(info["obs_mask"]).reshape(-1)
                     if mm.shape[0] == base_obs_dim:
                         mask_obj_np = mm[obj_idx]

                obj_t = th.as_tensor(o_obs[obj_idx], device=device).float().unsqueeze(0)
                
                mask_all = np.ones(base_obs_dim, dtype=bool)
                mask_all[obj_idx] = False
                obs_rest = o_obs[mask_all]
                
                rest_t = th.as_tensor(obs_rest, device=device).float().unsqueeze(0)
                ag_t = th.as_tensor(o_ag, device=device).float().unsqueeze(0)
                dg_t = th.as_tensor(o_dg, device=device).float().unsqueeze(0)
                ap_t = th.as_tensor(prev_action, device=device).float().unsqueeze(0)
                m_t = th.as_tensor(mask_obj_np, device=device).float().unsqueeze(0)
                
                obj_ff = fourier(obj_t)
                x = th.cat([obj_ff, rest_t, ag_t, dg_t, ap_t, m_t], dim=-1).unsqueeze(1)
                
                with th.no_grad():
                    delta, _, h = residual.forward_sequence(x, h, deterministic=True)
                
                delta_np = delta.squeeze(1).cpu().numpy()[0]
                delta_env = delta_np * action_scale.cpu().numpy()
                
                action = a0 + 1.0 * delta_env
                action = np.clip(action, env.action_space.low, env.action_space.high)
                
                obs, r, done, truncated, info = env.step(action)
                ep_reward += r
                
                if "executed_action" in info:
                    prev_action = info["executed_action"]
                else:
                    prev_action = action
            
            success = float(info.get("is_success", 0.0))
            successes.append(success)
            rewards.append(ep_reward)
        
        env.close()
        sr = np.mean(successes)
        mr = np.mean(rewards)
        results[setting_name] = {"sr": sr, "mr": mr}

    return results

def eval_all(args):
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
        seed_str = d.split("seed")[-1].split("__")[0]
        # Skip if not a valid seed folder format? No, just try best effort
        print(f"--- Evaluating Seed {seed_str} at {os.path.basename(d)} ---")
        res = eval_reach_dir(d, n_episodes=n_episodes)
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
    
    # Order matters: use PAPER_SETTINGS order
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
    print(f"FINAL AGGREGATED RESULTS (N={n} Seeds)")
    print("==========================================================================================")
    print(f"{'Scenario':<20} | {'SR Mean':<8} | {'SR Std':<8} | {'Return':<8}")
    print("-" * 80)
    for name in scenario_names:
        stats = avg_stats[name]
        print(f"{name:<20} | {stats['sr_mean']*100:5.1f}%  | {stats['sr_std']*100:5.1f}%  | {stats['mr_mean']:6.2f}")
    print("==========================================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, required=True, help="Path to directory containing runs (e.g. runs/our)")
    parser.add_argument("--n-episodes", type=int, default=1000)
    args = parser.parse_args()
    
    eval_all(args)
