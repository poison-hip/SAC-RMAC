
import argparse
import sys
import yaml
import json
import torch as th
import numpy as np
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
from perturbation_wrappers import build_perturbed_env, parse_indices
from residual_mem_modules import FourierFeatures, ResidualMemPolicy, build_residual_input

# Use best-effort action scaling
def _unscale_action(a_scaled: th.Tensor, action_bias: th.Tensor, action_scale: th.Tensor) -> th.Tensor:
    return action_bias + action_scale * a_scaled

def _scale_action(a_env: th.Tensor, action_bias: th.Tensor, action_scale: th.Tensor) -> th.Tensor:
    denom = th.where(action_scale.abs() < 1e-8, th.ones_like(action_scale), action_scale)
    return (a_env - action_bias) / denom

# Paper settings for Fetch
PAPER_SETTINGS = {
    "Nominal":    {"k_obs": 0, "k_act": 0},
    "ObsOnly3":   {"k_obs": 3, "k_act": 0},
    "ActOnly1":   {"k_obs": 0, "k_act": 1},
    "MixedHard":  {"k_obs": 3, "k_act": 1},
}

def eval_pnp(args):
    run_dir = args.run_dir
    meta_path = f"{run_dir}/residual_mem_meta.yaml"
    config_path = f"{run_dir}/config.json"
    
    print(f"Loading metadata from {meta_path}")
    with open(meta_path, 'r') as f:
        meta = yaml.safe_load(f)
    with open(config_path, 'r') as f:
        config = json.load(f)

    env_id = config.get("env", "FetchPickAndPlace-v4")
    print(f"Base Env: {env_id}")
    
    # Load Models
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Base SAC
    base_model_path = f"{run_dir}/base_sac.zip"
    # Create dummy env for loading model structure
    dummy_env = gym.make(env_id)
    base_model = SAC.load(base_model_path, env=dummy_env, device=device)
    base_policy = base_model.policy
    base_policy.set_training_mode(False)
    
    # 2. Residual
    residual_path = f"{run_dir}/residual_mem.pt"
    
    # Reconstruct dims
    obs_space = dummy_env.observation_space
    base_obs_dim = int(obs_space["observation"].shape[0])
    ag_dim = int(obs_space["achieved_goal"].shape[0]) if "achieved_goal" in obs_space.spaces else 0
    dg_dim = int(obs_space["desired_goal"].shape[0]) if "desired_goal" in obs_space.spaces else 0
    act_dim = int(dummy_env.action_space.shape[0])
    
    obj_idx = meta["obj_idx"]
    if obj_idx is None:
         # Default for Fetch if None
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
    
    print(f"Loading Residual weights from {residual_path}")
    residual.load_state_dict(th.load(residual_path, map_location=device))
    residual.eval()
    
    dummy_env.close()

    # Evaluation Loop per setting
    results = {}
    
    for setting_name, delays in PAPER_SETTINGS.items():
        k_obs = delays["k_obs"]
        k_act = delays["k_act"]
        
        print(f"\nEvaluating Setting: {setting_name} (k_obs={k_obs}, k_act={k_act}, dropout=0.0)")
        
        env = gym.make(env_id)
        env = build_perturbed_env(
            env,
            dropout_p=0.0,
            dropout_level="none",
            action_delay_k=k_act,
            obs_delay_k=k_obs,
            bias_b=0.0,   # Nominal logic usually implies no bias too
            force_wrappers=True
        )
        
        # Action scaling
        low = th.as_tensor(env.action_space.low, device=device).float()
        high = th.as_tensor(env.action_space.high, device=device).float()
        action_scale = 0.5 * (high - low)
        action_bias = 0.5 * (high + low)
        
        successes = []
        rewards = []
        
        for i in range(args.n_episodes):
            obs, info = env.reset(seed=args.seed + i)
            done = False
            truncated = False
            ep_reward = 0
            
            # Reset Hidden
            h = residual.init_hidden(1, device=device)
            prev_action = np.zeros(act_dim, dtype=np.float32)
            
            while not (done or truncated):
                # Base Action (Deterministic)
                o_tensor = th.as_tensor(obs["observation"], device=device).float().unsqueeze(0)
                with th.no_grad():
                    a0_scaled = base_policy.actor(o_tensor, deterministic=True)
                    a0 = _unscale_action(a0_scaled, action_bias, action_scale).cpu().numpy()[0]
                
                # Residual Input
                o_obs = obs["observation"]
                o_ag = obs.get("achieved_goal", np.zeros(0))
                o_dg = obs.get("desired_goal", np.zeros(0))
                
                mask = np.ones(obj_dim, dtype=np.float32) # No dropout
                
                obj_t = th.as_tensor(o_obs[obj_idx], device=device).float().unsqueeze(0)
                # handle rest
                mask_all = np.ones(base_obs_dim, dtype=bool)
                mask_all[obj_idx] = False
                obs_rest = o_obs[mask_all]
                
                rest_t = th.as_tensor(obs_rest, device=device).float().unsqueeze(0)
                ag_t = th.as_tensor(o_ag, device=device).float().unsqueeze(0)
                dg_t = th.as_tensor(o_dg, device=device).float().unsqueeze(0)
                ap_t = th.as_tensor(prev_action, device=device).float().unsqueeze(0)
                m_t = th.as_tensor(mask, device=device).float().unsqueeze(0)
                
                obj_ff = fourier(obj_t)
                x = th.cat([obj_ff, rest_t, ag_t, dg_t, ap_t, m_t], dim=-1).unsqueeze(1)
                
                with th.no_grad():
                    delta, _, h = residual.forward_sequence(x, h, deterministic=True)
                
                delta_np = delta.squeeze(1).cpu().numpy()[0]
                delta_env = delta_np * action_scale.cpu().numpy()
                
                # Combine
                # Final alpha is 1.0
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
        print(f"  -> Success Rate: {sr*100:.1f}%, Mean Reward: {mr:.1f}")

    print("\n==========================================")
    print("FINAL RESULTS (No Dropout)")
    print("==========================================")
    for k, v in results.items():
        print(f"{k:10s} | SR: {v['sr']*100:5.1f}% | Return: {v['mr']:6.1f}")
    print("==========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    
    gym.register_envs(gymnasium_robotics)
    eval_pnp(args)
