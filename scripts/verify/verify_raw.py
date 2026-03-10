
import gymnasium as gym
import gymnasium_robotics
import panda_gym
import numpy as np
import torch as th
from stable_baselines3 import SAC
from perturbation_wrappers import build_perturbed_env
import argparse

def _unscale_action(a_scaled, low, high):
    action_scale = 0.5 * (high - low)
    action_bias = 0.5 * (high + low)
    return action_bias + action_scale * a_scaled

def _torch_obs_dict(obs, device):
    # Mimic trainer logic
    out = {}
    for k, v in obs.items():
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        out[k] = th.as_tensor(arr, device=device)
    return out

def verify():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    args = parser.parse_args()

    # 1. Raw Env Setup (Exact match to train.py n_envs=1 optimization)
    gym.register_envs(gymnasium_robotics)
    e = gym.make("PandaReachDense-v3")
    env = build_perturbed_env(e, dropout_p=0.0, obs_delay_k=3, action_delay_k=1)
    
    print(f"Env: {env}")
    
    # 2. Model Load
    print(f"Loading model from {args.model_path}")
    model = SAC.load(args.model_path, env=env, device="cpu")
    actor = model.policy.actor
    model.policy.set_training_mode(False) # Force eval mode
    
    # 3. Loop with model.predict
    print("--- Testing model.predict ---")
    obs, _ = env.reset()
    total_rew = 0
    for i in range(50):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_rew += reward
        if terminated or truncated:
            obs, _ = env.reset()
    print(f"Total Reward (predict): {total_rew}")

    # 4. Loop with manual _compute_base_a0_env
    print("--- Testing manual compute ---")
    obs, _ = env.reset()
    low = th.as_tensor(env.action_space.low).float()
    high = th.as_tensor(env.action_space.high).float()
    total_rew = 0
    
    for i in range(50):
        obs_t = _torch_obs_dict(obs, model.device)
        with th.no_grad():
            a0_scaled = actor(obs_t, deterministic=True)
            a0_env = _unscale_action(a0_scaled, low, high)
        
        action = a0_env.numpy()[0] # squeeze (1, dim) -> (dim,)
        
        obs, reward, terminated, truncated, _ = env.step(action)
        total_rew += reward
        if terminated or truncated:
             obs, _ = env.reset()
             
    print(f"Total Reward (manual): {total_rew}")
    env.close()

if __name__ == "__main__":
    verify()
