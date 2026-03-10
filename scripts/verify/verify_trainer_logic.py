
import gymnasium as gym
import gymnasium_robotics
import panda_gym
import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from perturbation_wrappers import build_perturbed_env
import argparse

def _unscale_action(a_scaled, low, high):
    action_scale = 0.5 * (high - low)
    action_bias = 0.5 * (high + low)
    return action_bias + action_scale * a_scaled

def verify():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    args = parser.parse_args()

    # 1. Environment Setup (Same as trainer)
    def make_env():
        e = gym.make("PandaReachDense-v3")
        # Easy mode (k=1)
        e = build_perturbed_env(e, dropout_p=0.0, obs_delay_k=1, action_delay_k=1)
        return e

    env = DummyVecEnv([make_env])
    obs_space = env.observation_space
    act_space = env.action_space
    
    print(f"Action Space: {act_space}")
    
    # 2. Model Load
    print(f"Loading model from {args.model_path}")
    model = SAC.load(args.model_path, env=env, device="cpu") # Force CPU for simplicity
    actor = model.policy.actor
    
    # 3. Loop
    obs = env.reset()
    low = th.as_tensor(act_space.low).float()
    high = th.as_tensor(act_space.high).float()
    
    total_rew = 0
    for i in range(50):
        # Trainer logic replica
        # _as_batch_obs -> already done by DummyVecEnv (obs is dict of (1, dim))
        
        # _torch_obs_dict
        obs_t = {k: th.as_tensor(v).float() for k, v in obs.items()}
        
        with th.no_grad():
            a0_scaled = actor(obs_t, deterministic=True)
            a0_env = _unscale_action(a0_scaled, low, high)
            
        action = a0_env.numpy()
        
        # Step
        obs, rewards, dones, infos = env.step(action)
        total_rew += rewards[0]
        # print(f"Step {i}: Reward {rewards[0]}")
        
    print(f"Total Reward (50 steps): {total_rew}")
    env.close()

if __name__ == "__main__":
    verify()
