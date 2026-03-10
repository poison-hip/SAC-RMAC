
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

def _torch_obs_dict(obs, device):
    # Mimic trainer logic for VecEnv (obs is already batched numpy)
    out = {}
    for k, v in obs.items():
        arr = np.asarray(v, dtype=np.float32)
        # VecEnv obs is already (B, ...), so no manual unsqueeze if we assume B=1 or B=N
        out[k] = th.as_tensor(arr, device=device)
    return out

def verify():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    args = parser.parse_args()

    # 1. DummyVecEnv Setup
    gym.register_envs(gymnasium_robotics)
    def make_env():
        e = gym.make("PandaReachDense-v3")
        return build_perturbed_env(e, dropout_p=0.0, obs_delay_k=3, action_delay_k=1)
    
    env = DummyVecEnv([make_env])
    
    print(f"Env: {env}")
    
    # 2. Model Load
    print(f"Loading model from {args.model_path}")
    model = SAC.load(args.model_path, env=env, device="cpu")
    actor = model.policy.actor
    model.policy.set_training_mode(False) # Force eval mode
    
    act_space = env.action_space
    low = th.as_tensor(act_space.low).float()
    high = th.as_tensor(act_space.high).float()

    # 3. Loop with manual _compute (VecEnv style)
    print("--- Testing manual compute (DummyVecEnv) ---")
    obs = env.reset()
    total_rew = 0
    
    for i in range(50):
        # obs is already batched dict from VecEnv
        
        # 1. Model Predict
        action_pred, _ = model.predict(obs, deterministic=True)
        
        # 2. Manual Compute
        obs_t = _torch_obs_dict(obs, model.device)
        with th.no_grad():
            a0_scaled = actor(obs_t, deterministic=True)
            a0_env = _unscale_action(a0_scaled, low, high)
        action_manual = a0_env.cpu().numpy()
        
        # Compare
        diff = np.abs(action_pred - action_manual).max()
        if diff > 1e-5:
            print(f"Step {i} DIFF: {diff}")
            print(f"Pred: {action_pred[0]}")
            print(f"Manu: {action_manual[0]}")
        
        action = action_pred # Use correct one to proceed
        
        obs, reward, done, infos = env.step(action)
        total_rew += reward[0]
             
    print(f"Total Reward (manual vec): {total_rew}")
    env.close()

if __name__ == "__main__":
    verify()
