
import os
import argparse
import gymnasium as gym
try:
    import panda_gym
except ImportError:
    pass
import numpy as np
import torch as th
from stable_baselines3 import SAC
from perturbation_wrappers import build_perturbed_env

def make_eval_env(env_id, setting_id):
    k_obs, k_act = 0, 0
    if setting_id == "Nominal":
        pass
    elif setting_id == "ObsOnly3":
        k_obs = 3
    elif setting_id == "ActOnly1":
        k_act = 1
    elif setting_id == "MixedHard":
        k_obs, k_act = 3, 1
    else:
        raise ValueError(f"Unknown setting: {setting_id}")

    env = gym.make(env_id)
    env = build_perturbed_env(
        env,
        dropout_level="none", # User confirmed dropout is bad, so we test delay only
        dropout_p=0.0,
        obs_delay_k=k_obs,
        action_delay_k=k_act,
        force_wrappers=True,
        enable_disturbance_control=True
    )
    return env

def evaluate(model, env, n_episodes=20):
    successes = []
    rewards = []
    
    for i in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_rew = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            
            # DEBUG PRINT
            if np.random.rand() < 0.05:
                 o_vec = obs["observation"]
                 print(f"[VERIFY] Obs: {o_vec.flatten()[:3]}... Mean: {np.mean(o_vec):.3f}")
                 print(f"[VERIFY] Action: {action.flatten()[:3]}... Mean: {np.mean(action):.3f}")

            obs, reward, done, truncated, info = env.step(action)
            ep_rew += reward
            
        is_success = info.get("is_success", 0.0) if "is_success" in info else (1.0 if info.get("success", False) else 0.0)
        successes.append(is_success)
        rewards.append(ep_rew)
        
    return np.mean(successes), np.mean(rewards)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--env-id", type=str, default="PandaReachDense-v3")
    args = parser.parse_args()

    # Create VecEnv to test DummyVecEnv issues
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    print("Testing with DummyVecEnv...")
    def make_wrapped_env():
        import gymnasium as gym
        import panda_gym 
        from perturbation_wrappers import build_perturbed_env
        e = gym.make("PandaReachDense-v3")
        return build_perturbed_env(e, dropout_p=0.0, obs_delay_k=3, action_delay_k=1)

    dummy_env = DummyVecEnv([make_wrapped_env for _ in range(1)])
    
    print(f"Loading model from: {args.model_path}")
    model = SAC.load(args.model_path, env=dummy_env)
    
    # DEBUG: Check model weights
    s = sum([p.data.sum() for p in model.policy.actor.parameters()])
    print(f"[VERIFY] Actor weight sum: {s}")
    
    # Eval loop using the vec env
    obs = dummy_env.reset()
    for _ in range(50):
        action, _ = model.predict(obs, deterministic=True)
        # DEBUG CHECK
        if np.random.rand() < 0.1:
             # Extract observation if dict
             if isinstance(obs, dict):
                 print(f"[VERIFY] Obs: {obs['observation'].flatten()[:3]}")
                 print(f"[VERIFY] AG: {obs['achieved_goal'].flatten()[:3]}")
                 print(f"[VERIFY] DG: {obs['desired_goal'].flatten()[:3]}")
             else:
                 print(f"[VERIFY] Obs: {obs.flatten()[:3]}")
             print(f"[VERIFY] Action: {action.flatten()[:3]}")

        # Check action shape
        # print(f"Action shape: {action.shape}")
        obs, rewards, dones, infos = dummy_env.step(action)
        print(f"[VERIFY Dummy] Rewards: {rewards}")
    
    dummy_env.close()
    exit()

    settings = ["Nominal", "ObsOnly3", "ActOnly1", "MixedHard"]
    
    print("-" * 60)
    print(f"{'Setting':<15} | {'SR':<10} | {'Return':<10}")
    print("-" * 60)

    for setting in settings:
        env = make_eval_env(args.env_id, setting)
        sr, ret = evaluate(model, env, n_episodes=20)
        print(f"{setting:<15} | {sr:<10.2f} | {ret:<10.2f}")
        env.close()
    print("-" * 60)
