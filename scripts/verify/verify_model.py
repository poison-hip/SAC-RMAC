import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
import numpy as np

env_id = "PandaReachDense-v3"
model_path = "logs/sac-PandaReachDense-v3.zip"

print(f"Creating {env_id}...")
env = gym.make(env_id)

print(f"Loading model from {model_path}...")
try:
    model = SAC.load(model_path, env=env)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

print("Evaluating for 20 episodes...")
for ep in range(20):
    obs, _ = env.reset()
    done = False
    truncated = False
    total_rew = 0
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, truncated, info = env.step(action)
        total_rew += rew
    
    is_success = info.get("is_success", False)
    print(f"Episode {ep}: Reward={total_rew:.2f}, Success={is_success}")

env.close()
