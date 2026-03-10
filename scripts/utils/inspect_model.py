import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC

model_path = "logs/sac-PandaSlide-v3.zip"
try:
    model = SAC.load(model_path)
    print(f"Model loaded successfully.")
    print(f"Policy Architecture: {model.policy.net_arch}")
    print(f"Policy Args: {model.policy_kwargs}")
except Exception as e:
    print(f"Error loading model: {e}")
