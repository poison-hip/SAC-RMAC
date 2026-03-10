
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import SAC
from perturbation_wrappers import DelayWrapper

MODEL_PATH = "logs/FetchPickAndPlace-v4/SAC_250204_065306/best_model.zip"

def debug_env_model():
    print("Creating Env...")
    gym.register_envs(gymnasium_robotics)
    env = gym.make("FetchPickAndPlace-v4")
    env = DelayWrapper(env, action_delay_k=1, obs_delay_k=0, role="action")

    # Inspect limit
    limit = getattr(env, "_max_episode_steps", None)
    if hasattr(env, "spec") and env.spec:
         limit = env.spec.max_episode_steps
    print(f"Max Episode Steps: {limit}")

    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = SAC.load(MODEL_PATH, env=env, device="cpu")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"FAILED to load model: {e}")
        return

    obs, _ = env.reset(seed=42)
    
    print("Testing Inference & Specific Action...")
    try:
        action, _ = model.predict(obs, deterministic=True)
        print(f"Predicted Action: {action}")
        
        # Step
        obs2, r, term, trunc, info = env.step(action)
        print(f"Step Result: r={r}, term={term}, trunc={trunc}, done={term or trunc}")
        if (term or trunc):
            print("ALERT: Premature termination detected with Policy Action.")
            
    except Exception as e:
        print(f"Inference/Step Failed: {e}")

if __name__ == "__main__":
    debug_env_model()
