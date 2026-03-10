import gymnasium as gym
try:
    import panda_gym
    env = gym.make('PandaSlide-v3')
    print("Success: PandaSlide-v3 created.")
    env.close()
except ImportError:
    print("Error: panda_gym not installed.")
except Exception as e:
    print(f"Error creating environment: {e}")
