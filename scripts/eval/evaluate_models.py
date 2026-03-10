#!/usr/bin/env python3
"""
Evaluate trained models on four test scenarios:
- Nominal: No disturbance
- ObsOnly3: Observation delay k_obs=3
- ActOnly1: Action delay k_act=1  
- MixedHard: Both delays k_obs=3, k_act=1
"""

import os
import sys
import json
import argparse
import numpy as np
import gymnasium as gym
import gymnasium_robotics
from pathlib import Path

from stable_baselines3 import SAC
from perturbation_wrappers import build_perturbed_env

def evaluate_model(model_path, env_name, scenario, n_episodes=50, seed=42):
    """Evaluate a model on a specific scenario."""
    
    # Define scenario parameters
    scenarios = {
        "Nominal": {"obs_delay_k": 0, "action_delay_k": 0},
        "ObsOnly3": {"obs_delay_k": 3, "action_delay_k": 0},
        "ActOnly1": {"obs_delay_k": 0, "action_delay_k": 1},
        "MixedHard": {"obs_delay_k": 3, "action_delay_k": 1},
    }
    
    if scenario not in scenarios:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    params = scenarios[scenario]
    
    # Create environment
    base_env = gym.make(env_name)
    env = build_perturbed_env(
        base_env,
        obs_delay_k=params["obs_delay_k"],
        action_delay_k=params["action_delay_k"],
        enable_disturbance_control=True,
    )
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = SAC.load(model_path, env=env)
    
    # Run evaluation
    episode_returns = []
    episode_successes = []
    episode_lengths = []
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        episode_return = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            episode_length += 1
            done = terminated or truncated
        
        success = info.get("is_success", 0.0)
        episode_returns.append(episode_return)
        episode_successes.append(success)
        episode_lengths.append(episode_length)
        
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes}: SR={np.mean(episode_successes):.2f}, "
                  f"Return={np.mean(episode_returns):.2f}")
    
    # Compute statistics
    results = {
        "scenario": scenario,
        "n_episodes": n_episodes,
        "success_rate": float(np.mean(episode_successes)),
        "return_mean": float(np.mean(episode_returns)),
        "return_std": float(np.std(episode_returns)),
        "length_mean": float(np.mean(episode_lengths)),
        "length_std": float(np.std(episode_lengths)),
    }
    
    env.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", type=str, required=True, help="Path to first model directory")
    parser.add_argument("--model2", type=str, required=True, help="Path to second model directory")
    parser.add_argument("--env", type=str, default="FetchPickAndPlace-v4", help="Environment name")
    parser.add_argument("--n-episodes", type=int, default=50, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    scenarios = ["Nominal", "ObsOnly3", "ActOnly1", "MixedHard"]
    
    # Evaluate both models
    for model_idx, model_dir in enumerate([args.model1, args.model2], 1):
        model_path = os.path.join(model_dir, "base_sac.zip")
        model_name = Path(model_dir).name
        
        print(f"\n{'='*80}")
        print(f"Evaluating Model {model_idx}: {model_name}")
        print(f"{'='*80}\n")
        
        all_results = []
        for scenario in scenarios:
            print(f"\n--- Scenario: {scenario} ---")
            results = evaluate_model(model_path, args.env, scenario, args.n_episodes, args.seed)
            results["model_name"] = model_name
            all_results.append(results)
            
            print(f"Results: SR={results['success_rate']:.2%}, "
                  f"Return={results['return_mean']:.2f}±{results['return_std']:.2f}")
        
        # Save results
        output_file = os.path.join(model_dir, "evaluation_results.json")
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    print(f"\n{'='*80}")
    print("Evaluation complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
