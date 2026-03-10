#!/usr/bin/env python3
"""
Evaluate a trained model on four test scenarios with 1000 episodes each.
"""

import os
import sys
import json
import argparse
import numpy as np
import gymnasium as gym
import gymnasium_robotics
import panda_gym  # Register Panda environments
from pathlib import Path

from stable_baselines3 import SAC
from perturbation_wrappers import build_perturbed_env

def evaluate_model(model_path, env_name, scenario, n_episodes=1000, seed=42):
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
    
    print(f"Running {n_episodes} episodes...")
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
        
        if (ep + 1) % 100 == 0:
            current_sr = np.mean(episode_successes)
            current_return = np.mean(episode_returns)
            print(f"  Episode {ep+1}/{n_episodes}: SR={current_sr:.3f}, Return={current_return:.2f}")
    
    # Compute statistics
    results = {
        "scenario": scenario,
        "n_episodes": n_episodes,
        "success_rate": float(np.mean(episode_successes)),
        "success_std": float(np.std(episode_successes)),
        "return_mean": float(np.mean(episode_returns)),
        "return_std": float(np.std(episode_returns)),
        "length_mean": float(np.mean(episode_lengths)),
        "length_std": float(np.std(episode_lengths)),
    }
    
    env.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--env", type=str, default="FetchPickAndPlace-v4", help="Environment name")
    parser.add_argument("--n-episodes", type=int, default=1000, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    scenarios = ["Nominal", "ObsOnly3", "ActOnly1", "MixedHard"]
    
    # Find model file
    model_dir = args.model
    if os.path.exists(os.path.join(model_dir, "base_sac.zip")):
        model_path = os.path.join(model_dir, "base_sac.zip")
    else:
        # Look in checkpoints
        checkpoint_dir = os.path.join(model_dir, "checkpoints")
        if os.path.exists(checkpoint_dir):
            checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')])
            if checkpoints:
                model_path = os.path.join(checkpoint_dir, checkpoints[-1])
            else:
                raise FileNotFoundError(f"No model found in {model_dir}")
        else:
            raise FileNotFoundError(f"No model found in {model_dir}")
    
    model_name = Path(model_dir).name
    
    print(f"\n{'='*80}")
    print(f"Evaluating Model: {model_name}")
    print(f"Model Path: {model_path}")
    print(f"Episodes per scenario: {args.n_episodes}")
    print(f"{'='*80}\n")
    
    all_results = []
    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"Scenario: {scenario}")
        print(f"{'='*80}")
        results = evaluate_model(model_path, args.env, scenario, args.n_episodes, args.seed)
        results["model_name"] = model_name
        all_results.append(results)
        
        print(f"\n✓ Results: SR={results['success_rate']:.1%} ± {results['success_std']:.3f}, "
              f"Return={results['return_mean']:.2f} ± {results['return_std']:.2f}\n")
    
    # Save results
    output_file = os.path.join(model_dir, f"evaluation_results_{args.n_episodes}ep.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary table
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}\n")
    print(f"{'Scenario':<15} {'Success Rate':<20} {'Mean Return':<20}")
    print("-" * 80)
    for r in all_results:
        sr_str = f"{r['success_rate']:.1%} ± {r['success_std']:.3f}"
        ret_str = f"{r['return_mean']:.2f} ± {r['return_std']:.2f}"
        print(f"{r['scenario']:<15} {sr_str:<20} {ret_str:<20}")
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
