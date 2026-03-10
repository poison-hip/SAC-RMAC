#!/usr/bin/env python3
"""
Evaluate a baseline SAC model (no residual memory) on five test scenarios with 1000 episodes each.
"""

import os
import sys
import json
import argparse
import numpy as np
import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC
from pathlib import Path

# Import the perturbation wrapper
sys.path.insert(0, str(Path(__file__).parent))
from wrappers import build_perturbed_env


def evaluate_model(model_path: str, env_name: str, scenario: dict, n_episodes: int = 1000, seed: int = 42):
    """
    Evaluate a baseline SAC model on a specific scenario.
    
    Args:
        model_path: Path to the .zip model file
        env_name: Environment name
        scenario: Dict with 'name', 'k_obs', 'k_act', 'dropout_p', 'dropout_level'
        n_episodes: Number of episodes to run
        seed: Random seed
    
    Returns:
        Dict with evaluation metrics
    """
    print(f"\n--- Scenario: {scenario['name']} ---")
    print(f"Loading model from: {model_path}")
    
    # Create base environment
    env = gym.make(env_name)
    
    # Apply perturbations
    env = build_perturbed_env(
        env,
        dropout_level=scenario.get('dropout_level', 'none'),
        dropout_p=scenario.get('dropout_p', 0.0),
        bias_b=scenario.get('bias_b', 0.0),
        action_delay_k=scenario.get('k_act', 0),
        obs_delay_k=scenario.get('k_obs', 0),
        force_wrappers=True,
        enable_disturbance_control=True,
    )
    
    # Load model
    model = SAC.load(model_path, env=env)
    
    # Run episodes
    successes = []
    returns = []
    lengths = []
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        truncated = False
        ep_return = 0.0
        ep_length = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            step_out = env.step(action)
            
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = terminated
            else:
                obs, reward, done, info = step_out
                truncated = False
            
            ep_return += reward
            ep_length += 1
        
        # Extract success
        success = float(info.get('is_success', 0.0))
        successes.append(success)
        returns.append(ep_return)
        lengths.append(ep_length)
        
        if (ep + 1) % 100 == 0:
            sr = np.mean(successes)
            print(f"  Progress: {ep + 1}/{n_episodes} episodes | Success Rate: {sr:.3f}")
    
    env.close()
    
    # Compute statistics
    success_rate = np.mean(successes)
    success_std = np.std(successes)
    return_mean = np.mean(returns)
    return_std = np.std(returns)
    length_mean = np.mean(lengths)
    length_std = np.std(lengths)
    
    print(f"  Results: SR={success_rate:.3f}±{success_std:.3f}, Return={return_mean:.2f}±{return_std:.2f}")
    
    return {
        'scenario': scenario['name'],
        'success_rate': float(success_rate),
        'success_std': float(success_std),
        'return_mean': float(return_mean),
        'return_std': float(return_std),
        'length_mean': float(length_mean),
        'length_std': float(length_std),
        'n_episodes': n_episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model directory or .zip file')
    parser.add_argument('--env', type=str, default='PandaReachDense-v3', help='Environment name')
    parser.add_argument('--n-episodes', type=int, default=1000, help='Number of episodes per scenario')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Determine model path
    if args.model.endswith('.zip'):
        model_path = args.model
        model_dir = str(Path(args.model).parent)
    else:
        model_dir = args.model
        # Try best_model.zip first, then base_sac.zip
        if os.path.exists(os.path.join(model_dir, 'best_model.zip')):
            model_path = os.path.join(model_dir, 'best_model.zip')
        elif os.path.exists(os.path.join(model_dir, 'base_sac.zip')):
            model_path = os.path.join(model_dir, 'base_sac.zip')
        else:
            raise FileNotFoundError(f"No model file found in {model_dir}")
    
    print("=" * 80)
    print(f"Evaluating Baseline Model: {model_path}")
    print("=" * 80)
    
    # Define five test scenarios
    scenarios = [
        {'name': 'Nominal', 'k_obs': 0, 'k_act': 0, 'dropout_level': 'none', 'dropout_p': 0.0, 'bias_b': 0.0},
        {'name': 'ObsOnly3', 'k_obs': 3, 'k_act': 0, 'dropout_level': 'none', 'dropout_p': 0.0, 'bias_b': 0.0},
        {'name': 'ActOnly1', 'k_obs': 0, 'k_act': 1, 'dropout_level': 'none', 'dropout_p': 0.0, 'bias_b': 0.0},
        {'name': 'MixedHard', 'k_obs': 3, 'k_act': 1, 'dropout_level': 'none', 'dropout_p': 0.0, 'bias_b': 0.0},
        {'name': 'DropoutHard', 'k_obs': 0, 'k_act': 0, 'dropout_level': 'hard', 'dropout_p': 0.5, 'bias_b': 0.0},
    ]
    
    # Run evaluation
    results = []
    for scenario in scenarios:
        result = evaluate_model(model_path, args.env, scenario, args.n_episodes, args.seed)
        results.append(result)
    
    # Save results
    output_file = os.path.join(model_dir, f'evaluation_results_{args.n_episodes}ep.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\n{'Scenario':<15} {'Success Rate':<20} {'Mean Return':<20}")
    print("-" * 80)
    for r in results:
        sr_str = f"{r['success_rate']*100:.1f}% ± {r['success_std']:.3f}"
        ret_str = f"{r['return_mean']:.2f} ± {r['return_std']:.2f}"
        print(f"{r['scenario']:<15} {sr_str:<20} {ret_str:<20}")
    
    print("\n" + "=" * 80)
    print(f"Results saved to: {output_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
