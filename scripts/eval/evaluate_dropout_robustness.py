#!/usr/bin/env python3
"""
Evaluate baseline PandaReach model on dropout scenarios to test robustness.
Tests models trained WITHOUT dropout on environments WITH dropout.
"""

import os
import sys
import json
import argparse
import numpy as np
import gymnasium as gym
import panda_gym  # Register Panda environments
from pathlib import Path
from stable_baselines3 import SAC

# Import custom wrappers
sys.path.insert(0, str(Path(__file__).parent))
from perturbation_wrappers import DelayWrapper, ObsDropoutWrapper


def evaluate_model(model_path, env_name, scenario, n_episodes=1000, seed=42):
    """
    Evaluate a model on a specific scenario.
    
    Args:
        model_path: Path to the saved model (.zip file)
        env_name: Name of the environment
        scenario: Dict with keys 'name', 'k_obs', 'k_act', 'dropout_p', 'dropout_level'
        n_episodes: Number of episodes to evaluate
        seed: Random seed
    
    Returns:
        Dict with evaluation results
    """
    print(f"\n--- Scenario: {scenario['name']} ---")
    print(f"Config: k_obs={scenario.get('k_obs', 0)}, k_act={scenario.get('k_act', 0)}, "
          f"dropout_p={scenario.get('dropout_p', 0)}, dropout_level={scenario.get('dropout_level', 'none')}")
    print(f"Loading model from: {model_path}")
    
    # Create base environment
    env = gym.make(env_name)
    
    # Apply wrappers based on scenario
    if scenario.get('dropout_p', 0) > 0:
        env = ObsDropoutWrapper(
            env,
            p=scenario['dropout_p'],
            mode='hold-last',
            keys=('observation',)
        )
    
    if scenario.get('k_obs', 0) > 0 or scenario.get('k_act', 0) > 0:
        env = DelayWrapper(
            env,
            obs_delay_k=scenario.get('k_obs', 0),
            action_delay_k=scenario.get('k_act', 0)
        )
    
    # Load model
    model = SAC.load(model_path, env=env)
    
    # Evaluate
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_successes.append(float(info.get('is_success', 0)))
        
        if (ep + 1) % 100 == 0:
            print(f"  Episode {ep + 1}/{n_episodes}: SR={np.mean(episode_successes[-100:]):.2f}")
    
    env.close()
    
    # Compute statistics
    success_rate = np.mean(episode_successes)
    success_std = np.std(episode_successes)
    return_mean = np.mean(episode_rewards)
    return_std = np.std(episode_rewards)
    len_mean = np.mean(episode_lengths)
    len_std = np.std(episode_lengths)
    
    print(f"Results: SR={success_rate:.3f}±{success_std:.3f}, "
          f"Return={return_mean:.2f}±{return_std:.2f}, "
          f"Length={len_mean:.1f}±{len_std:.1f}")
    
    return {
        'scenario': scenario['name'],
        'config': {
            'k_obs': scenario.get('k_obs', 0),
            'k_act': scenario.get('k_act', 0),
            'dropout_p': scenario.get('dropout_p', 0),
            'dropout_level': scenario.get('dropout_level', 'none')
        },
        'n_episodes': n_episodes,
        'success_rate': float(success_rate),
        'success_std': float(success_std),
        'return_mean': float(return_mean),
        'return_std': float(return_std),
        'length_mean': float(len_mean),
        'length_std': float(len_std)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model directory')
    parser.add_argument('--env', type=str, default='PandaReachDense-v3', help='Environment name')
    parser.add_argument('--n-episodes', type=int, default=1000, help='Number of episodes per scenario')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Find model file
    model_dir = Path(args.model)
    if (model_dir / 'best_model.zip').exists():
        model_path = str(model_dir / 'best_model.zip')
    elif (model_dir / 'base_sac.zip').exists():
        model_path = str(model_dir / 'base_sac.zip')
    else:
        raise FileNotFoundError(f"No model found in {model_dir}")
    
    print("=" * 80)
    print(f"Evaluating Model: {model_dir.name}")
    print("=" * 80)
    
    # Define test scenarios - focus on dropout robustness
    scenarios = [
        # Baseline scenarios (from original test)
        {'name': 'Nominal', 'k_obs': 0, 'k_act': 0, 'dropout_p': 0, 'dropout_level': 'none'},
        {'name': 'ObsOnly3', 'k_obs': 3, 'k_act': 0, 'dropout_p': 0, 'dropout_level': 'none'},
        {'name': 'ActOnly1', 'k_obs': 0, 'k_act': 1, 'dropout_p': 0, 'dropout_level': 'none'},
        {'name': 'MixedHard', 'k_obs': 3, 'k_act': 1, 'dropout_p': 0, 'dropout_level': 'none'},
        
        # NEW: Dropout scenarios
        {'name': 'DropoutEasy', 'k_obs': 0, 'k_act': 0, 'dropout_p': 0.3, 'dropout_level': 'easy'},
        {'name': 'DropoutMed', 'k_obs': 0, 'k_act': 0, 'dropout_p': 0.5, 'dropout_level': 'medium'},
        {'name': 'DropoutHard', 'k_obs': 0, 'k_act': 0, 'dropout_p': 0.5, 'dropout_level': 'hard'},
        
        # Combined: Delay + Dropout
        {'name': 'Delay+DropoutEasy', 'k_obs': 3, 'k_act': 1, 'dropout_p': 0.3, 'dropout_level': 'easy'},
        {'name': 'Delay+DropoutMed', 'k_obs': 3, 'k_act': 1, 'dropout_p': 0.5, 'dropout_level': 'medium'},
        {'name': 'Delay+DropoutHard', 'k_obs': 3, 'k_act': 1, 'dropout_p': 0.5, 'dropout_level': 'hard'},
    ]
    
    # Evaluate on all scenarios
    all_results = []
    for scenario in scenarios:
        result = evaluate_model(model_path, args.env, scenario, args.n_episodes, args.seed)
        all_results.append(result)
    
    # Save results
    output_file = model_dir / f'dropout_robustness_results_{args.n_episodes}ep.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("DROPOUT ROBUSTNESS EVALUATION SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Scenario':<25}{'Success Rate':<21}{'Mean Return':<20}")
    print("-" * 80)
    
    # Group by category
    print("\n--- Baseline (No Dropout) ---")
    for result in all_results[:4]:
        sr = result['success_rate']
        sr_std = result['success_std']
        ret = result['return_mean']
        ret_std = result['return_std']
        print(f"{result['scenario']:<25}{sr:.1%} ± {sr_std:.3f}{'':<8}{ret:.2f} ± {ret_std:.2f}{'':<8}")
    
    print("\n--- Dropout Only ---")
    for result in all_results[4:7]:
        sr = result['success_rate']
        sr_std = result['success_std']
        ret = result['return_mean']
        ret_std = result['return_std']
        print(f"{result['scenario']:<25}{sr:.1%} ± {sr_std:.3f}{'':<8}{ret:.2f} ± {ret_std:.2f}{'':<8}")
    
    print("\n--- Delay + Dropout Combined ---")
    for result in all_results[7:]:
        sr = result['success_rate']
        sr_std = result['success_std']
        ret = result['return_mean']
        ret_std = result['return_std']
        print(f"{result['scenario']:<25}{sr:.1%} ± {sr_std:.3f}{'':<8}{ret:.2f} ± {ret_std:.2f}{'':<8}")
    
    print()
    print("=" * 80)
    print(f"Results saved to: {output_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
