"""Evaluation script for trained RL models on robotic manipulation tasks. (currently using SAC)"""

from __future__ import annotations

# Make repo root importable when running as: python scripts/eval/eval.py
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import os
import argparse
import json
import re
import zipfile

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from stable_baselines3 import DDPG, TD3, SAC
from perturbation_wrappers import build_perturbed_env, parse_indices
from delay_compensation import HistoryObsActWrapper
from conservative_wrappers import MotorCostConfig, MotorCostWrapper, predict_with_uncertainty_gating

# configuration
CONFIG = {
    "env_id": "FetchPickAndPlace-v4",
    "model_class": "SAC",
    "seed": 1,
    "n_eval_episodes": 10,
    "render": True,
    "log_dir": "./logs",
}


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent.")
    parser.add_argument(
        "--model",
        type=str,
        default=CONFIG["model_class"],
        choices=["DDPG", "TD3", "SAC"],
        help="RL model type",
    )
    parser.add_argument(
        "--env", type=str, default=CONFIG["env_id"], help="Gymnasium environment ID"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=CONFIG["n_eval_episodes"],
        help="Number of evaluation episodes",
    )
    parser.add_argument("--seed", type=int, default=CONFIG["seed"], help="Random seed")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Explicit path to a Stable-Baselines3 .zip model to evaluate "
            "(overrides auto-discovery under logs/)."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help=(
            "Run directory name under logs/<env>/ (e.g. SAC_250204_065306). "
            "If provided, evaluation will load from that run."
        ),
    )

    # Delay-aware history wrapper (must match how the model was trained)
    parser.add_argument(
        "--history-k",
        type=int,
        default=None,
        help=(
            "History length K for delay-aware input. If omitted, eval.py will try to infer it "
            "from the loaded model's observation dim."
        ),
    )

    # --- motor-friendly cost (Module D) ---
    parser.add_argument(
        "--log-motor-metrics",
        action="store_true",
        help="If set, wrap env to log motor_energy/motor_jerk per step in info (and print episode averages).",
    )
    parser.add_argument(
        "--motor-energy-w",
        type=float,
        default=0.0,
        help="Weight for energy proxy ||a||^2 (used for motor_cost; metrics are always logged when enabled).",
    )
    parser.add_argument(
        "--motor-jerk-w",
        type=float,
        default=0.0,
        help="Weight for jerk proxy ||a_t-a_{t-1}||^2 (used for motor_cost; metrics are always logged when enabled).",
    )
    parser.add_argument(
        "--motor-cost-in-reward",
        action="store_true",
        help="If set, subtract motor cost from reward during evaluation (usually leave off; mainly for debugging).",
    )

    # --- uncertainty-aware action gating (Module E/F) ---
    parser.add_argument(
        "--uncertainty-gating",
        action="store_true",
        help="Enable uncertainty-aware action gating based on critic disagreement (double-Q).",
    )
    parser.add_argument(
        "--uncertainty-threshold",
        type=float,
        default=1.0,
        help="Uncertainty threshold u0. If u<=u0, gate g=1; else g=exp(-alpha*(u-u0)).",
    )
    parser.add_argument(
        "--uncertainty-alpha",
        type=float,
        default=1.0,
        help="Gating sharpness alpha (bigger -> more conservative when uncertain).",
    )
    parser.add_argument(
        "--uncertainty-min-g",
        type=float,
        default=0.0,
        help="Lower bound on gate g in [0,1].",
    )

    # --- robustness / perturbations ---
    parser.add_argument(
        "--obs-dropout-level",
        type=str,
        default="none",
        choices=["none", "easy", "med", "hard"],
        help="Observation dropout severity (none/easy/med/hard).",
    )
    parser.add_argument(
        "--obs-dropout-p",
        type=float,
        default=None,
        help="Override dropout probability p directly (takes priority over --obs-dropout-level).",
    )
    parser.add_argument(
        "--obs-dropout-mode",
        type=str,
        default="hold-last",
        choices=["drop-to-zero", "hold-last"],
        help="Dropout mode: drop-to-zero or hold-last.",
    )
    parser.add_argument(
        "--obs-dropout-keys",
        type=str,
        default="observation,achieved_goal",
        help="Comma-separated dict observation keys to perturb (default: observation,achieved_goal).",
    )
    parser.add_argument(
        "--obs-dropout-obj-idx",
        type=str,
        default=None,
        help="Comma-separated indices into obs['observation'] to perturb (e.g. '3,4,5'). Default: whole vector.",
    )
    parser.add_argument(
        "--obs-dropout-exclude-gripper",
        action="store_true",
        help="When --obs-dropout-obj-idx is not set, exclude first 3 dims (gripper position) from dropout.",
    )
    parser.add_argument(
        "--action-delay-k",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Action delay in control cycles (k steps).",
    )
    parser.add_argument(
        "--obs-delay-k",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Observation delay in control cycles (k steps).",
    )
    parser.add_argument(
        "--obs-bias-b",
        type=float,
        default=0.0,
        help="Per-episode constant bias magnitude b (Uniform([-b,b])) in meters.",
    )
    parser.add_argument(
        "--obs-bias-keys",
        type=str,
        default="observation,achieved_goal",
        help="Comma-separated dict observation keys to bias (default: observation,achieved_goal).",
    )
    parser.add_argument(
        "--obs-bias-obj-idx",
        type=str,
        default=None,
        help="Comma-separated indices into obs['observation'] to bias (e.g. '3,4,5'). Default: auto [3,4,5] for Fetch.",
    )
    return parser.parse_args()


def _select_model_path(log_dir: str, run_dir: str | None, env_name: str, model_name: str) -> str:
    """
    Select a model path from logs.

    Preference order:
    1) If run_dir is provided: choose best_model.zip if present else <env_name>.zip
    2) Otherwise: choose the latest run that contains best_model.zip
    3) Otherwise: choose the latest run (even if it doesn't have best_model.zip)
    """
    prefix = f"{model_name}_"
    candidates = [d for d in os.listdir(log_dir) if d.startswith(prefix)]
    if not candidates:
        raise FileNotFoundError(f"No {model_name} runs found in {log_dir}")

    candidates = sorted(candidates, reverse=True)

    def pick_from_run(run: str) -> str:
        run_path = os.path.join(log_dir, run)
        best_path = os.path.join(run_path, "best_model.zip")
        last_path = os.path.join(run_path, f"{env_name}.zip")
        if os.path.exists(best_path):
            return best_path
        if os.path.exists(last_path):
            return last_path
        raise FileNotFoundError(
            f"Run exists but no model zip found in {run_path} (expected best_model.zip or {env_name}.zip)"
        )

    if run_dir is not None:
        if run_dir not in candidates:
            raise FileNotFoundError(
                f"Requested run_dir '{run_dir}' not found under {log_dir}. Available: {candidates[:10]}"
            )
        return pick_from_run(run_dir)

    # prefer the newest run that has best_model.zip
    for run in candidates:
        if os.path.exists(os.path.join(log_dir, run, "best_model.zip")):
            return pick_from_run(run)

    # fallback: newest run
    return pick_from_run(candidates[0])


def _read_model_obs_dim(model_zip_path: str) -> int | None:
    """Extract obs['observation'] dim from SB3 .zip without instantiating the model (HER requires env)."""
    try:
        with zipfile.ZipFile(model_zip_path, "r") as z:
            data = json.loads(z.read("data").decode("utf-8"))
        spaces_str = data.get("observation_space", {}).get("spaces", "")
        m = re.search(r"'observation':\s*Box\([^\)]*\(\s*(\d+)\s*,\s*\)", str(spaces_str))
        return int(m.group(1)) if m else None
    except Exception:
        return None


def _infer_history_k(model_obs_dim: int, env_obs_dim: int, act_dim: int) -> int | None:
    """
    Solve: model_obs_dim = env_obs_dim + k * (env_obs_dim + act_dim)
    where env_obs_dim is the base obs['observation'] dim (no history).
    """
    if model_obs_dim == env_obs_dim:
        return 0
    num = model_obs_dim - env_obs_dim
    den = env_obs_dim + act_dim
    if num <= 0 or den <= 0 or num % den != 0:
        return None
    k = int(num // den)
    return k if k >= 0 else None


def main():
    # parse arguments and update config
    args = parse_args()
    CONFIG.update(
        {
            "model_class": args.model,
            "env_id": args.env,
            "seed": args.seed,
            "n_eval_episodes": args.episodes,
            "render": not args.no_render,
        }
    )

    # setup model paths
    env_name = CONFIG["env_id"]
    model_name = CONFIG["model_class"]
    log_dir = os.path.join(CONFIG["log_dir"], env_name)

    model_path = args.model_path or _select_model_path(log_dir, args.run_dir, env_name, model_name)

    model_class = {
        "DDPG": DDPG,
        "TD3": TD3,
        "SAC": SAC,
    }[CONFIG["model_class"]]

    # environment setup
    gym.register_envs(gymnasium_robotics)
    env = gym.make(CONFIG["env_id"], render_mode="human" if CONFIG["render"] else None)

    dropout_obj_idx = parse_indices(args.obs_dropout_obj_idx)
    bias_obj_idx = parse_indices(args.obs_bias_obj_idx)
    dropout_keys = tuple([k.strip() for k in args.obs_dropout_keys.split(",") if k.strip()])
    bias_keys = tuple([k.strip() for k in args.obs_bias_keys.split(",") if k.strip()])
    dropout_level = None if args.obs_dropout_p is not None else args.obs_dropout_level
    dropout_p = float(args.obs_dropout_p) if args.obs_dropout_p is not None else 0.0

    env = build_perturbed_env(
        env,
        dropout_level=dropout_level,
        dropout_p=dropout_p,
        dropout_mode=args.obs_dropout_mode,
        dropout_keys=dropout_keys,
        dropout_exclude_gripper=args.obs_dropout_exclude_gripper,
        dropout_obj_idx=dropout_obj_idx,
        action_delay_k=args.action_delay_k,
        obs_delay_k=args.obs_delay_k,
        bias_b=args.obs_bias_b,
        bias_keys=bias_keys,
        bias_obj_idx=bias_obj_idx,
    )

    # History wrapper must be outermost, and must match the trained model.
    if args.history_k is not None:
        hist_k = int(args.history_k)
    else:
        model_obs_dim = _read_model_obs_dim(model_path)
        if model_obs_dim is None:
            hist_k = 0
        else:
            env_obs_dim = int(env.observation_space.spaces["observation"].shape[0])
            act_dim = int(env.action_space.shape[0])
            hist_k = _infer_history_k(int(model_obs_dim), env_obs_dim, act_dim) or 0

    if hist_k > 0:
        print(f"Using history_k={hist_k} (delay-aware observation augmentation)")
        env = HistoryObsActWrapper(env, history_k=hist_k)

    # Motor metrics wrapper (outermost, after history)
    if args.log_motor_metrics or float(args.motor_energy_w) != 0.0 or float(args.motor_jerk_w) != 0.0 or bool(args.motor_cost_in_reward):
        env = MotorCostWrapper(
            env,
            cfg=MotorCostConfig(
                energy_w=float(args.motor_energy_w),
                jerk_w=float(args.motor_jerk_w),
                add_to_reward=bool(args.motor_cost_in_reward),
                include_in_compute_reward=True,
            ),
        )

    env.reset(seed=CONFIG["seed"])

    print(f"Loading model from: {model_path}")
    # HER models require an env at load time.
    model = model_class.load(model_path, env=env)

    # evaluation loop
    rewards = []
    successes = []

    try:
        for episode in range(CONFIG["n_eval_episodes"]):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            energy_sum = 0.0
            jerk_sum = 0.0
            g_sum = 0.0
            u_sum = 0.0

            while not done:
                if args.uncertainty_gating:
                    action, _state, diag = predict_with_uncertainty_gating(
                        model,
                        obs,
                        deterministic=True,
                        threshold=float(args.uncertainty_threshold),
                        alpha=float(args.uncertainty_alpha),
                        min_g=float(args.uncertainty_min_g),
                    )
                else:
                    action, _ = model.predict(obs, deterministic=True)
                    diag = None

                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                steps += 1

                if isinstance(info, dict):
                    energy_sum += float(info.get("motor_energy", 0.0))
                    jerk_sum += float(info.get("motor_jerk", 0.0))
                if diag is not None:
                    g_sum += float(diag.get("g", 1.0))
                    u_sum += float(diag.get("uncertainty", 0.0))

            rewards.append(episode_reward)
            successes.append(info.get("is_success", 0))

            print(
                f"Episode {episode + 1}: Reward = {episode_reward:.2f}, "
                f"Success = {info.get('is_success', 0)}"
            )
            if args.log_motor_metrics:
                denom = float(steps) if steps > 0 else 1.0
                print(
                    f"  motor: E[||a||^2]={energy_sum/denom:.4f}, E[||da||^2]={jerk_sum/denom:.4f}"
                    + (
                        f", gate: E[g]={g_sum/denom:.3f}, E[u]={u_sum/denom:.3f}"
                        if args.uncertainty_gating
                        else ""
                    )
                )

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")

    finally:
        # print summary statistics
        if rewards:
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            success_rate = np.mean(successes) * 100

            print("\nEvaluation Results:")
            print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"Success rate: {success_rate:.1f}%")

        # cleanup
        env.close()


if __name__ == "__main__":
    main()



