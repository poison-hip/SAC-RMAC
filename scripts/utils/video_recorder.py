# modified from https://gymnasium.farama.org/introduction/record_agent/

from __future__ import annotations

# Make repo root importable when running as: python scripts/utils/video_recorder.py
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
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from stable_baselines3 import SAC
from perturbation_wrappers import build_perturbed_env, parse_indices
from delay_compensation import HistoryObsActWrapper
from conservative_wrappers import MotorCostConfig, MotorCostWrapper, predict_with_uncertainty_gating

gym.register_envs(gymnasium_robotics)

# configuration
DEFAULT_ENV_ID = "FetchPickAndPlace-v4"
DEFAULT_ALGO = "SAC"
DEFAULT_NUM_EVAL_EPISODES = 3
DEFAULT_LOG_DIR = "./logs"


def parse_args():
    parser = argparse.ArgumentParser(description="Record videos for a trained RL agent.")
    parser.add_argument("--env", type=str, default=DEFAULT_ENV_ID, help="Gymnasium environment ID")
    parser.add_argument("--algo", type=str, default=DEFAULT_ALGO, help="Algorithm name (folder prefix under logs/)")
    parser.add_argument("--episodes", type=int, default=DEFAULT_NUM_EVAL_EPISODES, help="Number of episodes to record")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=DEFAULT_LOG_DIR,
        help="Base log directory that contains logs/<env>/<algo>_<timestamp>/",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Explicit path to a Stable-Baselines3 .zip model (overrides auto-discovery under logs/).",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory name under logs/<env>/ (e.g. SAC_250204_065306).",
    )

    # Delay-aware history wrapper (must match how the model was trained)
    parser.add_argument(
        "--history-k",
        type=int,
        default=None,
        help=(
            "History length K for delay-aware input. If omitted, video_recorder.py will try to infer it "
            "from the loaded model's observation dim."
        ),
    )

    # --- motor-friendly cost (Module D) ---
    parser.add_argument(
        "--log-motor-metrics",
        action="store_true",
        help="If set, wrap env to log motor_energy/motor_jerk per step in info (and print episode averages).",
    )
    parser.add_argument("--motor-energy-w", type=float, default=0.0)
    parser.add_argument("--motor-jerk-w", type=float, default=0.0)
    parser.add_argument("--motor-cost-in-reward", action="store_true")

    # --- uncertainty-aware action gating (Module E/F) ---
    parser.add_argument("--uncertainty-gating", action="store_true")
    parser.add_argument("--uncertainty-threshold", type=float, default=1.0)
    parser.add_argument("--uncertainty-alpha", type=float, default=1.0)
    parser.add_argument("--uncertainty-min-g", type=float, default=0.0)

    # --- robustness / perturbations ---
    parser.add_argument("--obs-dropout-level", type=str, default="none", choices=["none", "easy", "med", "hard"])
    parser.add_argument("--obs-dropout-p", type=float, default=None)
    parser.add_argument("--obs-dropout-mode", type=str, default="hold-last", choices=["drop-to-zero", "hold-last"])
    parser.add_argument("--obs-dropout-keys", type=str, default="observation,achieved_goal")
    parser.add_argument("--obs-dropout-obj-idx", type=str, default=None)
    parser.add_argument("--obs-dropout-exclude-gripper", action="store_true")

    parser.add_argument("--action-delay-k", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--obs-delay-k", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--obs-bias-b", type=float, default=0.0)
    parser.add_argument("--obs-bias-keys", type=str, default="observation,achieved_goal")
    parser.add_argument("--obs-bias-obj-idx", type=str, default=None)

    # output
    parser.add_argument("--out-dir", type=str, default="./videos", help="Output directory for videos.")
    parser.add_argument("--name-prefix", type=str, default=None, help="Video filename prefix (default: derived).")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def _select_model_path(log_dir: str, env_name: str, algo: str, run_dir: str | None) -> str:
    prefix = f"{algo}_"
    env_dir = os.path.join(log_dir, env_name)
    if not os.path.isdir(env_dir):
        raise FileNotFoundError(f"Log directory not found: {env_dir}")

    runs = [d for d in os.listdir(env_dir) if d.startswith(prefix)]
    if not runs:
        raise FileNotFoundError(f"No {algo} runs found in {env_dir}")
    runs = sorted(runs, reverse=True)

    def pick_from_run(run: str) -> str:
        run_path = os.path.join(env_dir, run)
        best = os.path.join(run_path, "best_model.zip")
        last = os.path.join(run_path, f"{env_name}.zip")
        if os.path.exists(best):
            return best
        if os.path.exists(last):
            return last
        raise FileNotFoundError(f"No model zip found in {run_path}")

    if run_dir is not None:
        if run_dir not in runs:
            raise FileNotFoundError(f"Requested run_dir '{run_dir}' not found under {env_dir}.")
        return pick_from_run(run_dir)

    for run in runs:
        if os.path.exists(os.path.join(env_dir, run, "best_model.zip")):
            return pick_from_run(run)
    return pick_from_run(runs[0])


def _read_model_obs_dim(model_zip_path: str) -> int | None:
    try:
        with zipfile.ZipFile(model_zip_path, "r") as z:
            data = json.loads(z.read("data").decode("utf-8"))
        spaces_str = data.get("observation_space", {}).get("spaces", "")
        m = re.search(r"'observation':\s*Box\([^\)]*\(\s*(\d+)\s*,\s*\)", str(spaces_str))
        return int(m.group(1)) if m else None
    except Exception:
        return None


def _infer_history_k(model_obs_dim: int, env_obs_dim: int, act_dim: int) -> int | None:
    if model_obs_dim == env_obs_dim:
        return 0
    num = model_obs_dim - env_obs_dim
    den = env_obs_dim + act_dim
    if num <= 0 or den <= 0 or num % den != 0:
        return None
    k = int(num // den)
    return k if k >= 0 else None


def main():
    args = parse_args()

    model_path = args.model_path or _select_model_path(args.log_dir, args.env, args.algo, args.run_dir)
    print(f"Loading model from: {model_path}")

    env = gym.make(args.env, render_mode="rgb_array")

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
        dropout_exclude_gripper=bool(args.obs_dropout_exclude_gripper),
        dropout_obj_idx=dropout_obj_idx,
        action_delay_k=int(args.action_delay_k),
        obs_delay_k=int(args.obs_delay_k),
        bias_b=float(args.obs_bias_b),
        bias_keys=bias_keys,
        bias_obj_idx=bias_obj_idx,
    )

    # history wrapper
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

    # motor wrapper (optional)
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

    # video wrappers
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    prefix = args.name_prefix or f"{args.algo}_{args.env}"
    env = RecordEpisodeStatistics(env)
    env = RecordVideo(env, video_folder=out_dir, name_prefix=prefix, episode_trigger=lambda ep: True, fps=int(args.fps))

    model = SAC.load(model_path, env=env)

    for ep in range(int(args.episodes)):
        obs, _ = env.reset(seed=int(args.seed) + ep)
        done = False
        while not done:
            if args.uncertainty_gating:
                action, _state, _diag = predict_with_uncertainty_gating(
                    model,
                    obs,
                    deterministic=True,
                    threshold=float(args.uncertainty_threshold),
                    alpha=float(args.uncertainty_alpha),
                    min_g=float(args.uncertainty_min_g),
                )
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, _r, terminated, truncated, _info = env.step(action)
            done = terminated or truncated

    env.close()
    print(f"Wrote videos to: {out_dir}")


if __name__ == "__main__":
    main()



