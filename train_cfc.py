"""
Train CfC-SAC(+HER) on Gymnasium-Robotics with an "industrial disturbance suite":

- Observation latency (k_obs): perception / comms delay (obs_delay_k)
- Action latency (k_act): actuation / comms delay (action_delay_k)
- Sampling jitter (Δt): passed explicitly to CfC as timespans
- Systematic bias (b): per-episode constant bias (calibration drift)
- Occlusion / frame-drop: dropout / hold-last

CfC-specific "defensibility" additions:
  1) Feed Δt via CfC(timespans=...)
  2) Feed (k_obs, k_act) as explicit scalar features
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Optional

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import yaml
from stable_baselines3 import DDPG, TD3, SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from cfc_modules import CfCFeatureExtractor, CfCInputWrapper, DtJitterConfig
from conservative_wrappers import MotorCostConfig, MotorCostWrapper
from perturbation_wrappers import build_perturbed_env, parse_indices


def _make_yaml_safe(x):
    """Best-effort conversion to YAML-serializable objects (for reproducibility logs)."""
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _make_yaml_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_make_yaml_safe(v) for v in x]
    return str(x)


def parse_args():
    p = argparse.ArgumentParser(description="Train CfC-SAC(+HER) on Gymnasium-Robotics.")
    p.add_argument("--model", type=str, default="SAC", choices=["DDPG", "TD3", "SAC"])
    p.add_argument("--env", type=str, default="FetchPickAndPlace-v4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_dir", type=str, default="./logs")
    p.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    p.add_argument("--pretrained", type=str, default=None, help="Path to SB3 .zip model to resume from.")
    p.add_argument("--no-load-replay-buffer", action="store_true")

    # --- vec env / throughput ---
    p.add_argument("--n-envs", type=int, default=1, help="Number of parallel environments (n_envs).")
    p.add_argument(
        "--vec-env",
        type=str,
        default="dummy",
        choices=["dummy", "subproc"],
        help="Vectorized env backend. 'subproc' uses multiple processes (faster sampling, more overhead).",
    )

    # --- robustness / domain perturbations ---
    p.add_argument("--obs-dropout-level", type=str, default="none", choices=["none", "easy", "med", "hard"])
    p.add_argument("--obs-dropout-p", type=float, default=None)
    p.add_argument("--obs-dropout-mode", type=str, default="hold-last", choices=["drop-to-zero", "hold-last"])
    p.add_argument("--obs-dropout-keys", type=str, default="observation")
    p.add_argument("--obs-dropout-obj-idx", type=str, default=None)
    p.add_argument("--obs-dropout-exclude-gripper", action="store_true")

    p.add_argument("--action-delay-k", type=int, default=0, choices=[0, 1, 2, 3])
    p.add_argument("--obs-delay-k", type=int, default=0, choices=[0, 1, 2, 3])

    p.add_argument("--obs-bias-b", type=float, default=0.0)
    p.add_argument("--obs-bias-keys", type=str, default="observation")
    p.add_argument("--obs-bias-obj-idx", type=str, default=None)

    # --- CfC config ---
    p.add_argument("--cfc-window", type=int, default=8, help="History window length T for CfC input.")
    p.add_argument("--cfc-include-actions", action="store_true", help="Include action history in CfC input.")
    p.add_argument("--cfc-units", type=int, default=256)
    p.add_argument("--cfc-mixed-memory", action="store_true")
    p.add_argument("--cfc-mode", type=str, default="default", choices=["default", "pure", "no_gate"])
    p.add_argument("--goal-embed-dim", type=int, default=64)

    # Δt sampling jitter fed to CfC(timespans)
    p.add_argument("--dt-jitter-mode", type=str, default="none", choices=["none", "uniform", "normal"])
    p.add_argument("--dt-min", type=float, default=1.0)
    p.add_argument("--dt-max", type=float, default=1.0)
    p.add_argument("--dt-mean", type=float, default=1.0)
    p.add_argument("--dt-std", type=float, default=0.0)

    # --- motor-friendly cost (Module D) ---
    p.add_argument("--motor-energy-w", type=float, default=0.0)
    p.add_argument("--motor-jerk-w", type=float, default=0.0)
    p.add_argument("--motor-cost-in-reward", action="store_true")

    return p.parse_args()


def _resolve_pretrained_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    expanded = os.path.expanduser(path)
    if not os.path.exists(expanded):
        raise FileNotFoundError(f"Pretrained model not found: {expanded}")
    if not expanded.endswith(".zip"):
        raise ValueError(f"--pretrained must point to a .zip model file, got: {expanded}")
    return expanded


def _auto_load_replay_buffer(model, pretrained_path: str, env_name: str) -> bool:
    if not pretrained_path.endswith(".zip"):
        return False
    base_no_ext = pretrained_path[: -len(".zip")]
    same_dir = os.path.dirname(pretrained_path)
    candidates = [
        f"{base_no_ext}_buffer.pkl",
        os.path.join(same_dir, f"{env_name}_buffer.pkl"),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            try:
                model.load_replay_buffer(cand)
                print(f"Loaded replay buffer from: {cand}")
                return True
            except Exception as e:  # noqa: BLE001
                print(f"Found replay buffer at {cand} but failed to load it: {e}")
                return False
    return False


def load_config(model_class: str, env_id: str) -> dict:
    yaml_filename = f"{model_class}_{env_id}.yaml"
    config_path = os.path.join("hyperparams", yaml_filename)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Hyperparameters config file not found: {config_path}")
    print(f"Reading hyperparameters from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if "replay_buffer_class" in config:
        config["replay_buffer_class"] = HerReplayBuffer
    return config


def main():
    args = parse_args()
    config = {
        "model_class": args.model,
        "env_id": args.env,
        "seed": args.seed,
        "log_dir": args.log_dir,
        "verbose": args.verbose,
    }
    config.update(load_config(args.model, args.env))

    # If we shape reward with motor costs, make HER recompute consistent by copying info dicts
    # into the replay buffer (so compute_reward() can see motor_reward_penalty per transition).
    if bool(args.motor_cost_in_reward) and config.get("replay_buffer_class") is HerReplayBuffer:
        rb_kwargs = dict(config.get("replay_buffer_kwargs") or {})
        rb_kwargs["copy_info_dict"] = True
        config["replay_buffer_kwargs"] = rb_kwargs

    env_name = config["env_id"]
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    base_dir = os.path.join(config["log_dir"], env_name)
    run_dir = os.path.join(base_dir, f"{config['model_class']}_CFC_{timestamp}")
    config.update(
        {
            "checkpoint_dir": os.path.join(run_dir, "checkpoints"),
            "tensorboard_log_dir": os.path.join(base_dir, "tensorboard"),
        }
    )
    for dir_path in [config["checkpoint_dir"], config["tensorboard_log_dir"]]:
        os.makedirs(dir_path, exist_ok=True)

    # Save CLI args + resolved config for reproducibility
    try:
        with open(os.path.join(run_dir, "args.yaml"), "w") as f:
            yaml.safe_dump(_make_yaml_safe(vars(args)), f, sort_keys=False)
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(_make_yaml_safe(config), f, sort_keys=False)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] Failed to write args/config YAML to {run_dir}: {e}")

    gym.register_envs(gymnasium_robotics)
    # We will build a vec env later; first capture some base dims from a single env.
    probe_env = gym.make(config["env_id"])

    # perturbations
    dropout_obj_idx = parse_indices(args.obs_dropout_obj_idx)
    bias_obj_idx = parse_indices(args.obs_bias_obj_idx)
    dropout_keys = tuple([k.strip() for k in args.obs_dropout_keys.split(",") if k.strip()])
    bias_keys = tuple([k.strip() for k in args.obs_bias_keys.split(",") if k.strip()])
    dropout_level = None if args.obs_dropout_p is not None else args.obs_dropout_level
    dropout_p = float(args.obs_dropout_p) if args.obs_dropout_p is not None else 0.0

    probe_env = build_perturbed_env(
        probe_env,
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

    # Capture base dims BEFORE CfCInputWrapper changes them.
    base_obs_dim = int(probe_env.observation_space.spaces["observation"].shape[0])
    act_dim = int(probe_env.action_space.shape[0])
    probe_env.close()

    dt_cfg = DtJitterConfig(
        mode=str(args.dt_jitter_mode),
        dt_min=float(args.dt_min),
        dt_max=float(args.dt_max),
        dt_mean=float(args.dt_mean),
        dt_std=float(args.dt_std),
    )

    def _make_one_env(*, rank: int):
        env_i = gym.make(config["env_id"])
        env_i = build_perturbed_env(
            env_i,
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

        env_i = CfCInputWrapper(
            env_i,
            window_T=int(args.cfc_window),
            include_actions=bool(args.cfc_include_actions),
            k_obs=int(args.obs_delay_k),
            k_act=int(args.action_delay_k),
            dt_jitter=dt_cfg,
            seed=int(args.seed) + int(rank),
        )

        # Motor-friendly cost wrapper (outermost)
        if float(args.motor_energy_w) != 0.0 or float(args.motor_jerk_w) != 0.0 or bool(args.motor_cost_in_reward):
            env_i = MotorCostWrapper(
                env_i,
                cfg=MotorCostConfig(
                    energy_w=float(args.motor_energy_w),
                    jerk_w=float(args.motor_jerk_w),
                    add_to_reward=bool(args.motor_cost_in_reward),
                    include_in_compute_reward=True,
                ),
            )

        env_i.reset(seed=int(config["seed"]) + int(rank))
        env_i.action_space.seed(int(config["seed"]) + int(rank))
        return env_i

    def _env_fn(rank: int):
        return lambda: _make_one_env(rank=rank)

    if int(args.n_envs) <= 1:
        env = _make_one_env(rank=0)
    else:
        n_envs = int(args.n_envs)
        env_fns = [_env_fn(i) for i in range(n_envs)]
        if args.vec_env == "subproc":
            env = SubprocVecEnv(env_fns)
        else:
            env = DummyVecEnv(env_fns)

    checkpoint_callback = CheckpointCallback(save_freq=config["checkpoint_freq"], save_path=config["checkpoint_dir"])
    eval_callback = EvalCallback(env, best_model_save_path=run_dir, log_path=run_dir, eval_freq=config["eval_freq"])
    callback = CallbackList([checkpoint_callback, eval_callback])

    model_class = {"DDPG": DDPG, "TD3": TD3, "SAC": SAC}[config["model_class"]]

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=config["action_noise_sigma"] * np.ones(n_actions))

    # Inject CfC feature extractor into policy kwargs
    policy_kwargs = dict(config.get("policy_kwargs") or {})
    policy_kwargs.update(
        {
            "share_features_extractor": True,
            "features_extractor_class": CfCFeatureExtractor,
            "features_extractor_kwargs": {
                "window_T": int(args.cfc_window),
                "base_obs_dim": int(base_obs_dim),
                "action_dim": int(act_dim),
                "include_actions": bool(args.cfc_include_actions),
                "cfc_units": int(args.cfc_units),
                "cfc_mixed_memory": bool(args.cfc_mixed_memory),
                "cfc_mode": str(args.cfc_mode),
                "goal_embed_dim": int(args.goal_embed_dim),
            },
        }
    )
    config["policy_kwargs"] = policy_kwargs

    pretrained_path = _resolve_pretrained_path(args.pretrained)
    replay_buffer_loaded = False

    if pretrained_path is None:
        model = model_class(
            policy=config["policy"],
            env=env,
            buffer_size=config["buffer_size"],
            batch_size=config["batch_size"],
            gamma=config["gamma"],
            tau=config["tau"],
            learning_rate=config["learning_rate"],
            replay_buffer_class=config.get("replay_buffer_class"),
            replay_buffer_kwargs=config.get("replay_buffer_kwargs"),
            verbose=config["verbose"],
            action_noise=action_noise,
            tensorboard_log=config["tensorboard_log_dir"],
            policy_kwargs=config["policy_kwargs"],
            seed=config["seed"],
        )
    else:
        print(f"Loading pretrained model from: {pretrained_path}")
        model = model_class.load(
            pretrained_path,
            env=env,
            verbose=config["verbose"],
            tensorboard_log=config["tensorboard_log_dir"],
            action_noise=action_noise,
            seed=config["seed"],
        )
        if not args.no_load_replay_buffer:
            replay_buffer_loaded = _auto_load_replay_buffer(model, pretrained_path, env_name)

        if not replay_buffer_loaded and isinstance(getattr(model, "replay_buffer", None), HerReplayBuffer):
            max_episode_steps = None
            if getattr(env, "spec", None) is not None:
                max_episode_steps = getattr(env.spec, "max_episode_steps", None)
            if max_episode_steps is None:
                max_episode_steps = getattr(env, "_max_episode_steps", None)
            if max_episode_steps is not None:
                min_learning_starts = int(max_episode_steps) + 1
                if getattr(model, "learning_starts", 0) < min_learning_starts:
                    print(
                        f"Adjusting learning_starts from {model.learning_starts} to {min_learning_starts} "
                        f"(HER requires at least one full episode before sampling)."
                    )
                    model.learning_starts = min_learning_starts

    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callback,
            reset_num_timesteps=(pretrained_path is None) or (not replay_buffer_loaded),
        )
        print("\nTraining completed. Saving model...")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model...")
    finally:
        model_path = os.path.join(run_dir, f"{env_name}")
        model.save(model_path)
        model.save_replay_buffer(f"{model_path}_buffer")
        print(f"Model and replay buffer saved to: {run_dir}")
        env.close()


if __name__ == "__main__":
    main()


