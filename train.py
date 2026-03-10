"""Training script for robotic manipulation tasks using RL algorithms (current: SAC)."""

import os
import sys
import json
import argparse
from functools import partial
from datetime import datetime
from pathlib import Path
import yaml
from typing import Optional

import gymnasium as gym
import gymnasium_robotics
try:
    import panda_gym
except ImportError:
    pass
import numpy as np

import multiprocessing as mp
from stable_baselines3 import HerReplayBuffer, DDPG, TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor  # Added for paper logging
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from perturbation_wrappers import build_perturbed_env, parse_indices
try:
    from delay_compensation import HistoryObsActWrapper, DelayAwareTransformerExtractor
except ImportError:
    pass

from conservative_wrappers import MotorCostConfig, MotorCostWrapper
from residual_mem_trainer import train_pretrained_sac_with_residual_mem
from residual_mem_modules import DisturbanceCurriculum
from adversary_controller import BudgetedAdversary, AdversaryConfig
from experiment_logger import PaperLogger, PaperExperimentCallback # New logger


class AdversarialTrainingCallback(BaseCallback):
    """
    Orchestrates the budgeted adversary loop:
    1. Sample disturbance (obs delay, act delay) at episode start.
    2. Apply to environment (via DisturbanceManagerWrapper).
    3. Store episode outcome (return, cost) in adversary buffer.
    4. Update adversary periodically.
    """

    def __init__(self, adversary: BudgetedAdversary, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.adversary = adversary
        self.env_disturbances = {}  # Map env_idx -> current disturbance dict

    def _on_training_start(self) -> None:
        # Initialize disturbances for all envs
        n_envs = self.training_env.num_envs
        for i in range(n_envs):
            self._sample_and_apply(i, is_start=True)

    def _sample_and_apply(self, env_idx: int, is_start: bool = False):
        # Sample from adversary
        k_obs, k_act, lp_obs, lp_act, stage = self.adversary.sample_disturbance(self.num_timesteps)

        # Store for later retrieval when episode ends
        self.env_disturbances[env_idx] = {
            "k_obs": k_obs,
            "k_act": k_act,
            "lp_obs": lp_obs,
            "lp_act": lp_act,
            "stage": stage,
        }

        # Apply to environment via wrapper
        spec = {"k_obs": int(k_obs), "k_act": int(k_act)}
        # Use env_method to reach the DisturbanceManagerWrapper inside VecEnv
        self.training_env.env_method("set_disturbance", indices=[env_idx], **spec)

        if self.verbose > 1 or (self.verbose > 0 and (is_start or self.num_timesteps % 10000 == 0)):
            print(f"[adversary] env={env_idx} stage={stage} set k_obs={k_obs} k_act={k_act}", flush=True)

    def _on_step(self) -> bool:
        # Check for dones to process episode results
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for i, done in enumerate(dones):
            if done:
                # Episode finished
                
                # Retrieve the disturbance used for this completed episode
                dist = self.env_disturbances.get(i)
                if dist:
                    # Inject into info so PaperExperimentCallback can log it to CSV
                    infos[i]["adv_stage"] = dist["stage"]
                    infos[i]["adv_cost"] = dist["k_obs"] + self.adversary.cfg.w_act * dist["k_act"] # Recompute or store? Storing is better but cost logic is consistent.
                    infos[i]["adv_lambda"] = self.adversary.cfg.adv_lambda
                    
                    # Store in adversary buffer
                    ep_ret = infos[i].get("episode", {}).get("r", 0.0)
                    self.adversary.store_episode(
                        log_prob_obs=dist["lp_obs"],
                        log_prob_act=dist["lp_act"],
                        ep_return=ep_ret,
                        k_obs=dist["k_obs"],
                        k_act=dist["k_act"],
                    )
                    
                    # Log to TensorBoard
                    self.logger.record("adversary/k_obs", dist["k_obs"])
                    self.logger.record("adversary/k_act", dist["k_act"])
                    self.logger.record("adversary/stage", dist["stage"])
                    self.logger.record("adversary/episode_return", ep_ret)

                # Sample new disturbance for the NEXT episode
                self._sample_and_apply(i)
                
                # Update adversary if buffer is full (handled by batch_size)
                adv_infos = self.adversary.update()
                for k, v in adv_infos.items():
                    self.logger.record(f"adversary/{k}", v)

        return True


class DisturbanceCurriculumCallback(BaseCallback):
    """
    Apply disturbance curriculum during standard SB3 training (baseline path).
    
    This updates wrappers at runtime via env.set_disturbance(...) exposed by DisturbanceManagerWrapper.
    """

    def __init__(self, *, curriculum: DisturbanceCurriculum, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.curriculum = curriculum
        self._last_stage: Optional[str] = None

    def _apply(self, step: int) -> None:
        ds = self.curriculum.spec(int(step))
        if ds.name == self._last_stage:
            return

        spec: dict = {"k_obs": int(ds.k_obs), "k_act": int(ds.k_act), "bias_b": float(ds.bias_b)}
        if ds.dropout_level is not None:
            spec["dropout_level"] = str(ds.dropout_level)
        elif ds.dropout_p is not None:
            spec["dropout_p"] = float(ds.dropout_p)

        # VecEnv path
        if hasattr(self.training_env, "env_method"):
            self.training_env.env_method("set_disturbance", **spec)
        else:
            # Fallback (should not happen with SB3)
            if hasattr(self.training_env, "set_disturbance"):
                self.training_env.set_disturbance(**spec)

        self._last_stage = ds.name
        if self.verbose:
            print(f"[curriculum] step={int(step)} stage={ds.name} spec={spec}", flush=True)

    def _on_training_start(self) -> None:
        # Ensure we start from the curriculum's "none" stage, even if CLI args specify max disturbances.
        self._apply(int(self.num_timesteps))

    def _on_step(self) -> bool:
        self._apply(int(self.num_timesteps))
        return True


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


def _infer_max_episode_steps(vec_env) -> Optional[int]:
    """
    Try to infer max_episode_steps for both Gym env and SB3 VecEnv.
    Needed for HER warmup (cannot sample until at least one episode is finished).
    """
    # VecEnv path
    if hasattr(vec_env, "get_attr"):
        try:
            specs = vec_env.get_attr("spec", indices=0)
            if specs and specs[0] is not None:
                m = getattr(specs[0], "max_episode_steps", None)
                if m is not None:
                    return int(m)
        except Exception:
            pass
        try:
            vals = vec_env.get_attr("_max_episode_steps", indices=0)
            if vals and vals[0] is not None:
                return int(vals[0])
        except Exception:
            pass

    # Plain env path
    try:
        if getattr(vec_env, "spec", None) is not None:
            m = getattr(vec_env.spec, "max_episode_steps", None)
            if m is not None:
                return int(m)
    except Exception:
        pass
    try:
        m = getattr(vec_env, "_max_episode_steps", None)
        if m is not None:
            return int(m)
    except Exception:
        pass
    return None


# load configuration from YAML file
def load_config(config_base: dict):
    yaml_filename = f"{config_base['model_class']}_{config_base['env_id']}.yaml"
    config_path = os.path.join("hyperparams", yaml_filename)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Hyperparameters config file not found: {config_path}")

    print(f"Reading hyperparameters from: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # convert replay_buffer_class from string to actual class
    if "replay_buffer_class" in config:
        config["replay_buffer_class"] = HerReplayBuffer
    return config


# argument parser for flexibility
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RL agent for Gymnasium-Robotics manipulation tasks (default: FetchPickAndPlace-v4)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="SAC",
        choices=["DDPG", "TD3", "SAC"],
        help="RL model to use (DDPG, TD3, SAC)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="FetchPickAndPlace-v4",
        help="Gymnasium environment ID",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument(
        "--log_dir", type=str, default="./logs", help="Base directory for logs"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Optional override for total training timesteps (overrides hyperparams YAML).",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level (0: no output, 1: info, 2: debug)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help=(
            "Path to a Stable-Baselines3 .zip model to continue training from. "
            "Example: logs/FetchPickAndPlace-v4/SAC_250204_065306/best_model.zip"
        ),
    )
    parser.add_argument(
        "--no-load-replay-buffer",
        action="store_true",
        help=(
            "When resuming from --pretrained, do NOT attempt to auto-load a replay buffer "
            "from the same folder (if present)."
        ),
    )

    # --- output organization ---
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional label to include in the run directory name (useful for baseline/ablation bookkeeping).",
    )
    parser.add_argument(
        "--bundle-outputs",
        action="store_true",
        help="If set, place tensorboard logs under the run directory (run_dir/tensorboard) so all outputs are self-contained.",
    )

    # --- vec env / throughput ---
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environments (n_envs). Use >1 to speed up sampling.",
    )
    parser.add_argument(
        "--vec-env",
        type=str,
        default="dummy",
        choices=["dummy", "subproc"],
        help="Vectorized env backend. 'subproc' uses multiple processes (faster sampling, more overhead).",
    )

    # --- robustness / domain perturbations ---
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
        default="observation",
        help=(
            "Comma-separated dict observation keys to perturb (train-safe default: observation). "
            "Example for evaluation-like behavior: observation,achieved_goal"
        ),
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
        help="Per-episode constant bias magnitude b (Uniform([-b,b])) in meters for object-related dims.",
    )
    parser.add_argument(
        "--obs-bias-keys",
        type=str,
        default="observation",
        help=(
            "Comma-separated dict observation keys to bias (train-safe default: observation). "
            "Example for evaluation-like behavior: observation,achieved_goal"
        ),
    )
    parser.add_argument(
        "--obs-bias-obj-idx",
        type=str,
        default=None,
        help="Comma-separated indices into obs['observation'] to bias (e.g. '3,4,5'). Default: auto [3,4,5] for Fetch.",
    )

    # --- pretrained SAC + residual memory controller (GRU/CfC) + Fourier + dual curriculum ---
    parser.add_argument(
        "--use-residual-mem",
        action="store_true",
        help="Enable: Pretrained SAC (base actor frozen) + residual memory controller (GRU/CfC) trained off-policy with sequence replay + burn-in.",
    )
    parser.add_argument(
        "--residual-mem-type",
        type=str,
        default="gru",
        choices=["gru", "cfc"],
        help="Residual memory type (default: gru). 'cfc' requires ncps.",
    )
    parser.add_argument("--residual-hidden", type=int, default=128, help="Residual memory hidden size.")
    parser.add_argument(
        "--residual-obj-idx",
        type=str,
        default=None,
        help="Comma-separated indices into obs['observation'] to Fourier-expand for the residual input (default: 3,4,5).",
    )
    parser.add_argument("--fourier-bands", type=int, default=8, help="Fourier features: number of frequency bands.")
    parser.add_argument("--fourier-max-freq", type=float, default=10.0, help="Fourier features: max frequency.")
    parser.add_argument(
        "--base-actor-unfreeze-last-layer",
        action="store_true",
        help="Optional: unfreeze the base SAC actor last heads (mu/log_std) while keeping the rest frozen.",
    )
    parser.add_argument("--alpha-start", type=float, default=0.0, help="Residual scale alpha schedule: start value.")
    parser.add_argument("--alpha-mid", type=float, default=0.2, help="Residual scale alpha schedule: mid value.")
    parser.add_argument("--alpha-final", type=float, default=1.0, help="Residual scale alpha schedule: final value.")
    parser.add_argument("--alpha-step-mid", type=int, default=100_000, help="Step when alpha reaches alpha-mid.")
    parser.add_argument("--alpha-step-final", type=int, default=300_000, help="Step when alpha reaches alpha-final.")
    parser.add_argument("--seq-len", type=int, default=20, help="Sequence length T used for training recurrent residual.")
    parser.add_argument("--burn-in", type=int, default=5, help="Burn-in steps B used to warm up hidden state (no loss).")
    parser.add_argument(
        "--rm-batch-size",
        type=int,
        default=None,
        help="Residual-mem training batch size override (default: use hyperparams YAML batch_size).",
    )
    parser.add_argument(
        "--rm-train-freq",
        type=int,
        default=50,
        help="Residual-mem: train every N environment steps (default: 50).",
    )
    parser.add_argument(
        "--rm-gradient-steps",
        type=int,
        default=5,
        help="Residual-mem: number of gradient steps each time we train (default: 5).",
    )
    parser.add_argument(
        "--use-disturbance-curriculum",
        action="store_true",
        help="Enable disturbance curriculum schedule (none->easy->med->hard) during training. Overrides static disturbance args.",
    )
    parser.add_argument("--curriculum-step-easy", type=int, default=200_000, help="Step to switch from none->easy.")
    parser.add_argument("--curriculum-step-med", type=int, default=400_000, help="Step to switch from easy->med.")
    parser.add_argument("--curriculum-step-hard", type=int, default=600_000, help="Step to switch from med->hard.")

    # --- Adversarial Training (Version B) ---
    parser.add_argument("--use-adversary", action="store_true", help="Enable Budgeted Adversary for episode-level disturbance selection.")
    parser.add_argument("--adv-lambda", type=float, default=0.5, help="Adversary budget penalty lambda.")
    parser.add_argument("--adv-w-act", type=float, default=2.0, help="Adversary cost weight for action delay.")
    parser.add_argument("--adv-update-every", type=int, default=10, help="Update adversary every N episodes.")
    parser.add_argument("--adv-stage-steps", type=int, nargs=2, default=[200000, 400000], help="Steps to switch adversary stages (2 ints).")
    parser.add_argument("--adv-k-obs-max", type=int, default=3, help="Max obs delay for adversary.")
    parser.add_argument("--adv-k-act-max", type=int, default=1, help="Max action delay for adversary.")
    parser.add_argument("--adv-lr", type=float, default=0.05, help="Adversary learning rate.")

    # --- delay compensation modules (Transformer + optional prediction head) ---
    parser.add_argument(
        "--history-k",
        type=int,
        default=0,
        help="History length K for delay-aware input. If >0, obs['observation'] is augmented with (K+1) obs + K past actions and encoded by a Transformer.",
    )
    parser.add_argument(
        "--tf-d-model",
        type=int,
        default=128,
        help="Transformer model dimension (d_model) for delay-aware encoder.",
    )
    parser.add_argument(
        "--tf-nhead",
        type=int,
        default=4,
        help="Transformer attention heads.",
    )
    parser.add_argument(
        "--tf-num-layers",
        type=int,
        default=2,
        help="Transformer encoder layers.",
    )
    parser.add_argument(
        "--tf-dropout",
        type=float,
        default=0.1,
        help="Transformer dropout.",
    )
    parser.add_argument(
        "--tf-pool",
        type=str,
        default="last",
        choices=["last", "mean"],
        help="Pooling over tokens: use last token or mean pooling.",
    )
    parser.add_argument(
        "--goal-embed-dim",
        type=int,
        default=64,
        help="MLP embedding dim for desired_goal before concatenation.",
    )
    parser.add_argument(
        "--use-pred-head",
        action="store_true",
        help="Enable Module B: add a small prediction head on top of the Transformer latent, and feed the predicted compact state to SAC (instead of the raw Transformer latent).",
    )
    parser.add_argument(
        "--pred-obs-idx",
        type=str,
        default=None,
        help="Comma-separated indices (into base obs['observation'] vector) that the prediction head should output. Default: predict the full base observation vector.",
    )
    parser.add_argument(
        "--pred-mode",
        type=str,
        default="delta",
        choices=["delta", "direct"],
        help="Prediction head output mode. 'delta' adds a learned correction to the latest observed state (more like an observer/compensator). 'direct' outputs the state directly.",
    )
    parser.add_argument(
        "--pred-append-transformer",
        action="store_true",
        help="If set, concatenate (pred_state, transformer_latent, goal_embed) as SAC features. Default: (pred_state, goal_embed).",
    )

    # --- motor-friendly cost (Module D) ---
    parser.add_argument(
        "--motor-energy-w",
        type=float,
        default=0.0,
        help="Weight for energy proxy ||a||^2. Used for logging and (optionally) reward shaping.",
    )
    parser.add_argument(
        "--motor-jerk-w",
        type=float,
        default=0.0,
        help="Weight for jerk proxy ||a_t - a_{t-1}||^2. Used for logging and (optionally) reward shaping.",
    )
    parser.add_argument(
        "--motor-cost-in-reward",
        action="store_true",
        help="If set, subtract motor cost from env reward (reward shaping). Stored in info so HER can recompute consistently.",
    )

    # --- Paper Automation ---
    parser.add_argument("--eval-freq-paper", type=int, default=50000, help="Frequency of fixed-setting evaluation for paper (steps).")
    parser.add_argument("--n-eval-paper", type=int, default=5, help="Number of episodes per setting for periodic paper eval.")
    parser.add_argument(
        "--paper-eval-settings", 
        type=str, 
        default="Nominal,ObsOnly3,ActOnly1,MixedHard",
        help="Comma-separated list of fixed settings to evaluate periodically."
    )
    
    # --- Project Standardization Arguments ---
    parser.add_argument("--out-root", type=str, default="runs", help="Root directory for run outputs")
    parser.add_argument("--fig-root", type=str, default="figures", help="Root directory for figure outputs")
    parser.add_argument("--run-id", type=str, default=None, help="Unique identifier for the run")
    parser.add_argument("--method-tag", type=str, default="baseline", help="Tag for the method/experiment type")

    return parser.parse_args()


def make_train_env(*, rank: int, args, config: dict):
    """
    Factory for creating one env instance, applying wrappers consistently.
    Needed for SubprocVecEnv/DummyVecEnv when n_envs>1.
    """
    env_i = gym.make(config["env_id"])

    # Apply perturbation wrappers (optional; defaults are no-ops)
    dropout_obj_idx = parse_indices(args.obs_dropout_obj_idx)
    bias_obj_idx = parse_indices(args.obs_bias_obj_idx)
    dropout_keys = tuple([k.strip() for k in args.obs_dropout_keys.split(",") if k.strip()])
    bias_keys = tuple([k.strip() for k in args.obs_bias_keys.split(",") if k.strip()])

    dropout_level = None if args.obs_dropout_p is not None else args.obs_dropout_level
    dropout_p = float(args.obs_dropout_p) if args.obs_dropout_p is not None else 0.0

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
        force_wrappers=bool(getattr(args, "use_disturbance_curriculum", False))
        or bool(getattr(args, "use_residual_mem", False))
        or bool(getattr(args, "use_adversary", False)),
        # Always expose env.set_disturbance/get_disturbance when using residual-mem,
        # so SubprocVecEnv env_method calls never crash even if curriculum is OFF.
        enable_disturbance_control=bool(getattr(args, "use_disturbance_curriculum", False))
        or bool(getattr(args, "use_residual_mem", False))
        or bool(getattr(args, "use_adversary", False)),
    )

    # Delay-aware history wrapper must be OUTERMOST (after all perturbations)
    if args.history_k and int(args.history_k) > 0:
        env_i = HistoryObsActWrapper(env_i, history_k=int(args.history_k))

    # Motor-friendly cost wrapper (outermost; does not change obs keys)
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

    # Monitor wrapper for SB3 info['episode'] stats
    env_i = Monitor(env_i)

    # Seed each env deterministically but differently
    env_i.reset(seed=int(config["seed"]) + int(rank))
    env_i.action_space.seed(int(config["seed"]) + int(rank))
    return env_i


def make_env_fn(rank: int, *, args, config: dict):
    return partial(make_train_env, rank=rank, args=args, config=config)


def make_eval_env_paper(setting_id: str, seed: int, *, args, config: dict):
    # Determine specs
    k_obs, k_act = 0, 0
    if setting_id == "Nominal":
        pass
    elif setting_id == "ObsOnly3":
        k_obs = 3
    elif setting_id == "ActOnly1":
        k_act = 1
    elif setting_id == "MixedHard":
        k_obs, k_act = 3, 1
    elif setting_id == "MixedLight":
        k_obs, k_act = 2, 1
    else:
        print(f"[Warn] Unknown paper eval setting '{setting_id}', using Nominal.")

    e = gym.make(config["env_id"])

    # Apply perturbations (Fixed Setting)
    e = build_perturbed_env(
        e,
        dropout_level="none",
        obs_delay_k=k_obs,
        action_delay_k=k_act,
        force_wrappers=True,
        enable_disturbance_control=True,
    )

    # History Wrapper
    if args.history_k and int(args.history_k) > 0:
        e = HistoryObsActWrapper(e, history_k=int(args.history_k))

    e.reset(seed=seed)
    return e


def auto_load_replay_buffer(model, pretrained_path: str, env_name: str) -> bool:
    """
    Try to locate and load a replay buffer saved next to a pretrained model.

    We support common naming patterns:
    - <model_path_without_.zip>_buffer.pkl
    - <same_dir>/<env_name>_buffer.pkl  (this repo's train.py saves this pattern)
    """
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
            except Exception as e:  # noqa: BLE001 - best-effort load
                print(f"Found replay buffer at {cand} but failed to load it: {e}")
                return False
    return False


def resolve_pretrained_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    expanded = os.path.expanduser(path)
    if not os.path.exists(expanded):
        raise FileNotFoundError(f"Pretrained model not found: {expanded}")
    if not expanded.endswith(".zip"):
        raise ValueError(f"--pretrained must point to a .zip model file, got: {expanded}")
    return expanded


def fixed_lr_schedule(_: float, *, lr: float) -> float:
    return float(lr)


def build_vec_env(*, args, config: dict):
    n_envs = int(args.n_envs)
    env_fns = [make_env_fn(i, args=args, config=config) for i in range(n_envs)]
    if args.vec_env == "subproc" and n_envs > 1:
        # On Linux, prefer fork to avoid spawn/forkserver "safe importing of main module" constraints
        # when running a script that executes at top level.
        start_method = "fork" if "fork" in mp.get_all_start_methods() else None
        return SubprocVecEnv(env_fns, start_method=start_method)
    return DummyVecEnv(env_fns)


def build_callbacks(*, args, config: dict, env, run_dir: Path, csv_dir: Path, run_id: str):
    checkpoint_callback = CheckpointCallback(
        save_freq=config["checkpoint_freq"], save_path=config["checkpoint_dir"]
    )
    eval_callback = EvalCallback(
        env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=config["eval_freq"],
    )
    cb_list = [checkpoint_callback, eval_callback]

    paper_logger = PaperLogger(log_dir=str(csv_dir), run_id=run_id, config=config)
    paper_cb = PaperExperimentCallback(
        logger=paper_logger,
        eval_freq=args.eval_freq_paper,
        n_eval_episodes=args.n_eval_paper,
        eval_settings=args.paper_eval_settings.split(","),
        make_eval_env_fn=partial(make_eval_env_paper, args=args, config=config),
        train_seed=config["seed"],
        method=args.run_name if args.run_name else "default",
        scenario_tag="train_dist",
        verbose=1,
    )
    cb_list.append(paper_cb)

    if getattr(args, "use_adversary", False):
        print(f"[adversary] Enabled Budgeted Adversary (update_every={args.adv_update_every})")
        adv_config = AdversaryConfig(
            enabled=True,
            adv_lambda=float(args.adv_lambda),
            w_act=float(args.adv_w_act),
            update_every=int(args.adv_update_every),
            batch_size=int(args.adv_update_every),  # Map update_every to batch_size for REINFORCE
            stage_steps=args.adv_stage_steps,
            k_obs_max=int(args.adv_k_obs_max),
            k_act_max=int(args.adv_k_act_max),
            lr=float(args.adv_lr),
        )
        adversary = BudgetedAdversary(adv_config)
        cb_list.insert(0, AdversarialTrainingCallback(adversary, verbose=1))

    if bool(getattr(args, "use_disturbance_curriculum", False)):
        curr = DisturbanceCurriculum(
            step_easy=int(args.curriculum_step_easy),
            step_med=int(args.curriculum_step_med),
            step_hard=int(args.curriculum_step_hard),
            k_obs_max=int(args.obs_delay_k),
            k_act_max=int(args.action_delay_k),
            bias_b_max=float(args.obs_bias_b),
            use_dropout_p=bool(args.obs_dropout_p is not None),
            dropout_p_max=float(args.obs_dropout_p or 0.0),
        )
        cb_list.insert(0, DisturbanceCurriculumCallback(curriculum=curr, verbose=1))

    return CallbackList(cb_list)


def build_or_load_model(*, args, config: dict, env, env_name: str):
    model_class = {"DDPG": DDPG, "TD3": TD3, "SAC": SAC}[config["model_class"]]

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=config["action_noise_sigma"] * np.ones(n_actions)
    )

    pretrained_path = resolve_pretrained_path(args.pretrained)
    replay_buffer_loaded = False

    # If we enabled history_k, inject the Transformer feature extractor into policy_kwargs.
    if args.history_k and int(args.history_k) > 0:
        pred_idx = parse_indices(args.pred_obs_idx)
        policy_kwargs = dict(config.get("policy_kwargs") or {})
        policy_kwargs.update(
            {
                "share_features_extractor": True,
                "features_extractor_class": DelayAwareTransformerExtractor,
                "features_extractor_kwargs": {
                    "history_k": int(args.history_k),
                    "action_dim": int(env.action_space.shape[-1]),
                    "k_obs": int(args.obs_delay_k),
                    "k_act": int(args.action_delay_k),
                    "include_k": True,
                    "d_model": int(args.tf_d_model),
                    "nhead": int(args.tf_nhead),
                    "num_layers": int(args.tf_num_layers),
                    "dropout": float(args.tf_dropout),
                    "goal_embed_dim": int(args.goal_embed_dim),
                    "use_pred_head": bool(args.use_pred_head),
                    "pred_obs_indices": pred_idx,
                    "pred_mode": str(args.pred_mode),
                    "pred_append_transformer_latent": bool(args.pred_append_transformer),
                    "pool": str(args.tf_pool),
                },
            }
        )
        config["policy_kwargs"] = policy_kwargs

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
            learning_starts=config.get("learning_starts", 100),
        )
        return model, pretrained_path, replay_buffer_loaded

    print(f"Loading pretrained model from: {pretrained_path}")
    model = model_class.load(
        pretrained_path,
        env=env,
        verbose=config["verbose"],
        tensorboard_log=config["tensorboard_log_dir"],
        action_noise=action_noise,
        seed=config["seed"],
        custom_objects={
            "lr_schedule": partial(fixed_lr_schedule, lr=float(config["learning_rate"])),
            "learning_rate": float(config["learning_rate"]),
        },
    )
    if not args.no_load_replay_buffer:
        replay_buffer_loaded = auto_load_replay_buffer(model, pretrained_path, env_name)

    # If we did not (or could not) load a replay buffer, we must "warm up" before training.
    # Otherwise, the loaded model may have a large num_timesteps and start training immediately,
    # while the HER buffer cannot sample before at least one full episode is collected.
    if not replay_buffer_loaded and isinstance(getattr(model, "replay_buffer", None), HerReplayBuffer):
        max_episode_steps = _infer_max_episode_steps(env)
        n_envs = int(getattr(env, "num_envs", 1))
        if max_episode_steps is not None:
            # SB3 counts num_timesteps as total transitions across all envs.
            # To guarantee at least one full episode finished before sampling with HER,
            # we need learning_starts >= (max_episode_steps + 1) * n_envs.
            min_learning_starts = (int(max_episode_steps) + 1) * max(1, n_envs)
            if getattr(model, "learning_starts", 0) < min_learning_starts:
                print(
                    f"Adjusting learning_starts from {model.learning_starts} to {min_learning_starts} "
                    f"(HER requires at least one full episode before sampling)."
                )
                model.learning_starts = min_learning_starts

    return model, pretrained_path, replay_buffer_loaded


def main() -> None:
    args = parse_args()

    CONFIG = {
        "model_class": args.model,
        "env_id": args.env,
        "seed": args.seed,
        "log_dir": args.log_dir,
        "verbose": args.verbose,
    }

    # update CONFIG with the loaded config
    CONFIG.update(load_config(CONFIG))

    # Optional CLI override for total timesteps (useful for smoke tests/ablations)
    if getattr(args, "total_timesteps", None) is not None:
        CONFIG["total_timesteps"] = int(args.total_timesteps)

    # If we shape reward with motor costs, make HER recompute consistent by copying info dicts
    # into the replay buffer (so compute_reward() can see motor_reward_penalty per transition).
    if bool(getattr(args, "motor_cost_in_reward", False)) and CONFIG.get("replay_buffer_class") is HerReplayBuffer:
        rb_kwargs = dict(CONFIG.get("replay_buffer_kwargs") or {})
        rb_kwargs["copy_info_dict"] = True
        CONFIG["replay_buffer_kwargs"] = rb_kwargs

    # organizing logs
    env_name = CONFIG["env_id"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- 1. Run ID & Directory Setup (Standardized) ---
    if args.run_id:
        run_id = args.run_id
    else:
        # Default format: <method>__<env>__seed<seed>__<timestamp>
        clean_env = args.env.replace("Fetch", "").replace("PickAndPlace", "PnP").replace("-v4", "")
        run_id = f"{args.method_tag}__{clean_env}__seed{args.seed}__{timestamp}"

    # Define paths using pathlib
    out_root = Path(args.out_root)
    run_dir = out_root / run_id

    # Subdirectories
    ckpt_dir = run_dir / "checkpoints"
    tb_dir = run_dir / "tb"
    csv_dir = run_dir / "csv"

    # Create directories
    for d in [run_dir, ckpt_dir, tb_dir, csv_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[Info] Run directory initialized at: {run_dir}")

    # --- 2. Save Config ---
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        config_data = vars(args)
        config_data["_python_version"] = sys.version
        config_data["_timestamp"] = timestamp
        json.dump(config_data, f, indent=4, sort_keys=True)

    # Update CONFIG for compatibility
    CONFIG["log_dir"] = str(run_dir)
    CONFIG["checkpoint_dir"] = str(ckpt_dir)
    CONFIG["tensorboard_log_dir"] = str(tb_dir)

    print(f"[run] run_dir={run_dir}")
    print(f"[run] tensorboard_log_dir={tb_dir}")
    print(f"[debug] CONFIG['learning_starts'] = {CONFIG.get('learning_starts')}")



    # Save CLI args + resolved config for reproducibility
    try:
        with open(os.path.join(run_dir, "args.yaml"), "w") as f:
            yaml.safe_dump(_make_yaml_safe(vars(args)), f, sort_keys=False)
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(_make_yaml_safe(CONFIG), f, sort_keys=False)
    except Exception as e:  # noqa: BLE001 - best-effort logging
        print(f"[warn] Failed to write args/config YAML to {run_dir}: {e}")

    # environment setup
    gym.register_envs(gymnasium_robotics)

    env = build_vec_env(args=args, config=CONFIG)

    # --- residual-mem training path (early exit) ---
    if bool(getattr(args, "use_residual_mem", False)):
        # This path runs its own training loop (sequence replay + burn-in) and saves artifacts into run_dir.
        train_pretrained_sac_with_residual_mem(args, CONFIG, env, run_dir)
        env.close()
        raise SystemExit(0)

    callback = build_callbacks(
        args=args, config=CONFIG, env=env, run_dir=run_dir, csv_dir=csv_dir, run_id=run_id
    )
    model, pretrained_path, replay_buffer_loaded = build_or_load_model(
        args=args, config=CONFIG, env=env, env_name=env_name
    )

    # training loop
    try:
        model.learn(
            total_timesteps=CONFIG["total_timesteps"],
            callback=callback,
            # If we resume from weights but have no replay buffer, reset timesteps so SB3
            # will collect enough transitions before starting gradient updates.
            reset_num_timesteps=(pretrained_path is None) or (not replay_buffer_loaded),
        )
        print("\nTraining completed. Saving model...")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model...")

    finally:
        # saving model and replay buffer
        final_model_path = ckpt_dir / "last_model.zip"
        model.save(str(final_model_path))

        # Optional: also save replay buffer if needed (but it's large)
        # model.save_replay_buffer(str(ckpt_dir / "last_model_buffer"))

        print(f"Model saved to: {final_model_path}")

        # cleanup
        env.close()


if __name__ == "__main__":
    main()
