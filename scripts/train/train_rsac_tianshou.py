#!/usr/bin/env python3
"""
Train recurrent SAC (LSTM actor/critic) with Tianshou on delayed environment only.

Outputs:
- runs/<run_id>/config.json
- runs/<run_id>/checkpoints/policy_latest.pth
- runs/<run_id>/logs/train_log.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make repo root importable when running as:
# python scripts/train/train_rsac_tianshou.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from perturbation_wrappers import build_perturbed_env  # noqa: E402

try:
    from tianshou.data import Batch, VectorReplayBuffer
    from tianshou.env import DummyVectorEnv, SubprocVectorEnv
    from tianshou.policy import SACPolicy
    from tianshou.utils.net.continuous import ActorProb, Critic
    from tianshou.data import Collector
except Exception as exc:  # noqa: BLE001
    raise ImportError(
        "This script requires tianshou. Please install it first, e.g. `pip install tianshou`."
    ) from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train recurrent SAC (LSTM) with Tianshou.")
    p.add_argument("--env", type=str, default="FetchPickAndPlace-v4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--out-root", type=str, default="runs")

    # Fixed-delay training line (defaults match requested setup).
    p.add_argument("--obs-delay-k", type=int, default=3)
    p.add_argument("--action-delay-k", type=int, default=1)

    # Sequence/recurrent training knobs (explicitly surfaced).
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--burn-in", type=int, default=8)

    # Throughput / training knobs.
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--vec-env", type=str, default="subproc", choices=["dummy", "subproc"])
    p.add_argument("--buffer-size", type=int, default=300_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--start-timesteps", type=int, default=20_000)
    p.add_argument("--step-per-collect", type=int, default=1000)
    p.add_argument("--update-per-step", type=float, default=1.0)
    p.add_argument("--save-interval", type=int, default=50_000)
    p.add_argument("--log-interval", type=int, default=10_000)
    p.add_argument(
        "--eval-interval",
        type=int,
        default=20_000,
        help="Run periodic deterministic eval every N env steps.",
    )
    p.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Episodes per periodic eval for success-rate printing.",
    )

    # SAC / net.
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--alpha-lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--lstm-hidden-size", type=int, default=256)
    p.add_argument("--lstm-layers", type=int, default=1)
    p.add_argument("--auto-alpha", action="store_true", default=True)
    p.add_argument("--alpha", type=float, default=0.2, help="Used only when --no-auto-alpha.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--no-auto-alpha", dest="auto_alpha", action="store_false")
    return p.parse_args()


@dataclass
class RunPaths:
    run_dir: Path
    checkpoint_dir: Path
    log_dir: Path
    docs_dir: Path
    config_json: Path
    latest_ckpt: Path
    train_log_jsonl: Path


def setup_run_dir(args: argparse.Namespace) -> tuple[str, RunPaths]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_id:
        run_id = args.run_id
    else:
        run_id = f"rsac_delay_only__seed{args.seed}__{ts}"

    run_dir = Path(args.out_root).expanduser().resolve() / run_id
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    docs_dir = run_dir / "docs"
    for d in (run_dir, checkpoint_dir, log_dir, docs_dir):
        d.mkdir(parents=True, exist_ok=True)

    paths = RunPaths(
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        docs_dir=docs_dir,
        config_json=run_dir / "config.json",
        latest_ckpt=checkpoint_dir / "policy_latest.pth",
        train_log_jsonl=log_dir / "train_log.jsonl",
    )
    return run_id, paths


class FlattenDictObsWrapper(gym.ObservationWrapper):
    """Flatten robotics dict obs into one vector: [observation, achieved_goal, desired_goal]."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_space = env.observation_space
        if not isinstance(obs_space, gym.spaces.Dict):
            raise TypeError("Expected Dict observation space for FlattenDictObsWrapper.")
        self.obs_dim = int(obs_space.spaces["observation"].shape[0])
        self.ag_dim = int(obs_space.spaces["achieved_goal"].shape[0])
        self.dg_dim = int(obs_space.spaces["desired_goal"].shape[0])
        flat_dim = self.obs_dim + self.ag_dim + self.dg_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(flat_dim,),
            dtype=np.float32,
        )

    def observation(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        out = np.concatenate(
            [
                np.asarray(observation["observation"], dtype=np.float32).reshape(-1),
                np.asarray(observation["achieved_goal"], dtype=np.float32).reshape(-1),
                np.asarray(observation["desired_goal"], dtype=np.float32).reshape(-1),
            ],
            axis=0,
        )
        return out.astype(np.float32, copy=False)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_single_env(env_id: str, seed: int, obs_delay_k: int, action_delay_k: int) -> gym.Env:
    env = gym.make(env_id, render_mode=None)
    env = build_perturbed_env(
        env,
        dropout_level="none",
        dropout_p=0.0,
        action_delay_k=int(action_delay_k),
        obs_delay_k=int(obs_delay_k),
        bias_b=0.0,
        force_wrappers=True,
        enable_disturbance_control=False,
        dropout_keys=("observation",),
        bias_keys=("observation",),
    )
    env = FlattenDictObsWrapper(env)
    env.reset(seed=int(seed))
    env.action_space.seed(int(seed))
    return env


class RecurrentPreprocessNet(nn.Module):
    """LSTM preprocess net for Tianshou actor/critic."""

    def __init__(
        self,
        input_dim: int,
        obs_dim: int,
        action_dim: int = 0,
        hidden_size: int = 256,
        lstm_hidden_size: int = 256,
        lstm_layers: int = 1,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.input_dim = int(input_dim)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.fc = nn.Linear(input_dim, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.output_dim = lstm_hidden_size
        self.to(device)

    def _reshape_sequence_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input to [B, T, input_dim].
        Handles:
        - plain obs: [B, obs_dim]
        - stacked obs: [B, T*obs_dim]
        - stacked obs + current action tail (critic): [B, T*obs_dim + action_dim]
        """
        if x.ndim == 1:
            x = x.view(1, -1)
        if x.ndim == 3:
            # Already [B, T, D].
            if x.shape[-1] == self.input_dim:
                return x
            # If D==obs_dim and critic expects +action_dim, append zeros as fallback.
            if self.action_dim > 0 and x.shape[-1] == self.obs_dim and self.input_dim == self.obs_dim + self.action_dim:
                zeros = torch.zeros(
                    (*x.shape[:-1], self.action_dim),
                    device=x.device,
                    dtype=x.dtype,
                )
                return torch.cat([x, zeros], dim=-1)
            raise RuntimeError(
                f"Unexpected 3D input shape {tuple(x.shape)} for preprocess "
                f"(expect last dim {self.input_dim})."
            )
        if x.ndim != 2:
            raise RuntimeError(f"Unsupported obs tensor ndim={x.ndim}, shape={tuple(x.shape)}")

        bsz, flat_dim = int(x.shape[0]), int(x.shape[1])
        if flat_dim == self.input_dim:
            return x.unsqueeze(1)

        # Case A: stacked obs only -> [B, T, obs_dim] for actor.
        if self.action_dim == 0 and self.obs_dim > 0 and flat_dim % self.obs_dim == 0:
            t = flat_dim // self.obs_dim
            return x.view(bsz, t, self.obs_dim)

        # Case B: critic-style packed input: [stacked_obs, action_tail]
        if self.action_dim > 0:
            obs_flat = flat_dim - self.action_dim
            if obs_flat > 0 and obs_flat % self.obs_dim == 0:
                t = obs_flat // self.obs_dim
                obs_seq = x[:, :obs_flat].view(bsz, t, self.obs_dim)
                act_tail = x[:, obs_flat:].view(bsz, 1, self.action_dim).expand(bsz, t, self.action_dim)
                seq = torch.cat([obs_seq, act_tail], dim=-1)
                if seq.shape[-1] != self.input_dim:
                    raise RuntimeError(
                        f"Critic packed input mismatch: got seq last dim {seq.shape[-1]}, expect {self.input_dim}."
                    )
                return seq

        raise RuntimeError(
            f"Cannot parse input shape {tuple(x.shape)} for preprocess "
            f"(obs_dim={self.obs_dim}, action_dim={self.action_dim}, input_dim={self.input_dim})."
        )

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: dict[str, torch.Tensor] | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del info
        x = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        x = self._reshape_sequence_input(x)

        x = F.relu(self.fc(x))
        # Avoid repeated cuDNN weight compaction overhead for LSTM.
        try:
            self.lstm.flatten_parameters()
        except Exception:  # noqa: BLE001
            pass

        hc = None
        if state is not None and "h" in state and "c" in state:
            hc = (state["h"].contiguous(), state["c"].contiguous())
        y, (h, c) = self.lstm(x, hc)
        feat = y[:, -1, :]
        return feat, {"h": h, "c": c}


def _safe_stat(result: Any, key_candidates: list[str], default: float = 0.0) -> float:
    for key in key_candidates:
        if isinstance(result, dict) and key in result:
            return float(result[key])
        if hasattr(result, key):
            try:
                return float(getattr(result, key))
            except Exception:  # noqa: BLE001
                pass
    return float(default)


def _as_numpy_action(act: Any) -> np.ndarray:
    if isinstance(act, torch.Tensor):
        return act.detach().cpu().numpy()
    return np.asarray(act)


def evaluate_success(
    policy: SACPolicy,
    *,
    env_id: str,
    base_seed: int,
    episodes: int,
    obs_delay_k: int,
    action_delay_k: int,
) -> dict[str, float]:
    """
    Deterministic-ish periodic eval to print key signals during training.
    """
    successes = 0
    total_reward = 0.0
    total_len = 0

    policy_was_training = policy.training
    policy.eval()
    try:
        for ep in range(int(episodes)):
            env = make_single_env(
                env_id=env_id,
                seed=int(base_seed) + ep,
                obs_delay_k=int(obs_delay_k),
                action_delay_k=int(action_delay_k),
            )
            obs, _ = env.reset(seed=int(base_seed) + ep)
            done = False
            ep_reward = 0.0
            ep_len = 0
            state = None
            last_info: dict[str, Any] = {}
            while not done:
                batch = Batch(obs=np.expand_dims(obs, axis=0), info={})
                with torch.no_grad():
                    out = policy(batch, state=state)
                action = _as_numpy_action(out.act)[0]
                state = getattr(out, "state", None)
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                ep_reward += float(reward)
                ep_len += 1
                if isinstance(info, dict):
                    last_info = info

            total_reward += ep_reward
            total_len += ep_len
            is_success = float(last_info.get("is_success", 0.0))
            successes += int(is_success >= 0.5)
            env.close()
    finally:
        if policy_was_training:
            policy.train()

    denom_ep = max(1, int(episodes))
    return {
        "success_rate": float(successes / denom_ep),
        "mean_reward": float(total_reward / denom_ep),
        "mean_ep_len": float(total_len / denom_ep),
    }


def save_checkpoint(
    path: Path,
    *,
    args: argparse.Namespace,
    policy: SACPolicy,
    obs_dim: int,
    act_dim: int,
    run_id: str,
    env_step: int,
) -> None:
    ckpt = {
        "run_id": run_id,
        "env_step": int(env_step),
        "obs_dim": int(obs_dim),
        "act_dim": int(act_dim),
        "args": vars(args),
        "actor": policy.actor.state_dict(),
        "critic1": policy.critic1.state_dict(),
        "critic2": policy.critic2.state_dict(),
        "actor_optim": policy.actor_optim.state_dict(),
        "critic1_optim": policy.critic1_optim.state_dict(),
        "critic2_optim": policy.critic2_optim.state_dict(),
    }
    # If automatic entropy tuning is enabled, store alpha optimizer states too.
    if hasattr(policy, "log_alpha"):
        ckpt["log_alpha"] = policy.log_alpha.detach().cpu()
    if hasattr(policy, "alpha_optim") and policy.alpha_optim is not None:
        ckpt["alpha_optim"] = policy.alpha_optim.state_dict()
    torch.save(ckpt, path)


def main() -> int:
    args = parse_args()
    run_id, run_paths = setup_run_dir(args)

    # Keep disturbance setup explicit in config.
    run_cfg: dict[str, Any] = {
        **vars(args),
        "resolved": {
            "repo_root": str(_REPO_ROOT),
            "run_id": run_id,
            "run_dir": str(run_paths.run_dir),
            "checkpoint_dir": str(run_paths.checkpoint_dir),
            "logs_dir": str(run_paths.log_dir),
            "docs_dir": str(run_paths.docs_dir),
            "context_len": int(args.seq_len + args.burn_in),
        },
        "train_disturbance": {
            "obs_delay_k": int(args.obs_delay_k),
            "action_delay_k": int(args.action_delay_k),
            "dropout_level": "none",
            "dropout_p": 0.0,
            "bias_b": 0.0,
        },
    }
    run_paths.config_json.write_text(json.dumps(run_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    set_global_seed(int(args.seed))
    device = choose_device(args.device)
    gym.register_envs(gymnasium_robotics)

    # Build one env first to infer dimensions.
    probe_env = make_single_env(
        env_id=args.env,
        seed=int(args.seed),
        obs_delay_k=int(args.obs_delay_k),
        action_delay_k=int(args.action_delay_k),
    )
    obs_dim = int(probe_env.observation_space.shape[0])
    act_dim = int(probe_env.action_space.shape[0])
    max_action = float(np.max(np.abs(probe_env.action_space.high)))
    probe_env.close()

    def train_env_fn(rank: int):
        return lambda: make_single_env(
            env_id=args.env,
            seed=int(args.seed) + int(rank),
            obs_delay_k=int(args.obs_delay_k),
            action_delay_k=int(args.action_delay_k),
        )

    def test_env_fn(rank: int):
        return lambda: make_single_env(
            env_id=args.env,
            seed=int(args.seed) + 10000 + int(rank),
            obs_delay_k=int(args.obs_delay_k),
            action_delay_k=int(args.action_delay_k),
        )

    env_fns = [train_env_fn(i) for i in range(int(args.n_envs))]
    if args.vec_env == "subproc" and int(args.n_envs) > 1:
        train_envs = SubprocVectorEnv(env_fns)
    else:
        train_envs = DummyVectorEnv(env_fns)
    test_envs = DummyVectorEnv([test_env_fn(0)])

    context_len = int(args.seq_len + args.burn_in)
    if context_len <= 0:
        raise ValueError("seq_len + burn_in must be > 0")

    actor_backbone = RecurrentPreprocessNet(
        input_dim=obs_dim,
        obs_dim=obs_dim,
        action_dim=0,
        hidden_size=int(args.hidden_size),
        lstm_hidden_size=int(args.lstm_hidden_size),
        lstm_layers=int(args.lstm_layers),
        device=device,
    )
    critic1_backbone = RecurrentPreprocessNet(
        input_dim=obs_dim + act_dim,
        obs_dim=obs_dim,
        action_dim=act_dim,
        hidden_size=int(args.hidden_size),
        lstm_hidden_size=int(args.lstm_hidden_size),
        lstm_layers=int(args.lstm_layers),
        device=device,
    )
    critic2_backbone = RecurrentPreprocessNet(
        input_dim=obs_dim + act_dim,
        obs_dim=obs_dim,
        action_dim=act_dim,
        hidden_size=int(args.hidden_size),
        lstm_hidden_size=int(args.lstm_hidden_size),
        lstm_layers=int(args.lstm_layers),
        device=device,
    )

    action_shape = tuple(probe for probe in [act_dim])
    actor = ActorProb(
        preprocess_net=actor_backbone,
        action_shape=action_shape,
        max_action=max_action,
        hidden_sizes=[256, 256],
        device=device,
        unbounded=True,
        conditioned_sigma=True,
        preprocess_net_output_dim=int(args.lstm_hidden_size),
    ).to(device)
    critic1 = Critic(
        preprocess_net=critic1_backbone,
        hidden_sizes=[256, 256],
        device=device,
        preprocess_net_output_dim=int(args.lstm_hidden_size),
    ).to(device)
    critic2 = Critic(
        preprocess_net=critic2_backbone,
        hidden_sizes=[256, 256],
        device=device,
        preprocess_net_output_dim=int(args.lstm_hidden_size),
    ).to(device)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=float(args.actor_lr))
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=float(args.critic_lr))
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=float(args.critic_lr))

    alpha_cfg: float | tuple[float, torch.Tensor, torch.optim.Optimizer]
    if bool(args.auto_alpha):
        target_entropy = -float(np.prod((act_dim,)))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=float(args.alpha_lr))
        alpha_cfg = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha_cfg = float(args.alpha)

    policy = SACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=float(args.tau),
        gamma=float(args.gamma),
        alpha=alpha_cfg,
        estimation_step=1,
        action_space=train_envs.action_space,
    )

    buffer = VectorReplayBuffer(
        total_size=int(args.buffer_size),
        buffer_num=int(args.n_envs),
        stack_num=context_len,
    )
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=False)

    print(f"[run] run_id={run_id}")
    print(f"[run] run_dir={run_paths.run_dir}")
    print(f"[env] env={args.env} obs_dim={obs_dim} act_dim={act_dim}")
    print(
        "[train] disturbance: "
        f"obs_delay_k={args.obs_delay_k} action_delay_k={args.action_delay_k} "
        "(dropout=none, bias=0)"
    )
    print(
        "[train] sequence params: "
        f"seq_len={args.seq_len} burn_in={args.burn_in} context_len={context_len}"
    )

    # Warm-up random data.
    if int(args.start_timesteps) > 0:
        print(f"[warmup] collecting {int(args.start_timesteps)} random steps ...")
        train_collector.collect(n_step=int(args.start_timesteps), random=True)

    env_step = 0
    last_save_step = 0
    last_log_step = 0
    start_time = time.time()

    with run_paths.train_log_jsonl.open("a", encoding="utf-8") as log_f:
        last_eval_step = 0
        while env_step < int(args.total_timesteps):
            to_collect = min(int(args.step_per_collect), int(args.total_timesteps) - env_step)
            result = train_collector.collect(n_step=to_collect)
            collected_steps = int(
                _safe_stat(result, ["n/st", "n_step", "n_collected_steps"], default=float(to_collect))
            )
            env_step += max(1, collected_steps)

            n_update = max(1, int(float(args.update_per_step) * collected_steps))
            last_loss: dict[str, float] = {}
            for _ in range(n_update):
                loss_out = policy.update(sample_size=int(args.batch_size), buffer=train_collector.buffer)
                if isinstance(loss_out, dict):
                    for k, v in loss_out.items():
                        try:
                            last_loss[k] = float(v)
                        except Exception:  # noqa: BLE001
                            pass

            need_log = env_step - last_log_step >= int(args.log_interval) or env_step >= int(args.total_timesteps)
            if need_log:
                elapsed = max(1e-6, time.time() - start_time)
                speed = float(env_step) / elapsed
                rew = _safe_stat(result, ["rew", "rews", "reward"], default=0.0)
                eval_stats: dict[str, float] | None = None
                if env_step - last_eval_step >= int(args.eval_interval) or env_step >= int(args.total_timesteps):
                    eval_stats = evaluate_success(
                        policy,
                        env_id=args.env,
                        base_seed=int(args.seed) + 500_000,
                        episodes=int(args.eval_episodes),
                        obs_delay_k=int(args.obs_delay_k),
                        action_delay_k=int(args.action_delay_k),
                    )
                    last_eval_step = env_step
                log_item = {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "env_step": int(env_step),
                    "speed_step_per_s": float(speed),
                    "collect_reward": float(rew),
                    "updates_this_iter": int(n_update),
                    "loss": last_loss,
                }
                if eval_stats is not None:
                    log_item["eval"] = eval_stats
                log_f.write(json.dumps(log_item, ensure_ascii=False) + "\n")
                log_f.flush()
                msg = (
                    f"[train] step={env_step}/{args.total_timesteps} "
                    f"speed={speed:.1f} step/s collect_reward={rew:.3f}"
                )
                if eval_stats is not None:
                    msg += (
                        " | "
                        f"eval_sr={eval_stats['success_rate']:.3f} "
                        f"eval_rew={eval_stats['mean_reward']:.2f} "
                        f"eval_len={eval_stats['mean_ep_len']:.1f} "
                        f"(n={int(args.eval_episodes)})"
                    )
                print(msg)
                last_log_step = env_step

            need_save = env_step - last_save_step >= int(args.save_interval) or env_step >= int(args.total_timesteps)
            if need_save:
                save_checkpoint(
                    run_paths.latest_ckpt,
                    args=args,
                    policy=policy,
                    obs_dim=obs_dim,
                    act_dim=act_dim,
                    run_id=run_id,
                    env_step=env_step,
                )
                step_ckpt = run_paths.checkpoint_dir / f"policy_step_{env_step}.pth"
                save_checkpoint(
                    step_ckpt,
                    args=args,
                    policy=policy,
                    obs_dim=obs_dim,
                    act_dim=act_dim,
                    run_id=run_id,
                    env_step=env_step,
                )
                print(f"[ckpt] wrote {run_paths.latest_ckpt}")
                last_save_step = env_step

    # Tiny post-train evaluation on training setting for quick sanity signal.
    test_result = test_collector.collect(n_episode=5)
    test_rew = _safe_stat(test_result, ["rew", "rews", "reward"], default=0.0)
    final_info = {
        "run_id": run_id,
        "final_env_step": int(env_step),
        "test_reward_5eps": float(test_rew),
        "checkpoint": str(run_paths.latest_ckpt),
    }
    (run_paths.docs_dir / "train_summary.json").write_text(
        json.dumps(final_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[done] run_id={run_id}")
    print(f"[done] final checkpoint: {run_paths.latest_ckpt}")

    train_envs.close()
    test_envs.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


