#!/usr/bin/env python3
"""
Evaluate recurrent SAC (Tianshou) run in 4 paper scenarios:
- Nominal   (k_act=0, k_obs=0)
- ObsOnly3  (k_act=0, k_obs=3)
- ActOnly1  (k_act=1, k_obs=0)
- MixedHard (k_act=1, k_obs=3)

Writes:
runs/<run_id>/docs/eval_4scenarios_ep1000.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make repo root importable when running as:
# python scripts/eval/eval_four_scenarios_rsac.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from perturbation_wrappers import build_perturbed_env  # noqa: E402

try:
    from tianshou.data import Batch
    from tianshou.policy import SACPolicy
    from tianshou.utils.net.continuous import ActorProb, Critic
except Exception as exc:  # noqa: BLE001
    raise ImportError(
        "This script requires tianshou. Please install it first, e.g. `pip install tianshou`."
    ) from exc


SCENARIOS = [
    ("Nominal", 0, 0),
    ("ObsOnly3", 0, 3),
    ("ActOnly1", 1, 0),
    ("MixedHard", 1, 3),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", type=str, required=True, help="Run id under runs/, or an absolute run dir path.")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env", type=str, default=None, help="Optional override for env id.")
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


@dataclass
class ScenarioMetrics:
    success_rate: float
    successes: int
    episodes: int


class FlattenDictObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_space = env.observation_space
        if not isinstance(obs_space, gym.spaces.Dict):
            raise TypeError("Expected Dict observation space.")
        obs_dim = int(obs_space.spaces["observation"].shape[0])
        ag_dim = int(obs_space.spaces["achieved_goal"].shape[0])
        dg_dim = int(obs_space.spaces["desired_goal"].shape[0])
        flat_dim = obs_dim + ag_dim + dg_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(flat_dim,),
            dtype=np.float32,
        )

    def observation(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(observation["observation"], dtype=np.float32).reshape(-1),
                np.asarray(observation["achieved_goal"], dtype=np.float32).reshape(-1),
                np.asarray(observation["desired_goal"], dtype=np.float32).reshape(-1),
            ],
            axis=0,
        ).astype(np.float32, copy=False)


class RecurrentPreprocessNet(nn.Module):
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
        if x.ndim == 1:
            x = x.view(1, -1)
        if x.ndim == 3:
            if x.shape[-1] == self.input_dim:
                return x
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

        if self.action_dim == 0 and self.obs_dim > 0 and flat_dim % self.obs_dim == 0:
            t = flat_dim // self.obs_dim
            return x.view(bsz, t, self.obs_dim)

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


def _resolve_run_dir(run_id: str) -> Path:
    p = Path(run_id).expanduser()
    if p.is_absolute() and p.exists():
        return p.resolve()
    cand = (_REPO_ROOT / "runs" / run_id).resolve()
    if cand.exists():
        return cand
    raise FileNotFoundError(f"Could not resolve run dir from run_id={run_id!r}")


def _device_name(arg: str) -> str:
    if arg != "auto":
        return arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_env(env_id: str, seed: int, k_act: int, k_obs: int) -> gym.Env:
    env = gym.make(env_id, render_mode=None)
    env = build_perturbed_env(
        env,
        dropout_level="none",
        dropout_p=0.0,
        action_delay_k=int(k_act),
        obs_delay_k=int(k_obs),
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


def _rebuild_policy(ckpt: dict[str, Any], env: gym.Env, device: str) -> SACPolicy:
    args = dict(ckpt.get("args", {}))
    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])

    actor_backbone = RecurrentPreprocessNet(
        input_dim=obs_dim,
        obs_dim=obs_dim,
        action_dim=0,
        hidden_size=int(args.get("hidden_size", 256)),
        lstm_hidden_size=int(args.get("lstm_hidden_size", 256)),
        lstm_layers=int(args.get("lstm_layers", 1)),
        device=device,
    )
    critic1_backbone = RecurrentPreprocessNet(
        input_dim=obs_dim + act_dim,
        obs_dim=obs_dim,
        action_dim=act_dim,
        hidden_size=int(args.get("hidden_size", 256)),
        lstm_hidden_size=int(args.get("lstm_hidden_size", 256)),
        lstm_layers=int(args.get("lstm_layers", 1)),
        device=device,
    )
    critic2_backbone = RecurrentPreprocessNet(
        input_dim=obs_dim + act_dim,
        obs_dim=obs_dim,
        action_dim=act_dim,
        hidden_size=int(args.get("hidden_size", 256)),
        lstm_hidden_size=int(args.get("lstm_hidden_size", 256)),
        lstm_layers=int(args.get("lstm_layers", 1)),
        device=device,
    )

    action_shape = tuple(v for v in [act_dim])
    max_action = float(np.max(np.abs(env.action_space.high)))
    actor = ActorProb(
        preprocess_net=actor_backbone,
        action_shape=action_shape,
        max_action=max_action,
        hidden_sizes=[256, 256],
        device=device,
        unbounded=True,
        conditioned_sigma=True,
        preprocess_net_output_dim=int(args.get("lstm_hidden_size", 256)),
    ).to(device)
    critic1 = Critic(
        preprocess_net=critic1_backbone,
        hidden_sizes=[256, 256],
        device=device,
        preprocess_net_output_dim=int(args.get("lstm_hidden_size", 256)),
    ).to(device)
    critic2 = Critic(
        preprocess_net=critic2_backbone,
        hidden_sizes=[256, 256],
        device=device,
        preprocess_net_output_dim=int(args.get("lstm_hidden_size", 256)),
    ).to(device)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=float(args.get("actor_lr", 3e-4)))
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=float(args.get("critic_lr", 3e-4)))
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=float(args.get("critic_lr", 3e-4)))

    if "log_alpha" in ckpt and "alpha_optim" in ckpt:
        target_entropy = -float(np.prod((act_dim,)))
        log_alpha = ckpt["log_alpha"].to(device)
        log_alpha = log_alpha.requires_grad_(True)
        alpha_optim = torch.optim.Adam([log_alpha], lr=float(args.get("alpha_lr", 3e-4)))
        alpha_cfg: float | tuple[float, torch.Tensor, torch.optim.Optimizer] = (
            target_entropy,
            log_alpha,
            alpha_optim,
        )
    else:
        alpha_cfg = float(args.get("alpha", 0.2))

    policy = SACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=float(args.get("tau", 0.005)),
        gamma=float(args.get("gamma", 0.98)),
        alpha=alpha_cfg,
        estimation_step=1,
        action_space=env.action_space,
    )

    policy.actor.load_state_dict(ckpt["actor"], strict=True)
    policy.critic1.load_state_dict(ckpt["critic1"], strict=True)
    policy.critic2.load_state_dict(ckpt["critic2"], strict=True)
    policy.actor_optim.load_state_dict(ckpt["actor_optim"])
    policy.critic1_optim.load_state_dict(ckpt["critic1_optim"])
    policy.critic2_optim.load_state_dict(ckpt["critic2_optim"])
    if "alpha_optim" in ckpt and hasattr(policy, "alpha_optim") and policy.alpha_optim is not None:
        policy.alpha_optim.load_state_dict(ckpt["alpha_optim"])
    return policy


def _eval_scenario(
    policy: SACPolicy,
    env: gym.Env,
    episodes: int,
    seed: int,
) -> ScenarioMetrics:
    successes = 0
    for ep in range(int(episodes)):
        obs, _ = env.reset(seed=int(seed) + ep)
        done = False
        state = None
        last_info: dict[str, Any] = {}
        while not done:
            batch = Batch(obs=np.expand_dims(obs, axis=0), info={})
            with torch.no_grad():
                out = policy(batch, state=state)
            act = out.act
            state = out.state
            if isinstance(act, torch.Tensor):
                action = act.detach().cpu().numpy()[0]
            else:
                action = np.asarray(act)[0]
            obs, _rew, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            if isinstance(info, dict):
                last_info = info

        is_success = float(last_info.get("is_success", 0.0))
        successes += int(is_success >= 0.5)

    sr = successes / max(1, int(episodes))
    return ScenarioMetrics(success_rate=float(sr), successes=int(successes), episodes=int(episodes))


def main() -> int:
    args = parse_args()
    gym.register_envs(gymnasium_robotics)

    run_dir = _resolve_run_dir(args.run_id)
    config_path = run_dir / "config.json"
    ckpt_path = run_dir / "checkpoints" / "policy_latest.pth"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    env_id = str(args.env or cfg.get("env", "FetchPickAndPlace-v4"))
    device = _device_name(args.device)

    load_env = _build_env(env_id, seed=int(args.seed), k_act=0, k_obs=0)
    ckpt = torch.load(ckpt_path, map_location=device)
    policy = _rebuild_policy(ckpt, env=load_env, device=device)
    policy.eval()

    results: dict[str, Any] = {
        "run_id": str(run_dir.name),
        "run_dir": str(run_dir),
        "env": env_id,
        "episodes_per_scenario": int(args.episodes),
        "seed": int(args.seed),
        "scenarios": {},
    }

    for name, k_act, k_obs in SCENARIOS:
        env = _build_env(env_id, seed=int(args.seed), k_act=int(k_act), k_obs=int(k_obs))
        m = _eval_scenario(policy, env, episodes=int(args.episodes), seed=int(args.seed))
        env.close()
        results["scenarios"][name] = {
            "k_act": int(k_act),
            "k_obs": int(k_obs),
            "episodes": int(m.episodes),
            "successes": int(m.successes),
            "success_rate": float(m.success_rate),
        }
        print(
            f"[{name}] k_act={k_act} k_obs={k_obs} "
            f"success_rate={m.success_rate:.4f} ({m.successes}/{m.episodes})"
        )

    out_path = run_dir / "docs" / f"eval_4scenarios_ep{int(args.episodes)}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[wrote] {out_path}")
    load_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


