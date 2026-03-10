"""
Conservative / motor-friendly modules:

Module D: Motor-friendly cost
  - energy proxy: ||a_t||^2
  - jerk proxy:   ||a_t - a_{t-1}||^2
  - can be logged as metrics and optionally added to reward

Module E/F: Uncertainty-aware action gating
  - use critic disagreement (double Q / ensemble) as an uncertainty proxy
  - uncertainty high -> gate down action magnitude (a = g * a_raw), g in [0, 1]

Design goals:
  - Keep Gymnasium-Robotics dict obs HER-compatible (do NOT add new obs keys)
  - Motor cost is goal-independent; we store per-step penalties in info so HER can
    include it when recomputing rewards (if enabled).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

try:
    import torch as th
except Exception as e:  # pragma: no cover
    raise ImportError("conservative_wrappers.py requires PyTorch to be installed") from e


@dataclass(frozen=True)
class MotorCostConfig:
    energy_w: float = 0.0
    jerk_w: float = 0.0
    add_to_reward: bool = False
    # If True, wrapper.compute_reward() will add motor penalty from info (so HER recompute stays consistent).
    include_in_compute_reward: bool = True


class MotorCostWrapper(gym.Wrapper):
    """
    Log (and optionally shape reward with) motor-friendly costs.

    Metrics added to info each step:
      - motor_energy: ||a_t||^2
      - motor_jerk:   ||a_t - a_{t-1}||^2
      - motor_cost:   energy_w * motor_energy + jerk_w * motor_jerk
      - motor_reward_penalty: -motor_cost  (only when add_to_reward=True else 0)

    If add_to_reward=True:
      reward <- reward + motor_reward_penalty

    HER compatibility:
      Gymnasium-Robotics reward is recomputed via env.compute_reward(ag, dg, info) inside HER.
      That signature does NOT include action. To keep consistency, we store the penalty in `info`
      and add it back inside wrapper.compute_reward() when include_in_compute_reward=True.
    """

    def __init__(self, env: gym.Env, cfg: MotorCostConfig | None = None):
        super().__init__(env)
        self.cfg = cfg or MotorCostConfig()
        self._prev_action: np.ndarray | None = None

        if not isinstance(self.action_space, gym.spaces.Box) or self.action_space.shape is None:
            raise TypeError("MotorCostWrapper requires a continuous Box action space")

        self._act_dim = int(self.action_space.shape[0])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_action = np.zeros((self._act_dim,), dtype=np.float32)
        return obs, info

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != self._act_dim:
            a = a[: self._act_dim]

        prev = self._prev_action
        if prev is None:
            prev = np.zeros((self._act_dim,), dtype=np.float32)

        energy = float(np.dot(a, a))
        da = a - prev
        jerk = float(np.dot(da, da))
        cost = float(self.cfg.energy_w) * energy + float(self.cfg.jerk_w) * jerk
        reward_penalty = -cost if bool(self.cfg.add_to_reward) else 0.0

        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info) if isinstance(info, dict) else {"info": info}
        info["motor_energy"] = energy
        info["motor_jerk"] = jerk
        info["motor_cost"] = cost
        info["motor_reward_penalty"] = float(reward_penalty)

        if bool(self.cfg.add_to_reward):
            reward = float(reward) + float(reward_penalty)

        self._prev_action = a
        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info: dict[str, Any] | None = None):
        """
        Goal reward + (optional) motor penalty stored in info.
        This makes HER reward recomputation consistent with step() when enabled.
        """
        base = None
        if hasattr(self.env, "compute_reward"):
            base = self.env.compute_reward(achieved_goal, desired_goal, info)
        else:
            # Fallback: no compute_reward; just return 0.
            base = 0.0

        if not (bool(self.cfg.add_to_reward) and bool(self.cfg.include_in_compute_reward)):
            return base

        if info is None:
            return base

        # SB3 may pass a dict-like or list of dicts; handle dict only (non-Vec env).
        try:
            pen = float(info.get("motor_reward_penalty", 0.0))
        except Exception:
            pen = 0.0
        return base + pen


def _to_torch_obs(model, obs):
    """
    Convert an env observation into policy input tensors (SB3-compatible).
    Uses model.policy.obs_to_tensor when available.
    """
    if hasattr(model, "policy") and hasattr(model.policy, "obs_to_tensor"):
        obs_t, _ = model.policy.obs_to_tensor(obs)
        return obs_t
    # Fallback: assume array obs
    x = th.as_tensor(obs).float()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return x


def _scale_action_for_critic(model, action: np.ndarray) -> np.ndarray:
    """
    Critic usually expects actions in [-1, 1] space. If policy exposes scale_action(), use it.
    """
    a = np.asarray(action, dtype=np.float32)
    if hasattr(model, "policy") and hasattr(model.policy, "scale_action"):
        try:
            return np.asarray(model.policy.scale_action(a), dtype=np.float32)
        except Exception:
            return a
    return a


def estimate_critic_disagreement(model, obs, action) -> float:
    """
    Uncertainty proxy from critic ensemble disagreement.

    For SAC double-Q this is typically |Q1 - Q2| evaluated at the current (obs, action).
    For N critics, we use (max - min).
    """
    obs_t = _to_torch_obs(model, obs)
    a_scaled = _scale_action_for_critic(model, action)
    a_t = th.as_tensor(a_scaled).float()
    if a_t.dim() == 1:
        a_t = a_t.unsqueeze(0)

    device = getattr(getattr(model, "policy", None), "device", None)
    if device is not None:
        obs_t = (  # dict obs
            {k: v.to(device) for k, v in obs_t.items()} if isinstance(obs_t, dict) else obs_t.to(device)
        )
        a_t = a_t.to(device)

    with th.no_grad():
        critic = getattr(getattr(model, "policy", None), "critic", None)
        if critic is None:
            return 0.0
        qs = critic(obs_t, a_t)
        # qs is usually a tuple/list of tensors (batch, 1)
        if isinstance(qs, (list, tuple)):
            qcat = th.cat([q if q.dim() == 2 else q.unsqueeze(1) for q in qs], dim=1)
            u = (qcat.max(dim=1).values - qcat.min(dim=1).values)
            return float(u.item())
        if isinstance(qs, th.Tensor):
            if qs.dim() == 2 and qs.shape[1] > 1:
                u = qs.max(dim=1).values - qs.min(dim=1).values
                return float(u.item())
            return float(qs.std().item())
    return 0.0


def gate_from_uncertainty(u: float, *, threshold: float, alpha: float, min_g: float = 0.0) -> float:
    """
    Map uncertainty -> gate in [min_g, 1].

    If u <= threshold => g = 1
    Else g = exp(-alpha * (u - threshold))
    """
    if u <= float(threshold):
        return 1.0
    g = float(np.exp(-float(alpha) * (float(u) - float(threshold))))
    g = float(np.clip(g, float(min_g), 1.0))
    return g


def predict_with_uncertainty_gating(
    model,
    obs,
    *,
    deterministic: bool = True,
    threshold: float = 1.0,
    alpha: float = 1.0,
    min_g: float = 0.0,
):
    """
    Returns gated action and diagnostics dict:
      action = g * action_raw
      diag = {"uncertainty": u, "g": g}
    """
    action_raw, state = model.predict(obs, deterministic=deterministic)
    u = estimate_critic_disagreement(model, obs, action_raw)
    g = gate_from_uncertainty(u, threshold=threshold, alpha=alpha, min_g=min_g)

    a = np.asarray(action_raw, dtype=np.float32) * float(g)
    # keep action bounds
    if hasattr(model, "action_space"):
        a = np.clip(a, model.action_space.low, model.action_space.high)
    return a, state, {"uncertainty": float(u), "g": float(g)}


