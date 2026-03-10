"""
Perturbation wrappers for robustness experiments.

Designed for Gymnasium-Robotics dict observations:
  {"observation": np.ndarray, "achieved_goal": np.ndarray, "desired_goal": np.ndarray}

Wrappers implemented:
- ObsDropoutWrapper: simulate occlusion / dropped frames (drop-to-zero or hold-last)
- DelayWrapper: action delay and/or observation delay
- ObsBiasWrapper: per-episode constant calibration bias on object-related dimensions
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import gymnasium as gym
import numpy as np


def parse_indices(indices: str | None) -> list[int] | None:
    """Parse comma-separated indices like '3,4,5' -> [3,4,5]."""
    if indices is None:
        return None
    s = indices.strip()
    if not s:
        return None
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out or None


def _copy_obs(obs):
    """Shallow-copy an observation, copying arrays so we can mutate safely."""
    if isinstance(obs, dict):
        out = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                out[k] = v.copy()
            else:
                out[k] = v
        return out
    if isinstance(obs, np.ndarray):
        return obs.copy()
    return obs


def _apply_to_1d(arr: np.ndarray, idx: Sequence[int] | None, fn) -> np.ndarray:
    """Apply fn to selected indices of a 1D array. fn receives a view and must return values."""
    if arr.ndim != 1:
        # Keep it simple/robust: if shape is not 1D, apply to the whole array.
        return fn(arr)
    if idx is None:
        return fn(arr)
    idx_arr = np.asarray(idx, dtype=int)
    arr[idx_arr] = fn(arr[idx_arr])
    return arr


@dataclass(frozen=True)
class DropoutSpec:
    """Convenience mapping for easy/med/hard."""

    level: str

    def p(self) -> float:
        level = self.level.lower()
        if level in ("none", "off", "0"):
            return 0.0
        if level == "easy":
            return 0.1
        if level in ("med", "medium"):
            return 0.3
        if level == "hard":
            return 0.5
        raise ValueError(f"Unknown dropout level: {self.level!r} (use none/easy/med/hard)")


class ObsDropoutWrapper(gym.Wrapper):
    """
    Simulate occlusion / dropped frames on selected observation dimensions.

    Two modes:
    - drop-to-zero: with probability p, set selected dims to 0
    - hold-last: with probability p, replace selected dims with previous output (frame hold)

    Notes:
    - For dict obs, by default we perturb keys ("observation", "achieved_goal") and leave "desired_goal" intact.
    - obj_idx applies only to the "observation" vector (1D). For achieved_goal, all dims are perturbed when included.
    """

    def __init__(
        self,
        env: gym.Env,
        p: float,
        mode: str = "hold-last",
        keys: Sequence[str] = ("observation", "achieved_goal"),
        obj_idx: Sequence[int] | None = None,
        exclude_gripper: bool = False,
    ):
        super().__init__(env)
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p}")
        mode = mode.lower()
        if mode not in ("drop-to-zero", "hold-last"):
            raise ValueError("mode must be 'drop-to-zero' or 'hold-last'")

        self.p = float(p)
        self.mode = mode
        self.keys = tuple(keys)
        self.obj_idx = list(obj_idx) if obj_idx is not None else None
        self.exclude_gripper = bool(exclude_gripper)

        self._last_obs = None
        self._last_obs_mask: np.ndarray | None = None
        self._rng = np.random.default_rng()

    def set_p(self, p: float):
        """Update dropout probability at runtime (curriculum support)."""
        if not (0.0 <= float(p) <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.p = float(p)

    def set_mode(self, mode: str):
        mode = str(mode).lower()
        if mode not in ("drop-to-zero", "hold-last"):
            raise ValueError("mode must be 'drop-to-zero' or 'hold-last'")
        self.mode = mode

    def reset(self, **kwargs):
        seed = kwargs.get("seed", None)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs, info = self.env.reset(**kwargs)
        obs_out, mask = self._apply(obs, is_reset=True)
        self._last_obs = _copy_obs(obs_out)
        self._last_obs_mask = mask
        if isinstance(info, dict):
            info = dict(info)
            if mask is not None:
                info.setdefault("obs_mask", mask.copy())
        return obs_out, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_out, mask = self._apply(obs, is_reset=False)
        self._last_obs = _copy_obs(obs_out)
        self._last_obs_mask = mask
        if isinstance(info, dict):
            info = dict(info)
        else:
            info = {}
        if mask is not None:
            # A generic mask for obs["observation"] dims: 1=valid, 0=occluded/held.
            info.setdefault("obs_mask", mask.copy())
        return obs_out, reward, terminated, truncated, info

    def _apply(self, obs, *, is_reset: bool):
        """
        Returns (obs_out, mask) where mask is a float32 vector aligned with obs["observation"]
        (or None if we cannot infer it).
        """
        if self.p <= 0.0:
            mask = None
            if isinstance(obs, dict) and isinstance(obs.get("observation", None), np.ndarray):
                mask = np.ones_like(obs["observation"], dtype=np.float32)
            return obs, mask
        if self.mode == "hold-last" and self._last_obs is None and not is_reset:
            # Should not happen, but be defensive.
            self._last_obs = _copy_obs(obs)

        drop = self._rng.random() < self.p
        if not drop:
            mask = None
            if isinstance(obs, dict) and isinstance(obs.get("observation", None), np.ndarray):
                mask = np.ones_like(obs["observation"], dtype=np.float32)
            return obs, mask

        obs_out = _copy_obs(obs)

        def maybe_idx_for_key(key: str, arr: np.ndarray) -> list[int] | None:
            if key != "observation":
                return None
            if self.obj_idx is not None:
                return self.obj_idx
            if self.exclude_gripper and arr.ndim == 1 and arr.shape[0] >= 3:
                return list(range(3, arr.shape[0]))
            return None

        if isinstance(obs_out, dict):
            for k in self.keys:
                if k not in obs_out:
                    continue
                v = obs_out[k]
                if not isinstance(v, np.ndarray):
                    continue
                idx = maybe_idx_for_key(k, v)
                if self.mode == "drop-to-zero":
                    obs_out[k] = _apply_to_1d(v, idx, lambda x: np.zeros_like(x))
                else:
                    prev = self._last_obs[k] if isinstance(self._last_obs, dict) and k in self._last_obs else v
                    prev = prev.copy() if isinstance(prev, np.ndarray) else v
                    if idx is None:
                        obs_out[k] = prev
                    else:
                        idx_arr = np.asarray(idx, dtype=int)
                        v[idx_arr] = prev[idx_arr]
                        obs_out[k] = v
            # Build a mask for obs["observation"] if present
            mask = None
            if "observation" in obs_out and isinstance(obs_out["observation"], np.ndarray):
                mask = np.ones_like(obs_out["observation"], dtype=np.float32)
                idx = maybe_idx_for_key("observation", obs_out["observation"])
                if idx is None:
                    mask[:] = 0.0
                else:
                    mask[np.asarray(idx, dtype=int)] = 0.0
            return obs_out, mask

        if isinstance(obs_out, np.ndarray):
            idx = self.obj_idx
            if idx is None and self.exclude_gripper and obs_out.ndim == 1 and obs_out.shape[0] >= 3:
                idx = list(range(3, obs_out.shape[0]))
            if self.mode == "drop-to-zero":
                out = _apply_to_1d(obs_out, idx, lambda x: np.zeros_like(x))
                mask = np.ones_like(out, dtype=np.float32)
                if idx is None:
                    mask[:] = 0.0
                else:
                    mask[np.asarray(idx, dtype=int)] = 0.0
                return out, mask
            prev = self._last_obs if isinstance(self._last_obs, np.ndarray) else obs_out
            if idx is None:
                out = prev.copy()
                mask = np.zeros_like(out, dtype=np.float32)
                return out, mask
            idx_arr = np.asarray(idx, dtype=int)
            obs_out[idx_arr] = prev[idx_arr]
            mask = np.ones_like(obs_out, dtype=np.float32)
            mask[idx_arr] = 0.0
            return obs_out, mask

        return obs_out, None


class DelayWrapper(gym.Wrapper):
    """
    Add fixed k-step delay for actions and/or observations.

    - action_delay_k: executed action is delayed by k steps (queue). k=0 => no delay.
    - obs_delay_k: returned observation is delayed by k steps. k=0 => no delay.
    """

    def __init__(self, env: gym.Env, action_delay_k: int = 0, obs_delay_k: int = 0, *, role: str = "both"):
        super().__init__(env)
        if action_delay_k < 0 or obs_delay_k < 0:
            raise ValueError("Delay k must be >= 0")
        self.action_delay_k = int(action_delay_k)
        self.obs_delay_k = int(obs_delay_k)
        self.role = str(role)

        self._action_q: deque[np.ndarray] | None = None
        self._obs_q: deque | None = None
        self._last_obs = None

    def set_delays(self, *, action_delay_k: int | None = None, obs_delay_k: int | None = None):
        """Update delays at runtime (curriculum support)."""
        if action_delay_k is not None:
            k = int(action_delay_k)
            if k < 0:
                raise ValueError("action_delay_k must be >= 0")
            # Only reset queue if delay value actually changes
            if k != self.action_delay_k:
                self.action_delay_k = k
                if self.action_delay_k > 0:
                    zero_action = np.zeros(self.action_space.shape, dtype=self.action_space.dtype)
                    self._action_q = deque([zero_action.copy() for _ in range(self.action_delay_k)])
                else:
                    self._action_q = None
        if obs_delay_k is not None:
            k = int(obs_delay_k)
            if k < 0:
                raise ValueError("obs_delay_k must be >= 0")
            # Only reset queue if delay value actually changes
            if k != self.obs_delay_k:
                self.obs_delay_k = k
                if self.obs_delay_k > 0:
                    seed_obs = _copy_obs(self._last_obs) if self._last_obs is not None else None
                    self._obs_q = deque([seed_obs]) if seed_obs is not None else deque()
                else:
                    self._obs_q = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = _copy_obs(obs)
        # init action queue with zeros
        if self.action_delay_k > 0:
            zero_action = np.zeros(self.action_space.shape, dtype=self.action_space.dtype)
            self._action_q = deque([zero_action.copy() for _ in range(self.action_delay_k)])
        else:
            self._action_q = None

        # init obs queue with the initial obs
        if self.obs_delay_k > 0:
            self._obs_q = deque([_copy_obs(obs)])
        else:
            self._obs_q = None
        return _copy_obs(obs) if self.obs_delay_k > 0 else obs, info

    def step(self, action):
        exec_action = action
        if self.action_delay_k > 0:
            assert self._action_q is not None
            self._action_q.append(np.asarray(action).copy())
            exec_action = self._action_q.popleft()


        
        obs, reward, terminated, truncated, info = self.env.step(exec_action)
        self._last_obs = _copy_obs(obs)
        # Expose both the issued (current) action and the actually executed action.
        # Important for delay-aware history encoders: when action delay exists, the executed action
        # is NOT the same as the policy output at this step.
        if isinstance(info, dict):
            info = dict(info)
        else:
            info = {}
        # Don't overwrite inner wrapper's executed_action (e.g. when an obs-delay wrapper wraps an action-delay wrapper).
        info.setdefault("issued_action", np.asarray(action).copy())
        info.setdefault("executed_action", np.asarray(exec_action).copy())

        # Inject current delay values for logging
        if self.role in ("action", "both"):
            info["k_act"] = self.action_delay_k
        if self.role in ("obs", "both"):
            info["k_obs"] = self.obs_delay_k

        if self.obs_delay_k <= 0:
            return obs, reward, terminated, truncated, info

        assert self._obs_q is not None
        self._obs_q.append(_copy_obs(obs))
        # keep only last (k+1) observations; output oldest among them
        while len(self._obs_q) > self.obs_delay_k + 1:
            self._obs_q.popleft()
        delayed_obs = _copy_obs(self._obs_q[0])
        return delayed_obs, reward, terminated, truncated, info


class ObsBiasWrapper(gym.Wrapper):
    """
    Per-episode constant bias on selected observation dimensions.

    At each reset:
        bias ~ Uniform([-b, b]) for each biased dimension

    Bias remains fixed for the entire episode.

    Notes:
    - For dict obs, by default we bias keys ("observation", "achieved_goal") and leave "desired_goal" intact.
    - obj_idx applies only to the "observation" vector (1D). For achieved_goal, all dims are biased when included.
    """

    def __init__(
        self,
        env: gym.Env,
        b: float,
        keys: Sequence[str] = ("observation", "achieved_goal"),
        obj_idx: Sequence[int] | None = None,
    ):
        super().__init__(env)
        if b < 0:
            raise ValueError("b must be >= 0")
        self.b = float(b)
        self.keys = tuple(keys)
        self.obj_idx = list(obj_idx) if obj_idx is not None else None

        self._bias_obs = None  # bias vector for "observation" key indices (only selected dims non-zero)
        self._bias_goal = None  # bias for achieved_goal (full dims)
        self._rng = np.random.default_rng()
        self._last_obs = None

    def set_b(self, b: float):
        """Update bias magnitude at runtime (curriculum support)."""
        if float(b) < 0.0:
            raise ValueError("b must be >= 0")
        self.b = float(b)
        # Re-sample on next reset; if we're mid-episode, re-sample from last obs as a best-effort.
        if self._last_obs is not None:
            self._sample_bias(self._last_obs)

    def reset(self, **kwargs):
        seed = kwargs.get("seed", None)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs, info = self.env.reset(**kwargs)
        self._last_obs = _copy_obs(obs)
        self._sample_bias(obs)
        obs_out = self._apply(obs)
        # expose bias in info (useful for debugging/analysis; also for HER if needed)
        if isinstance(info, dict):
            info = dict(info)
            info["obs_bias_b"] = self.b
            if self._bias_goal is not None:
                info["obs_bias_goal"] = np.array(self._bias_goal, copy=True)
        return obs_out, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = _copy_obs(obs)
        obs_out = self._apply(obs)
        if isinstance(info, dict):
            info = dict(info)
            info["obs_bias_b"] = self.b
            if self._bias_goal is not None:
                info["obs_bias_goal"] = np.array(self._bias_goal, copy=True)
        return obs_out, reward, terminated, truncated, info

    def _sample_bias(self, obs):
        if self.b <= 0.0:
            self._bias_obs = None
            self._bias_goal = None
            return

        if isinstance(obs, dict):
            # bias for achieved_goal (if present)
            if "achieved_goal" in obs and isinstance(obs["achieved_goal"], np.ndarray):
                g = obs["achieved_goal"]
                self._bias_goal = self._rng.uniform(-self.b, self.b, size=g.shape).astype(g.dtype)
            else:
                self._bias_goal = None

            # bias for observation vector
            if "observation" in obs and isinstance(obs["observation"], np.ndarray):
                o = obs["observation"]
                bias = np.zeros_like(o)
                idx = self.obj_idx
                # If user didn't specify indices, default to Fetch-style object position [3:6] when possible.
                if idx is None and o.ndim == 1 and o.shape[0] >= 6:
                    idx = [3, 4, 5]
                if idx is None:
                    bias = self._rng.uniform(-self.b, self.b, size=o.shape).astype(o.dtype)
                else:
                    idx_arr = np.asarray(idx, dtype=int)
                    bias[idx_arr] = self._rng.uniform(-self.b, self.b, size=idx_arr.shape).astype(o.dtype)
                self._bias_obs = bias
            else:
                self._bias_obs = None
            return

        if isinstance(obs, np.ndarray):
            o = obs
            bias = self._rng.uniform(-self.b, self.b, size=o.shape).astype(o.dtype)
            self._bias_obs = bias
            self._bias_goal = None

    def _apply(self, obs):
        if self.b <= 0.0:
            return obs
        obs_out = _copy_obs(obs)
        if isinstance(obs_out, dict):
            for k in self.keys:
                if k not in obs_out:
                    continue
                v = obs_out[k]
                if not isinstance(v, np.ndarray):
                    continue
                if k == "observation" and self._bias_obs is not None:
                    obs_out[k] = v + self._bias_obs
                elif k == "achieved_goal" and self._bias_goal is not None:
                    obs_out[k] = v + self._bias_goal
            return obs_out

        if isinstance(obs_out, np.ndarray) and self._bias_obs is not None:
            return obs_out + self._bias_obs
        return obs_out


def build_perturbed_env(
    env: gym.Env,
    *,
    dropout_p: float = 0.0,
    dropout_mode: str = "hold-last",
    dropout_level: str | None = None,
    dropout_keys: Sequence[str] = ("observation", "achieved_goal"),
    dropout_exclude_gripper: bool = False,
    dropout_obj_idx: Sequence[int] | None = None,
    action_delay_k: int = 0,
    obs_delay_k: int = 0,
    bias_b: float = 0.0,
    bias_keys: Sequence[str] = ("observation", "achieved_goal"),
    bias_obj_idx: Sequence[int] | None = None,
    force_wrappers: bool = False,
    enable_disturbance_control: bool = False,
) -> gym.Env:
    """
    Apply wrappers in a consistent order:
    action delay -> base env -> bias -> dropout -> obs delay

    Order rationale:
    - action delay should affect control actually executed
    - bias/dropout simulate perception before delay (then delayed perception is what agent receives)
    """
    # If we want a curriculum that can turn disturbances ON later, we must instantiate wrappers
    # even when initial parameters are "off". When force_wrappers=True, wrappers must behave as no-ops
    # when their parameters are set to 0.
    use_force = bool(force_wrappers)

    # DelayWrapper handles both action and obs delay; we want obs delay outermost,
    # but action delay as innermost. We'll implement with two wrappers if needed.
    if use_force or action_delay_k > 0:
        env = DelayWrapper(env, action_delay_k=action_delay_k, obs_delay_k=0, role="action")

    if use_force or bias_b > 0:
        env = ObsBiasWrapper(env, b=bias_b, keys=bias_keys, obj_idx=bias_obj_idx)

    p = dropout_p
    if dropout_level is not None:
        p = DropoutSpec(dropout_level).p()
    if use_force or p > 0:
        env = ObsDropoutWrapper(
            env,
            p=p,
            mode=dropout_mode,
            keys=dropout_keys,
            obj_idx=dropout_obj_idx,
            exclude_gripper=dropout_exclude_gripper,
        )

    if use_force or obs_delay_k > 0:
        env = DelayWrapper(env, action_delay_k=0, obs_delay_k=obs_delay_k, role="obs")

    if bool(enable_disturbance_control) or use_force:
        env = DisturbanceManagerWrapper(env)
    return env


class DisturbanceManagerWrapper(gym.Wrapper):
    """
    A tiny "control plane" wrapper that exposes env.set_disturbance(...) for curriculum.

    It does not change observations/rewards; it only traverses inner wrappers and updates
    their parameters if supported.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._last = {
            "dropout_p": None,
            "dropout_level": None,
            "k_obs": None,
            "k_act": None,
            "bias_b": None,
        }

    def _iter_wrappers(self):
        e = self.env
        while True:
            yield e
            if not hasattr(e, "env"):
                break
            e = getattr(e, "env")

    def set_disturbance(
        self,
        *,
        dropout_level: str | None = None,
        dropout_p: float | None = None,
        k_obs: int | None = None,
        k_act: int | None = None,
        bias_b: float | None = None,
    ):
        """
        Update disturbance parameters at runtime.

        - dropout: either pass dropout_level (none/easy/med/hard) or dropout_p directly.
        - delays: k_obs / k_act are control-cycle delays.
        - bias: bias_b is meters (Uniform([-b,b])).
        """
        if dropout_level is not None and dropout_p is not None:
            raise ValueError("Pass only one of dropout_level or dropout_p.")

        p = None
        if dropout_level is not None:
            p = DropoutSpec(dropout_level).p()
        if dropout_p is not None:
            p = float(dropout_p)

        for w in self._iter_wrappers():
            if isinstance(w, ObsDropoutWrapper) and p is not None:
                w.set_p(p)
            if isinstance(w, ObsBiasWrapper) and bias_b is not None:
                w.set_b(float(bias_b))
            if isinstance(w, DelayWrapper):
                # Only set the intended delay wrapper (we usually create two: role=action and role=obs)
                if k_act is not None and getattr(w, "role", "both") in ("action", "both"):
                    w.set_delays(action_delay_k=int(k_act))
                if k_obs is not None and getattr(w, "role", "both") in ("obs", "both"):
                    w.set_delays(obs_delay_k=int(k_obs))

        # remember for debugging/logging
        if p is not None:
            self._last["dropout_p"] = float(p)
            self._last["dropout_level"] = dropout_level
        if k_obs is not None:
            self._last["k_obs"] = int(k_obs)
        if k_act is not None:
            self._last["k_act"] = int(k_act)
        if bias_b is not None:
            self._last["bias_b"] = float(bias_b)

    def get_disturbance(self) -> dict:
        return dict(self._last)


