"""
Residual-memory controller modules for: Pretrained SAC + Residual Memory + Fourier Features.

Key design constraints:
- Keep base SAC policy input dimension unchanged (do NOT touch env obs keys for HER).
- Residual controller consumes a custom input built from obs + a_prev + (optional) dropout mask.
- Default residual memory: GRU. Optional: CfC (ncps) if installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch as th
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:  # pragma: no cover
    raise ImportError("residual_mem_modules.py requires PyTorch to be installed") from e


class FourierFeatures(nn.Module):
    """
    Fourier features for low-dimensional continuous inputs.

    Given x in R^{in_dim}, output concat(sin(x*f), cos(x*f)) over log-spaced frequency bands.
    """

    def __init__(self, in_dim: int, num_bands: int = 8, max_freq: float = 10.0):
        super().__init__()
        if in_dim <= 0:
            raise ValueError("in_dim must be > 0")
        if num_bands <= 0:
            raise ValueError("num_bands must be > 0")
        if max_freq <= 0:
            raise ValueError("max_freq must be > 0")

        self.in_dim = int(in_dim)
        self.num_bands = int(num_bands)
        self.max_freq = float(max_freq)

        # logspace frequencies in [1, max_freq]
        freqs = th.logspace(0.0, float(np.log10(self.max_freq)), steps=self.num_bands)
        self.register_buffer("freqs", freqs, persistent=False)  # (num_bands,)

    @property
    def out_dim(self) -> int:
        return 2 * self.in_dim * self.num_bands

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x: (..., in_dim)
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"Expected last dim {self.in_dim}, got {x.shape[-1]}")
        # (..., in_dim, num_bands)
        xf = x.unsqueeze(-1) * self.freqs.view(*([1] * (x.dim() - 1)), 1, -1)
        s = th.sin(xf)
        c = th.cos(xf)
        # (..., in_dim*num_bands)
        s = s.reshape(*x.shape[:-1], self.in_dim * self.num_bands)
        c = c.reshape(*x.shape[:-1], self.in_dim * self.num_bands)
        return th.cat([s, c], dim=-1)


def _tanh_squash_log_prob(u: th.Tensor, log_prob_u: th.Tensor, eps: float = 1e-6) -> th.Tensor:
    """
    Apply tanh correction for a squashed Gaussian.
    u: pre-tanh action, log_prob_u: log N(u; mu, std).
    Returns log_prob(tanh(u)).
    """
    a = th.tanh(u)
    # log|det(d tanh(u) / du)| = sum log(1 - tanh(u)^2)
    corr = th.log(1.0 - a.pow(2) + eps)
    return log_prob_u - corr.sum(dim=-1, keepdim=True)


class ResidualMemPolicy(nn.Module):
    """
    Residual controller with memory.

    We model Δa as a tanh-squashed Gaussian:
      u ~ N(mu(h), std(h)),  Δa = tanh(u)
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        mem_type: str = "gru",
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        self.hidden_size = int(hidden_size)
        self.mem_type = str(mem_type).lower()
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        if self.mem_type not in ("gru", "cfc"):
            raise ValueError("mem_type must be one of {gru,cfc}")

        if self.mem_type == "gru":
            self.rnn = nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_size, batch_first=True)
        else:
            # Optional dependency (ncps). Keep import local.
            try:
                from ncps.torch import CfC  # type: ignore
            except Exception as e:  # pragma: no cover
                raise ImportError("mem_type=cfc requires `ncps` to be installed") from e
            # CfC supports variable dt via timespans; for simplicity we assume dt=1.0 here.
            self.rnn = CfC(self.input_dim, self.hidden_size, batch_first=True)

        self.mu = nn.Linear(self.hidden_size, self.action_dim)
        self.log_std = nn.Linear(self.hidden_size, self.action_dim)

    def init_hidden(self, batch_size: int, device: th.device) -> th.Tensor:
        # GRU: (num_layers=1, batch, hidden)
        return th.zeros((1, int(batch_size), self.hidden_size), device=device)

    def forward_sequence(
        self,
        x_seq: th.Tensor,
        h0: Optional[th.Tensor] = None,
        *,
        deterministic: bool = False,
    ):
        """
        x_seq: (batch, time, input_dim)
        Returns:
          delta:  (batch, time, action_dim) in [-1, 1]
          logp:   (batch, time, 1) log prob of delta under the policy
          h_last: hidden state after processing the sequence
        """
        if x_seq.dim() != 3:
            raise ValueError("x_seq must be (batch, time, input_dim)")
        b = int(x_seq.shape[0])
        device = x_seq.device
        if h0 is None:
            h0 = self.init_hidden(b, device)

        if self.mem_type == "gru":
            out, h_last = self.rnn(x_seq, h0)  # out: (b,t,h)
        else:
            # CfC: out, h_last (API compatible with batch_first=True)
            out, h_last = self.rnn(x_seq, h0)

        mu = self.mu(out)
        log_std = self.log_std(out).clamp(self.log_std_min, self.log_std_max)
        std = th.exp(log_std)

        if deterministic:
            u = mu
        else:
            u = mu + std * th.randn_like(mu)

        delta = th.tanh(u)

        # log prob (squashed Gaussian)
        # log N(u; mu, std) = -0.5 * (((u-mu)/std)^2 + 2log_std + log(2π))
        log_prob_u = -0.5 * (((u - mu) / (std + 1e-8)).pow(2) + 2.0 * log_std + np.log(2.0 * np.pi))
        log_prob_u = log_prob_u.sum(dim=-1, keepdim=True)
        logp = _tanh_squash_log_prob(u, log_prob_u)
        return delta, logp, h_last


@dataclass(frozen=True)
class AlphaSchedule:
    alpha_start: float = 0.0
    alpha_mid: float = 0.2
    alpha_final: float = 1.0
    # step at which we reach alpha_mid and alpha_final
    step_mid: int = 100_000
    step_final: int = 300_000

    def value(self, global_step: int) -> float:
        t = int(global_step)
        if t <= 0:
            return float(self.alpha_start)
        if t < int(self.step_mid):
            # start -> mid
            r = t / max(1, int(self.step_mid))
            return float(self.alpha_start + r * (self.alpha_mid - self.alpha_start))
        if t < int(self.step_final):
            # mid -> final
            r = (t - int(self.step_mid)) / max(1, int(self.step_final) - int(self.step_mid))
            return float(self.alpha_mid + r * (self.alpha_final - self.alpha_mid))
        return float(self.alpha_final)


@dataclass(frozen=True)
class DisturbanceSpec:
    name: str
    dropout_level: str | None = None
    dropout_p: float | None = None
    k_obs: int = 0
    k_act: int = 0
    bias_b: float = 0.0


@dataclass(frozen=True)
class DisturbanceCurriculum:
    """
    4-stage curriculum: none -> easy -> med -> hard.
    Steps define boundaries where we SWITCH to the next stage.
    """

    step_easy: int = 200_000
    step_med: int = 400_000
    step_hard: int = 600_000

    # Max disturbances (hard stage)
    k_obs_max: int = 0
    k_act_max: int = 0
    bias_b_max: float = 0.0
    # For dropout, user can either use a level schedule or provide max p directly.
    use_dropout_p: bool = False
    dropout_p_max: float = 0.0

    def _scale_int(self, k_max: int, frac: float) -> int:
        k_max = int(max(0, k_max))
        if k_max == 0:
            return 0
        return int(max(1, round(frac * k_max)))

    def _scale_float(self, x_max: float, frac: float) -> float:
        x_max = float(max(0.0, x_max))
        return float(frac * x_max)

    def spec(self, global_step: int) -> DisturbanceSpec:
        t = int(global_step)
        if t < int(self.step_easy):
            return DisturbanceSpec(name="none", dropout_level="none", dropout_p=0.0, k_obs=0, k_act=0, bias_b=0.0)
        if t < int(self.step_med):
            frac = 0.33
            if self.use_dropout_p:
                dp = self._scale_float(self.dropout_p_max, frac)
                return DisturbanceSpec(
                    name="easy",
                    dropout_level=None,
                    dropout_p=dp,
                    k_obs=self._scale_int(self.k_obs_max, frac),
                    k_act=self._scale_int(self.k_act_max, frac),
                    bias_b=self._scale_float(self.bias_b_max, frac),
                )
            return DisturbanceSpec(
                name="easy",
                dropout_level="easy",
                dropout_p=None,
                k_obs=self._scale_int(self.k_obs_max, frac),
                k_act=self._scale_int(self.k_act_max, frac),
                bias_b=self._scale_float(self.bias_b_max, frac),
            )
        if t < int(self.step_hard):
            frac = 0.66
            if self.use_dropout_p:
                dp = self._scale_float(self.dropout_p_max, frac)
                return DisturbanceSpec(
                    name="med",
                    dropout_level=None,
                    dropout_p=dp,
                    k_obs=self._scale_int(self.k_obs_max, frac),
                    k_act=self._scale_int(self.k_act_max, frac),
                    bias_b=self._scale_float(self.bias_b_max, frac),
                )
            return DisturbanceSpec(
                name="med",
                dropout_level="med",
                dropout_p=None,
                k_obs=self._scale_int(self.k_obs_max, frac),
                k_act=self._scale_int(self.k_act_max, frac),
                bias_b=self._scale_float(self.bias_b_max, frac),
            )

        # hard
        if self.use_dropout_p:
            return DisturbanceSpec(
                name="hard",
                dropout_level=None,
                dropout_p=float(self.dropout_p_max),
                k_obs=int(max(0, self.k_obs_max)),
                k_act=int(max(0, self.k_act_max)),
                bias_b=float(max(0.0, self.bias_b_max)),
            )
        return DisturbanceSpec(
            name="hard",
            dropout_level="hard",
            dropout_p=None,
            k_obs=int(max(0, self.k_obs_max)),
            k_act=int(max(0, self.k_act_max)),
            bias_b=float(max(0.0, self.bias_b_max)),
        )


def build_residual_input(
    *,
    obs: dict,
    obj_idx: list[int],
    fourier: FourierFeatures,
    a_prev: np.ndarray,
    obs_mask: Optional[np.ndarray],
    device: th.device,
) -> th.Tensor:
    """
    Build x_t for residual controller from a single env obs dict (numpy) + prev action.

    x_t = concat([Fourier(obj_dims), rest_obs, achieved_goal, desired_goal, a_prev, mask_obj])
    """
    o = np.asarray(obs["observation"], dtype=np.float32).reshape(-1)
    ag = np.asarray(obs.get("achieved_goal", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    dg = np.asarray(obs.get("desired_goal", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)

    idx = np.asarray(obj_idx, dtype=int)
    obj = o[idx]
    rest = np.delete(o, idx) if o.shape[0] > 0 else o

    m = None
    if obs_mask is not None:
        mm = np.asarray(obs_mask, dtype=np.float32).reshape(-1)
        if mm.shape[0] == o.shape[0]:
            m = mm[idx]
    if m is None:
        m = np.ones((len(obj_idx),), dtype=np.float32)

    a_prev = np.asarray(a_prev, dtype=np.float32).reshape(-1)

    # torch
    obj_t = th.as_tensor(obj, device=device).unsqueeze(0)  # (1, obj_dim)
    obj_ff = fourier(obj_t)  # (1, ff_dim)
    rest_t = th.as_tensor(rest, device=device).unsqueeze(0)
    ag_t = th.as_tensor(ag, device=device).unsqueeze(0)
    dg_t = th.as_tensor(dg, device=device).unsqueeze(0)
    ap_t = th.as_tensor(a_prev, device=device).unsqueeze(0)
    m_t = th.as_tensor(m, device=device).unsqueeze(0)
    x = th.cat([obj_ff, rest_t, ag_t, dg_t, ap_t, m_t], dim=-1)  # (1, input_dim)
    return x


