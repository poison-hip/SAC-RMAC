"""
Episode-based replay buffer that supports contiguous sequence sampling with burn-in.

This is designed for training recurrent residual policies (GRU/CfC) off-policy.

We keep implementation intentionally simple and robust:
- supports n_envs >= 1 (each env has its own episode stream)
- store transitions grouped by episode
- sample contiguous fragments within a single episode
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Transition:
    obs: dict
    action: np.ndarray
    reward: float
    done: bool
    next_obs: dict
    obs_mask: Optional[np.ndarray] = None


class Episode:
    def __init__(self):
        self.tr: list[Transition] = []

    def __len__(self) -> int:
        return len(self.tr)


class SequenceReplayBuffer:
    def __init__(self, max_transitions: int = 1_000_000, seed: int = 0, n_envs: int = 1):
        self.max_transitions = int(max_transitions)
        self._rng = np.random.default_rng(int(seed))
        self._episodes: list[Episode] = []
        self.n_envs = int(n_envs)
        if self.n_envs <= 0:
            raise ValueError("n_envs must be >= 1")
        self._cur: list[Episode] = [Episode() for _ in range(self.n_envs)]
        self._n = 0

    @property
    def size(self) -> int:
        return int(self._n)

    def add(
        self,
        *,
        env_idx: int = 0,
        obs: dict,
        action: np.ndarray,
        reward: float,
        done: bool,
        next_obs: dict,
        obs_mask: Optional[np.ndarray] = None,
    ):
        i = int(env_idx)
        if i < 0 or i >= self.n_envs:
            raise IndexError(f"env_idx out of range: {env_idx} (n_envs={self.n_envs})")

        self._cur[i].tr.append(
            Transition(
                obs=obs,
                action=np.asarray(action, dtype=np.float32).copy(),
                reward=float(reward),
                done=bool(done),
                next_obs=next_obs,
                obs_mask=None if obs_mask is None else np.asarray(obs_mask, dtype=np.float32).copy(),
            )
        )
        self._n += 1

        if bool(done):
            # finalize this env's episode stream
            self._episodes.append(self._cur[i])
            self._cur[i] = Episode()

        # evict old episodes if over max_transitions
        while self._n > self.max_transitions and self._episodes:
            old = self._episodes.pop(0)
            self._n -= len(old)

    def _eligible_episodes(self, total_len: int) -> list[Episode]:
        eps = [ep for ep in self._episodes if len(ep) >= total_len]
        # current episodes can be sampled too (unfinished)
        for cur in self._cur:
            if len(cur) >= total_len:
                eps.append(cur)
        return eps

    def sample_sequences(self, *, batch_size: int, seq_len: int, burn_in: int):
        """
        Sample sequences with burn-in.

        Returns a dict of numpy arrays:
          - obs:      list/dict of length L+1 (time-major), each entry is a dict obs (numpy)
          - actions:  (B, L, act_dim)
          - rewards:  (B, L)
          - dones:    (B, L)
          - masks:    (B, L+1, obs_dim) or None
        where L = burn_in + seq_len.
        """
        b = int(batch_size)
        L = int(burn_in) + int(seq_len)
        if L <= 0:
            raise ValueError("burn_in+seq_len must be > 0")

        eps = self._eligible_episodes(total_len=L)
        if not eps:
            raise RuntimeError("Not enough data to sample sequences yet.")

        # We will build time-major obs dict lists first (easy to keep dict structure).
        obs_seq: list[list[dict]] = [[] for _ in range(b)]  # per batch: list of dicts length L+1
        actions = None
        rewards = np.zeros((b, L), dtype=np.float32)
        dones = np.zeros((b, L), dtype=np.float32)
        masks = None

        for i in range(b):
            ep = eps[int(self._rng.integers(0, len(eps)))]
            start = int(self._rng.integers(0, len(ep) - L + 1))
            frag = ep.tr[start : start + L]  # length L transitions

            # obs_t for t=0..L-1 from transition.obs; obs_L from last transition.next_obs
            obs_list = [frag[t].obs for t in range(L)]
            obs_list.append(frag[-1].next_obs)
            obs_seq[i] = obs_list

            # actions/rewards/dones from transitions
            act_i = np.stack([frag[t].action for t in range(L)], axis=0).astype(np.float32, copy=False)
            if actions is None:
                actions = np.zeros((b, L, act_i.shape[-1]), dtype=np.float32)
            actions[i] = act_i

            rewards[i] = np.asarray([frag[t].reward for t in range(L)], dtype=np.float32)
            dones[i] = np.asarray([1.0 if frag[t].done else 0.0 for t in range(L)], dtype=np.float32)

            # optional mask aligned with obs["observation"]
            if frag[0].obs_mask is not None:
                if masks is None:
                    masks = np.zeros((b, L + 1, frag[0].obs_mask.shape[0]), dtype=np.float32)
                for t in range(L):
                    if frag[t].obs_mask is not None:
                        masks[i, t] = frag[t].obs_mask
                    else:
                        masks[i, t] = 1.0
                masks[i, L] = 1.0  # unknown for next_obs; treat as valid

        assert actions is not None
        return {
            "obs_seq": obs_seq,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "masks": masks,
        }


