"""
Evaluate a trained "Pretrained SAC + Residual Memory" run.

This run saves:
  - base_sac.zip         (SB3 SAC model; actor may be frozen, critic trained)
  - residual_mem.pt      (PyTorch state_dict for ResidualMemPolicy)
  - residual_mem_meta.yaml (Fourier/residual settings)

We evaluate in a standard Gymnasium-Robotics dict-observation env and compute success rate.
"""

from __future__ import annotations

# Make repo root importable when running as: python scripts/eval/eval_residual_mem.py
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import os
from pathlib import Path as _Path
from typing import Any

import numpy as np

import gymnasium as gym
import gymnasium_robotics

import torch as th

from stable_baselines3 import SAC

from perturbation_wrappers import build_perturbed_env, parse_indices
from residual_mem_modules import FourierFeatures, ResidualMemPolicy, build_residual_input


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate residual-mem SAC run.")
    p.add_argument("--env", type=str, default="FetchPickAndPlace-v4")
    p.add_argument("--run-dir", type=str, required=True, help="Run directory path or run name under ./logs/<env>/")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)

    # disturbances (match training)
    p.add_argument("--action-delay-k", type=int, default=1)
    p.add_argument("--obs-delay-k", type=int, default=0)
    p.add_argument("--obs-bias-b", type=float, default=0.0)
    p.add_argument("--obs-dropout-level", type=str, default="none", choices=["none", "easy", "med", "hard"])
    p.add_argument("--obs-dropout-p", type=float, default=None)
    p.add_argument("--obs-dropout-mode", type=str, default="hold-last", choices=["drop-to-zero", "hold-last"])
    p.add_argument("--obs-dropout-keys", type=str, default="observation")
    p.add_argument("--obs-dropout-obj-idx", type=str, default=None)
    p.add_argument("--obs-dropout-exclude-gripper", action="store_true")
    p.add_argument("--no-render", action="store_true", help="Disable rendering (ignored, default is no render).")
    p.add_argument("--obs-bias-keys", type=str, default="observation")
    p.add_argument("--obs-bias-obj-idx", type=str, default=None)

    # residual eval knobs
    p.add_argument("--alpha", type=float, default=1.0, help="Residual scale during evaluation.")
    p.add_argument("--deterministic-residual", action="store_true", help="Use deterministic residual (mu) instead of sampling.")
    p.add_argument(
        "--quiet",
        action="store_true",
        help="If set, do not print per-episode logs (recommended for large --episodes).",
    )
    return p.parse_args()


def _resolve_run_dir(env_id: str, run_dir_arg: str) -> _Path:
    p = _Path(os.path.expanduser(run_dir_arg))
    if p.exists():
        return p
    # treat as run name under ./logs/<env>/
    cand = _Path("./logs") / env_id / run_dir_arg
    if cand.exists():
        return cand
    raise FileNotFoundError(f"Run dir not found: {run_dir_arg} (checked {p} and {cand})")


def _load_meta(run_dir: _Path) -> dict[str, Any]:
    meta_path = run_dir / "residual_mem_meta.yaml"
    if not meta_path.exists():
        return {}
    import yaml

    with open(meta_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return dict(data)


def main():
    args = parse_args()
    gym.register_envs(gymnasium_robotics)

    run_dir = _resolve_run_dir(args.env, args.run_dir)
    base_path = run_dir / "base_sac.zip"
    residual_path = run_dir / "residual_mem.pt"
    if not base_path.exists():
        raise FileNotFoundError(f"Missing: {base_path}")
    if not residual_path.exists():
        raise FileNotFoundError(f"Missing: {residual_path}")

    meta = _load_meta(run_dir)
    obj_idx = meta.get("obj_idx", None)
    if obj_idx is None:
        # fallback: infer from pt file (we saved obj_idx there too)
        ckpt = th.load(residual_path, map_location="cpu")
        obj_idx = ckpt.get("obj_idx", [3, 4, 5])
    obj_idx = [int(x) for x in obj_idx]

    fourier_cfg = meta.get("fourier", {}) or {}
    fourier_bands = int(fourier_cfg.get("bands", 8))
    fourier_max_freq = float(fourier_cfg.get("max_freq", 10.0))

    # Build env with the evaluation disturbances
    env = gym.make(args.env)
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
        # ensure info.get_disturbance is safe (not required, but useful for debugging)
        enable_disturbance_control=True,
        force_wrappers=True,
    )

    # Load base SAC on the env
    model = SAC.load(str(base_path), env=env, device="auto")
    device = model.device

    # Rebuild residual network matching saved weights
    obs_space = env.observation_space
    base_obs_dim = int(obs_space["observation"].shape[0])
    ag_dim = int(obs_space["achieved_goal"].shape[0])
    dg_dim = int(obs_space["desired_goal"].shape[0])
    act_dim = int(env.action_space.shape[0])
    obj_dim = len(obj_idx)
    rest_dim = base_obs_dim - obj_dim

    fourier = FourierFeatures(obj_dim, num_bands=fourier_bands, max_freq=fourier_max_freq).to(device)
    input_dim = int(fourier.out_dim + rest_dim + ag_dim + dg_dim + act_dim + obj_dim)

    # infer hidden size from checkpoint
    ckpt = th.load(residual_path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    # GRU weight_ih_l0: (3H, input_dim)
    wih = sd.get("rnn.weight_ih_l0", None)
    if wih is None:
        raise RuntimeError("Could not infer hidden size (missing rnn.weight_ih_l0).")
    hidden = int(wih.shape[0] // 3)

    residual = ResidualMemPolicy(
        input_dim=input_dim,
        action_dim=act_dim,
        hidden_size=hidden,
        mem_type="gru",
    ).to(device)
    residual.load_state_dict(sd, strict=True)
    residual.eval()

    alpha = float(args.alpha)
    successes = 0
    prev_action = np.zeros((act_dim,), dtype=np.float32)
    h = residual.init_hidden(1, device=device)

    for ep in range(int(args.episodes)):
        obs, info = env.reset(seed=int(args.seed) + ep)
        prev_action[:] = 0.0
        h[:] = 0.0

        done = False
        while not done:
            # base mean action
            a0, _ = model.predict(obs, deterministic=True)
            a0 = np.asarray(a0, dtype=np.float32).reshape(-1)

            obs_mask = None
            if isinstance(info, dict) and "obs_mask" in info:
                obs_mask = np.asarray(info["obs_mask"], dtype=np.float32)

            x = build_residual_input(
                obs=obs,
                obj_idx=obj_idx,
                fourier=fourier,
                a_prev=prev_action,
                obs_mask=obs_mask,
                device=device,
            )  # (1, input_dim)
            x_seq = x.unsqueeze(1)  # (1,1,input_dim)
            with th.no_grad():
                delta, _, h = residual.forward_sequence(x_seq, h, deterministic=bool(args.deterministic_residual))
            delta = delta.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

            # scale delta to env action units
            low = env.action_space.low.astype(np.float32)
            high = env.action_space.high.astype(np.float32)
            action_scale = 0.5 * (high - low)

            act = a0 + alpha * (delta * action_scale)
            act = np.clip(act, low, high).astype(np.float32)

            obs, reward, terminated, truncated, info = env.step(act)
            done = bool(terminated or truncated)

            # use executed action if delay wrapper is active
            if isinstance(info, dict) and "executed_action" in info:
                prev_action = np.asarray(info["executed_action"], dtype=np.float32).reshape(-1)
            else:
                prev_action = act

        is_success = 0.0
        if isinstance(info, dict) and "is_success" in info:
            try:
                is_success = float(info["is_success"])
            except Exception:
                is_success = 0.0
        successes += int(is_success >= 0.5)

        if not bool(args.quiet):
            print(f"[eval] ep={ep+1}/{args.episodes} success={is_success}", flush=True)

    sr = successes / max(1, int(args.episodes))
    print(f"[eval] run_dir={run_dir}")
    print(f"[eval] success_rate={sr:.4f} ({successes}/{args.episodes})")

    out_dir = run_dir / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_residual_mem.json"
    import json

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "episodes": int(args.episodes),
                "successes": int(successes),
                "success_rate": float(sr),
                "alpha": float(alpha),
                "deterministic_residual": bool(args.deterministic_residual),
                "disturbance": {
                    "action_delay_k": int(args.action_delay_k),
                    "obs_delay_k": int(args.obs_delay_k),
                    "obs_bias_b": float(args.obs_bias_b),
                    "obs_dropout_level": str(args.obs_dropout_level),
                    "obs_dropout_p": None if args.obs_dropout_p is None else float(args.obs_dropout_p),
                },
            },
            f,
            indent=2,
        )
    print(f"[eval] wrote: {out_path}")


if __name__ == "__main__":
    main()



