"""
Robustness evaluation sweeps for perturbations:

A) ObsDropoutWrapper: success vs dropout probability p
B) DelayWrapper: success vs delay k (action delay and/or obs delay)
C) ObsBiasWrapper: success vs bias magnitude b

This script loads a trained SB3 model (default: SAC) and evaluates success rate
under different perturbation settings, saving plots + CSV under docs/.
"""

from __future__ import annotations

# Make repo root importable when running as: python scripts/eval/robustness_eval.py
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import csv
import json
import os
import re
import zipfile
from dataclasses import dataclass
from typing import Iterable

import gymnasium as gym
import gymnasium_robotics
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from stable_baselines3 import DDPG, TD3, SAC  # noqa: E402

from perturbation_wrappers import build_perturbed_env, parse_indices  # noqa: E402
from delay_compensation import HistoryObsActWrapper  # noqa: E402
from conservative_wrappers import MotorCostConfig, MotorCostWrapper, predict_with_uncertainty_gating  # noqa: E402


MODEL_CLASSES = {"DDPG": DDPG, "TD3": TD3, "SAC": SAC}


@dataclass(frozen=True)
class SweepMetrics:
    success_rate: float
    energy_per_step: float = 0.0
    jerk_per_step: float = 0.0
    gate_mean: float = 1.0
    uncertainty_mean: float = 0.0


def _parse_floats(csv_list: str) -> list[float]:
    return [float(x.strip()) for x in csv_list.split(",") if x.strip()]


def _parse_ints(csv_list: str) -> list[int]:
    return [int(x.strip()) for x in csv_list.split(",") if x.strip()]


def _select_model_zip(log_root: str, env_name: str, algo: str, run_dir: str | None) -> str:
    """
    Select a model zip path from logs.

    Preference order:
    1) If run_dir is provided: choose best_model.zip if present else <env_name>.zip
    2) Otherwise: choose the latest run that contains best_model.zip
    3) Otherwise: choose the latest run (even if it doesn't have best_model.zip)
    """
    log_dir = os.path.join(log_root, env_name)
    prefix = f"{algo}_"
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    runs = [d for d in os.listdir(log_dir) if d.startswith(prefix)]
    if not runs:
        raise FileNotFoundError(f"No {algo} runs found in {log_dir}")
    runs = sorted(runs, reverse=True)

    def pick_from_run(run: str) -> str:
        run_path = os.path.join(log_dir, run)
        best_path = os.path.join(run_path, "best_model.zip")
        last_path = os.path.join(run_path, f"{env_name}.zip")
        if os.path.exists(best_path):
            return best_path
        if os.path.exists(last_path):
            return last_path
        raise FileNotFoundError(
            f"Run exists but no model zip found in {run_path} (expected best_model.zip or {env_name}.zip)"
        )

    if run_dir is not None:
        if run_dir not in runs:
            raise FileNotFoundError(
                f"Requested run_dir '{run_dir}' not found under {log_dir}. Available: {runs[:10]}"
            )
        return pick_from_run(run_dir)

    for run in runs:
        if os.path.exists(os.path.join(log_dir, run, "best_model.zip")):
            return pick_from_run(run)
    return pick_from_run(runs[0])


def evaluate_metrics(
    model,
    env: gym.Env,
    n_episodes: int,
    seed: int,
    *,
    uncertainty_gating: bool = False,
    uncertainty_threshold: float = 1.0,
    uncertainty_alpha: float = 1.0,
    uncertainty_min_g: float = 0.0,
) -> SweepMetrics:
    successes: list[float] = []
    total_energy = 0.0
    total_jerk = 0.0
    total_steps = 0
    total_g = 0.0
    total_u = 0.0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            if uncertainty_gating:
                action, _state, diag = predict_with_uncertainty_gating(
                    model,
                    obs,
                    deterministic=True,
                    threshold=float(uncertainty_threshold),
                    alpha=float(uncertainty_alpha),
                    min_g=float(uncertainty_min_g),
                )
            else:
                action, _ = model.predict(obs, deterministic=True)
                diag = None

            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_steps += 1

            if isinstance(info, dict):
                total_energy += float(info.get("motor_energy", 0.0))
                total_jerk += float(info.get("motor_jerk", 0.0))

            if diag is not None:
                total_g += float(diag.get("g", 1.0))
                total_u += float(diag.get("uncertainty", 0.0))
            else:
                total_g += 1.0

        successes.append(float(info.get("is_success", 0.0)))

    sr = float(np.mean(successes)) if successes else 0.0
    denom = float(total_steps) if total_steps > 0 else 1.0
    return SweepMetrics(
        success_rate=sr,
        energy_per_step=float(total_energy / denom),
        jerk_per_step=float(total_jerk / denom),
        gate_mean=float(total_g / denom),
        uncertainty_mean=float(total_u / denom) if uncertainty_gating else 0.0,
    )


def _ensure_out_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _write_csv(path: str, header: list[str], rows: Iterable[list[object]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def _plot_xy(path: str, x: list[float], y: list[float], xlabel: str, title: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel("Success rate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description="Run robustness sweeps and save success-rate plots.")
    p.add_argument("--algo", type=str, default="SAC", choices=["DDPG", "TD3", "SAC"])
    p.add_argument("--env", type=str, default="FetchPickAndPlace-v4")
    p.add_argument("--log-dir", type=str, default="./logs", help="Base logs directory (contains <env>/...).")
    p.add_argument("--run-dir", type=str, default=None, help="Run directory name under logs/<env>/ (optional).")
    p.add_argument("--model-path", type=str, default=None, help="Explicit model .zip path (overrides logs selection).")
    p.add_argument("--episodes", type=int, default=50, help="Episodes per sweep point.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out-dir", type=str, default="./docs", help="Output directory for plots/CSVs.")

    # Delay-aware history wrapper (must match how the model was trained)
    p.add_argument(
        "--history-k",
        type=int,
        default=None,
        help=(
            "History length K for delay-aware input. If omitted, robustness_eval.py will try to infer it "
            "from the loaded model's observation dim."
        ),
    )

    # --- motor-friendly cost (Module D) ---
    p.add_argument(
        "--log-motor-metrics",
        action="store_true",
        help="If set, wrap env to log motor_energy/motor_jerk per step in info, and save them to CSV alongside success rate.",
    )
    p.add_argument("--motor-energy-w", type=float, default=0.0)
    p.add_argument("--motor-jerk-w", type=float, default=0.0)
    p.add_argument("--motor-cost-in-reward", action="store_true")

    # --- uncertainty-aware action gating (Module E/F) ---
    p.add_argument("--uncertainty-gating", action="store_true")
    p.add_argument("--uncertainty-threshold", type=float, default=1.0)
    p.add_argument("--uncertainty-alpha", type=float, default=1.0)
    p.add_argument("--uncertainty-min-g", type=float, default=0.0)

    # Dropout sweep
    p.add_argument("--sweep-dropout", action="store_true", help="Run dropout sweep (success vs p).")
    p.add_argument("--dropout-ps", type=str, default="0,0.1,0.3,0.5", help="Comma-separated p list.")
    p.add_argument("--dropout-mode", type=str, default="hold-last", choices=["drop-to-zero", "hold-last"])
    p.add_argument("--dropout-keys", type=str, default="observation,achieved_goal")
    p.add_argument("--dropout-obj-idx", type=str, default=None)
    p.add_argument("--dropout-exclude-gripper", action="store_true")

    # Delay sweep
    p.add_argument("--sweep-action-delay", action="store_true", help="Run action-delay sweep (success vs k).")
    p.add_argument("--sweep-obs-delay", action="store_true", help="Run obs-delay sweep (success vs k).")
    p.add_argument("--delay-ks", type=str, default="0,1,2,3", help="Comma-separated k list.")

    # Bias sweep
    p.add_argument("--sweep-bias", action="store_true", help="Run bias sweep (success vs b).")
    p.add_argument("--bias-bs", type=str, default="0,0.01,0.02,0.05", help="Comma-separated b list (meters).")
    p.add_argument("--bias-keys", type=str, default="observation,achieved_goal")
    p.add_argument("--bias-obj-idx", type=str, default=None)

    return p.parse_args()


def _infer_history_k(model_obs_dim: int, env_obs_dim: int, act_dim: int) -> int | None:
    """
    Solve: model_obs_dim = env_obs_dim + k * (env_obs_dim + act_dim)
    where env_obs_dim is the base obs['observation'] dim (no history).
    """
    if model_obs_dim == env_obs_dim:
        return 0
    num = model_obs_dim - env_obs_dim
    den = env_obs_dim + act_dim
    if num <= 0 or den <= 0 or num % den != 0:
        return None
    k = int(num // den)
    return k if k >= 0 else None


def _read_model_obs_dim(model_zip_path: str) -> int | None:
    """Extract obs['observation'] dim from SB3 .zip without instantiating the model (HER requires env)."""
    try:
        with zipfile.ZipFile(model_zip_path, "r") as z:
            data = json.loads(z.read("data").decode("utf-8"))
        spaces_str = data.get("observation_space", {}).get("spaces", "")
        m = re.search(r"'observation':\s*Box\([^\)]*\(\s*(\d+)\s*,\s*\)", str(spaces_str))
        return int(m.group(1)) if m else None
    except Exception:
        return None


def main():
    args = parse_args()

    gym.register_envs(gymnasium_robotics)

    model_path = args.model_path or _select_model_zip(args.log_dir, args.env, args.algo, args.run_dir)
    print(f"Model zip: {model_path}")
    model_class = MODEL_CLASSES[args.algo]

    # Infer history_k once (unless user pinned it).
    if args.history_k is not None:
        hist_k = int(args.history_k)
    else:
        probe_env = gym.make(args.env, render_mode=None)
        try:
            model_obs_dim = _read_model_obs_dim(model_path) or 0
            env_obs_dim = int(probe_env.observation_space.spaces["observation"].shape[0])
            act_dim = int(probe_env.action_space.shape[0])
            hist_k = _infer_history_k(model_obs_dim, env_obs_dim, act_dim) or 0
        except Exception:
            hist_k = 0
        finally:
            probe_env.close()
    if hist_k > 0:
        print(f"Using history_k={hist_k} (delay-aware observation augmentation)")

    # Load model with an env (HER requires env at load time). Use a base env with matching observation space.
    load_env = gym.make(args.env, render_mode=None)
    if hist_k > 0:
        load_env = HistoryObsActWrapper(load_env, history_k=hist_k)
    model = model_class.load(model_path, env=load_env)

    out_dir = _ensure_out_dir(args.out_dir)

    # If user didn't specify any sweeps, run all.
    run_any = args.sweep_dropout or args.sweep_action_delay or args.sweep_obs_delay or args.sweep_bias
    if not run_any:
        args.sweep_dropout = True
        args.sweep_action_delay = True
        args.sweep_obs_delay = True
        args.sweep_bias = True

    if args.sweep_dropout:
        ps = _parse_floats(args.dropout_ps)
        ys: list[float] = []
        es: list[float] = []
        js: list[float] = []
        gs: list[float] = []
        us: list[float] = []
        dropout_obj_idx = parse_indices(args.dropout_obj_idx)
        dropout_keys = tuple([k.strip() for k in args.dropout_keys.split(",") if k.strip()])
        for pval in ps:
            env = gym.make(args.env, render_mode=None)
            env = build_perturbed_env(
                env,
                dropout_p=pval,
                dropout_mode=args.dropout_mode,
                dropout_keys=dropout_keys,
                dropout_exclude_gripper=args.dropout_exclude_gripper,
                dropout_obj_idx=dropout_obj_idx,
            )
            if hist_k > 0:
                env = HistoryObsActWrapper(env, history_k=hist_k)
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
            m = evaluate_metrics(
                model,
                env,
                args.episodes,
                args.seed,
                uncertainty_gating=bool(args.uncertainty_gating),
                uncertainty_threshold=float(args.uncertainty_threshold),
                uncertainty_alpha=float(args.uncertainty_alpha),
                uncertainty_min_g=float(args.uncertainty_min_g),
            )
            env.close()
            ys.append(m.success_rate)
            es.append(m.energy_per_step)
            js.append(m.jerk_per_step)
            gs.append(m.gate_mean)
            us.append(m.uncertainty_mean)
            print(
                f"[dropout] p={pval:.3f} -> success={m.success_rate:.3f}, "
                f"E={m.energy_per_step:.4f}, J={m.jerk_per_step:.4f}, g={m.gate_mean:.3f}, u={m.uncertainty_mean:.3f}"
            )

        csv_path = os.path.join(out_dir, f"success_vs_dropout_p_{args.dropout_mode}.csv")
        _write_csv(
            csv_path,
            ["p", "success_rate", "energy_per_step", "jerk_per_step", "gate_mean", "uncertainty_mean"],
            [[p, y, e, j, g, u] for p, y, e, j, g, u in zip(ps, ys, es, js, gs, us)],
        )
        fig_path = os.path.join(out_dir, f"success_vs_dropout_p_{args.dropout_mode}.png")
        _plot_xy(fig_path, ps, ys, xlabel="dropout probability p", title=f"Success vs dropout p ({args.dropout_mode})")

    ks = _parse_ints(args.delay_ks)
    if args.sweep_action_delay:
        ys = []
        es = []
        js = []
        gs = []
        us = []
        for kval in ks:
            env = gym.make(args.env, render_mode=None)
            env = build_perturbed_env(env, action_delay_k=kval)
            if hist_k > 0:
                env = HistoryObsActWrapper(env, history_k=hist_k)
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
            m = evaluate_metrics(
                model,
                env,
                args.episodes,
                args.seed,
                uncertainty_gating=bool(args.uncertainty_gating),
                uncertainty_threshold=float(args.uncertainty_threshold),
                uncertainty_alpha=float(args.uncertainty_alpha),
                uncertainty_min_g=float(args.uncertainty_min_g),
            )
            env.close()
            ys.append(m.success_rate)
            es.append(m.energy_per_step)
            js.append(m.jerk_per_step)
            gs.append(m.gate_mean)
            us.append(m.uncertainty_mean)
            print(
                f"[action-delay] k={kval} -> success={m.success_rate:.3f}, "
                f"E={m.energy_per_step:.4f}, J={m.jerk_per_step:.4f}, g={m.gate_mean:.3f}, u={m.uncertainty_mean:.3f}"
            )

        csv_path = os.path.join(out_dir, "success_vs_action_delay_k.csv")
        _write_csv(
            csv_path,
            ["k", "success_rate", "energy_per_step", "jerk_per_step", "gate_mean", "uncertainty_mean"],
            [[k, y, e, j, g, u] for k, y, e, j, g, u in zip(ks, ys, es, js, gs, us)],
        )
        fig_path = os.path.join(out_dir, "success_vs_action_delay_k.png")
        _plot_xy(fig_path, [float(k) for k in ks], ys, xlabel="action delay k (steps)", title="Success vs action delay k")

    if args.sweep_obs_delay:
        ys = []
        es = []
        js = []
        gs = []
        us = []
        for kval in ks:
            env = gym.make(args.env, render_mode=None)
            env = build_perturbed_env(env, obs_delay_k=kval)
            if hist_k > 0:
                env = HistoryObsActWrapper(env, history_k=hist_k)
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
            m = evaluate_metrics(
                model,
                env,
                args.episodes,
                args.seed,
                uncertainty_gating=bool(args.uncertainty_gating),
                uncertainty_threshold=float(args.uncertainty_threshold),
                uncertainty_alpha=float(args.uncertainty_alpha),
                uncertainty_min_g=float(args.uncertainty_min_g),
            )
            env.close()
            ys.append(m.success_rate)
            es.append(m.energy_per_step)
            js.append(m.jerk_per_step)
            gs.append(m.gate_mean)
            us.append(m.uncertainty_mean)
            print(
                f"[obs-delay] k={kval} -> success={m.success_rate:.3f}, "
                f"E={m.energy_per_step:.4f}, J={m.jerk_per_step:.4f}, g={m.gate_mean:.3f}, u={m.uncertainty_mean:.3f}"
            )

        csv_path = os.path.join(out_dir, "success_vs_obs_delay_k.csv")
        _write_csv(
            csv_path,
            ["k", "success_rate", "energy_per_step", "jerk_per_step", "gate_mean", "uncertainty_mean"],
            [[k, y, e, j, g, u] for k, y, e, j, g, u in zip(ks, ys, es, js, gs, us)],
        )
        fig_path = os.path.join(out_dir, "success_vs_obs_delay_k.png")
        _plot_xy(fig_path, [float(k) for k in ks], ys, xlabel="obs delay k (steps)", title="Success vs observation delay k")

    if args.sweep_bias:
        bs = _parse_floats(args.bias_bs)
        ys = []
        es = []
        js = []
        gs = []
        us = []
        bias_obj_idx = parse_indices(args.bias_obj_idx)
        bias_keys = tuple([k.strip() for k in args.bias_keys.split(",") if k.strip()])
        for bval in bs:
            env = gym.make(args.env, render_mode=None)
            env = build_perturbed_env(env, bias_b=bval, bias_keys=bias_keys, bias_obj_idx=bias_obj_idx)
            if hist_k > 0:
                env = HistoryObsActWrapper(env, history_k=hist_k)
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
            m = evaluate_metrics(
                model,
                env,
                args.episodes,
                args.seed,
                uncertainty_gating=bool(args.uncertainty_gating),
                uncertainty_threshold=float(args.uncertainty_threshold),
                uncertainty_alpha=float(args.uncertainty_alpha),
                uncertainty_min_g=float(args.uncertainty_min_g),
            )
            env.close()
            ys.append(m.success_rate)
            es.append(m.energy_per_step)
            js.append(m.jerk_per_step)
            gs.append(m.gate_mean)
            us.append(m.uncertainty_mean)
            print(
                f"[bias] b={bval:.3f} -> success={m.success_rate:.3f}, "
                f"E={m.energy_per_step:.4f}, J={m.jerk_per_step:.4f}, g={m.gate_mean:.3f}, u={m.uncertainty_mean:.3f}"
            )

        csv_path = os.path.join(out_dir, "success_vs_bias_b.csv")
        _write_csv(
            csv_path,
            ["b", "success_rate", "energy_per_step", "jerk_per_step", "gate_mean", "uncertainty_mean"],
            [[b, y, e, j, g, u] for b, y, e, j, g, u in zip(bs, ys, es, js, gs, us)],
        )
        fig_path = os.path.join(out_dir, "success_vs_bias_b.png")
        _plot_xy(fig_path, bs, ys, xlabel="bias magnitude b (m)", title="Success vs observation bias b")

    try:
        load_env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()



