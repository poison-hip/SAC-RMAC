"""
Training loop for: Pretrained SAC + Residual Memory + Fourier Features + Dual Curriculum.

This is intentionally kept separate from train.py to minimize code churn.
train.py wires CLI args + env construction, then delegates here when enabled.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Optional

import numpy as np

import gymnasium as gym
import gymnasium_robotics

try:
    import torch as th
    import torch.nn.functional as F
    from torch.utils.tensorboard import SummaryWriter
except Exception as e:  # pragma: no cover
    raise ImportError("residual_mem_trainer.py requires PyTorch + tensorboard to be installed") from e

from stable_baselines3 import SAC

from experiment_logger import PaperLogger
from perturbation_wrappers import build_perturbed_env, parse_indices
from residual_mem_modules import (
    AlphaSchedule,
    DisturbanceCurriculum,
    FourierFeatures,
    ResidualMemPolicy,
)
from adversary_controller import BudgetedAdversary, AdversaryConfig
from sequence_replay_buffer import SequenceReplayBuffer


def _iter_wrapped_env(env):
    e = env
    while True:
        yield e
        if not hasattr(e, "env"):
            break
        e = getattr(e, "env")


def _maybe_set_disturbance(env, *, spec: dict[str, Any]):
    # VecEnv path: call on all sub-envs
    if hasattr(env, "env_method"):
        try:
            env.env_method("set_disturbance", **spec)
            return
        except Exception:
            pass
    for w in _iter_wrapped_env(env):
        if hasattr(w, "set_disturbance"):
            w.set_disturbance(**spec)
            return


def _maybe_get_disturbance(env) -> Optional[dict[str, Any]]:
    if hasattr(env, "env_method"):
        try:
            outs = env.env_method("get_disturbance")
            if outs:
                return outs[0]
        except Exception:
            pass
    for w in _iter_wrapped_env(env):
        if hasattr(w, "get_disturbance"):
            try:
                return w.get_disturbance()
            except Exception:
                return None
    return None


def _freeze_module(m, *, train_last_layer_only: bool = False):
    for p in m.parameters():
        p.requires_grad_(False)
    if not train_last_layer_only:
        return

    # Best-effort "half-freeze": unfreeze the last linear heads.
    # SB3 SAC actor usually exposes `mu` and `log_std` heads.
    for name in ("mu", "log_std"):
        head = getattr(m, name, None)
        if head is not None:
            for p in head.parameters():
                p.requires_grad_(True)


def _unscale_action_torch(action_bias: th.Tensor, action_scale: th.Tensor, a_scaled: th.Tensor) -> th.Tensor:
    return action_bias + action_scale * a_scaled


def _scale_action_torch(action_bias: th.Tensor, action_scale: th.Tensor, a_env: th.Tensor) -> th.Tensor:
    # avoid division by zero if bounds are degenerate
    denom = th.where(action_scale.abs() < 1e-8, th.ones_like(action_scale), action_scale)
    return (a_env - action_bias) / denom


def _paper_setting_to_delays(setting_id: str) -> tuple[int, int]:
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
        print(f"[residual][paper_eval][warn] Unknown setting_id='{setting_id}', using Nominal.", flush=True)
    return int(k_obs), int(k_act)


def _make_eval_env_residual(setting_id: str, *, seed: int, env_id: str) -> gym.Env:
    k_obs, k_act = _paper_setting_to_delays(setting_id)
    e = gym.make(env_id)
    # Match baseline paper-eval convention: only delays in these fixed settings; no dropout/bias
    e = build_perturbed_env(
        e,
        dropout_level="none",
        dropout_p=0.0,
        bias_b=0.0,
        action_delay_k=int(k_act),
        obs_delay_k=int(k_obs),
        force_wrappers=True,
        enable_disturbance_control=True,
    )
    e.reset(seed=int(seed))
    return e


def _as_batch_obs_dict(o: dict[str, Any]) -> dict[str, np.ndarray]:
    if not isinstance(o, dict):
        raise TypeError("Expected dict observation for Fetch environments.")
    out: dict[str, np.ndarray] = {}
    for k, v in o.items():
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        out[k] = arr
    return out


def _torch_obs_dict(o_batch: dict[str, np.ndarray], *, device) -> dict[str, th.Tensor]:
    return {k: th.as_tensor(v, device=device).float() for k, v in o_batch.items()}


def _compute_base_a0_env(
    *,
    base,
    device,
    action_bias: th.Tensor,
    action_scale: th.Tensor,
    o_batch: dict[str, np.ndarray],
) -> np.ndarray:
    obs_t = _torch_obs_dict(o_batch, device=device)
    with th.no_grad():
        a0_scaled = base.policy.actor(obs_t, deterministic=True)
        a0_env = _unscale_action_torch(action_bias, action_scale, a0_scaled)
    return a0_env.cpu().numpy().astype(np.float32, copy=False)


def _compute_residual_delta_eval(
    *,
    o_batch: dict[str, np.ndarray],
    info: dict[str, Any],
    h_in: th.Tensor,
    prev_action_eval: np.ndarray,
    deterministic: bool,
    device,
    ag_dim: int,
    dg_dim: int,
    base_obs_dim: int,
    obj_idx: list[int],
    obj_dim: int,
    fourier,
    residual,
    action_scale_np: np.ndarray,
) -> tuple[np.ndarray, th.Tensor]:
    o_obs = o_batch["observation"]  # (1, base_obs_dim)
    o_ag = o_batch.get("achieved_goal", np.zeros((1, ag_dim), dtype=np.float32))
    o_dg = o_batch.get("desired_goal", np.zeros((1, dg_dim), dtype=np.float32))
    keep = [i for i in range(base_obs_dim) if i not in set(obj_idx)]

    obj_np = o_obs[:, obj_idx].astype(np.float32, copy=False)
    rest_np = o_obs[:, keep].astype(np.float32, copy=False)

    m_obj = np.ones((1, obj_dim), dtype=np.float32)
    if isinstance(info, dict) and "obs_mask" in info:
        mm = np.asarray(info["obs_mask"], dtype=np.float32).reshape(-1)
        if mm.shape[0] == base_obs_dim:
            m_obj[0] = mm[np.asarray(obj_idx, dtype=int)]

    obj_t = th.as_tensor(obj_np, device=device).float()
    obj_ff = fourier(obj_t)
    rest_t = th.as_tensor(rest_np, device=device).float()
    ag_t = th.as_tensor(o_ag, device=device).float()
    dg_t = th.as_tensor(o_dg, device=device).float()
    ap_t = th.as_tensor(prev_action_eval.reshape(1, -1), device=device).float()
    m_t = th.as_tensor(m_obj, device=device).float()
    x = th.cat([obj_ff, rest_t, ag_t, dg_t, ap_t, m_t], dim=-1).unsqueeze(1)  # (1,1,input)

    with th.no_grad():
        delta, _, h_out = residual.forward_sequence(x, h_in, deterministic=bool(deterministic))
    delta_np = delta.squeeze(1).cpu().numpy().astype(np.float32, copy=False)  # (1, act_dim) in [-1,1]
    delta_env = delta_np * action_scale_np  # env-space delta
    return delta_env, h_out


def _compute_residual_delta_rollout(
    *,
    o_batch: dict[str, np.ndarray],
    infos_list: list[dict],
    h_in: th.Tensor,
    prev_action: np.ndarray,
    n_envs: int,
    device,
    ag_dim: int,
    dg_dim: int,
    base_obs_dim: int,
    obj_idx: list[int],
    obj_dim: int,
    fourier,
    residual,
    action_scale_np: np.ndarray,
) -> tuple[np.ndarray, th.Tensor]:
    o_obs = o_batch["observation"]  # (n_envs, base_obs_dim)
    o_ag = o_batch.get("achieved_goal", np.zeros((n_envs, ag_dim), dtype=np.float32))
    o_dg = o_batch.get("desired_goal", np.zeros((n_envs, dg_dim), dtype=np.float32))
    keep = [i for i in range(base_obs_dim) if i not in set(obj_idx)]

    obj_np = o_obs[:, obj_idx].astype(np.float32, copy=False)
    rest_np = o_obs[:, keep].astype(np.float32, copy=False)

    m_obj = np.ones((n_envs, obj_dim), dtype=np.float32)
    for i in range(n_envs):
        if isinstance(infos_list[i], dict) and "obs_mask" in infos_list[i]:
            mm = np.asarray(infos_list[i]["obs_mask"], dtype=np.float32).reshape(-1)
            if mm.shape[0] == base_obs_dim:
                m_obj[i] = mm[np.asarray(obj_idx, dtype=int)]

    obj_t = th.as_tensor(obj_np, device=device).float()
    obj_ff = fourier(obj_t)
    rest_t = th.as_tensor(rest_np, device=device).float()
    ag_t = th.as_tensor(o_ag, device=device).float()
    dg_t = th.as_tensor(o_dg, device=device).float()
    ap_t = th.as_tensor(prev_action, device=device).float()
    m_t = th.as_tensor(m_obj, device=device).float()
    x = th.cat([obj_ff, rest_t, ag_t, dg_t, ap_t, m_t], dim=-1).unsqueeze(1)  # (n_envs,1,input)

    with th.no_grad():
        delta, _, h_out = residual.forward_sequence(x, h_in, deterministic=False)
    delta_np = delta.squeeze(1).cpu().numpy().astype(np.float32, copy=False)  # (n_envs, act_dim) in [-1,1]
    delta_env = delta_np * action_scale_np
    return delta_env, h_out


def _step_env_vec(env, *, act_batch: np.ndarray, n_envs: int):
    out = env.step(act_batch)
    if isinstance(out, tuple) and len(out) == 5:
        o2, r, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return o2, np.array([float(r)], dtype=np.float32), np.array([1.0 if done else 0.0], dtype=np.float32), [info]
    if isinstance(out, tuple) and len(out) == 4:
        o2, r, d, infos_ = out
        r = np.asarray(r, dtype=np.float32).reshape(n_envs)
        d = np.asarray(d, dtype=np.float32).reshape(n_envs)
        infos_ = list(infos_)
        return o2, r, d, infos_
    raise RuntimeError(f"Unknown env.step() return signature: {type(out)} len={len(out) if isinstance(out, tuple) else 'n/a'}")


def _build_x_seq(
    *,
    obs_seq_list,
    actions_np,
    masks_np,
    base_obs_dim: int,
    ag_dim: int,
    dg_dim: int,
    obj_idx: list[int],
    obj_dim: int,
    act_dim: int,
    device,
    fourier,
):
    # obs_seq_list: (B, T) list of dicts
    B = len(obs_seq_list)
    T = len(obs_seq_list[0])

    obs_obs = np.empty((B, T, base_obs_dim), dtype=np.float32)
    has_ag = ag_dim > 0
    has_dg = dg_dim > 0
    obs_ag = np.empty((B, T, ag_dim), dtype=np.float32) if has_ag else np.zeros((B, T, 0), dtype=np.float32)
    obs_dg = np.empty((B, T, dg_dim), dtype=np.float32) if has_dg else np.zeros((B, T, 0), dtype=np.float32)

    for i in range(B):
        seq = obs_seq_list[i]
        for t in range(T):
            o = seq[t]
            obs_obs[i, t] = o["observation"]
            if has_ag:
                obs_ag[i, t] = o["achieved_goal"]
            if has_dg:
                obs_dg[i, t] = o["desired_goal"]

    keep = [i for i in range(base_obs_dim) if i not in set(obj_idx)]
    obj = obs_obs[:, :, obj_idx]  # (B,T,obj_dim)
    rest = obs_obs[:, :, keep]  # (B,T,rest_dim)
    if masks_np is None:
        m_obj = np.ones((B, T, obj_dim), dtype=np.float32)
    else:
        m_obj = masks_np[:, :, obj_idx].astype(np.float32, copy=False)

    a_prev = np.zeros((B, T, act_dim), dtype=np.float32)
    a_prev[:, 1:, :] = actions_np[:, : T - 1, :]

    obj_t = th.as_tensor(obj, device=device)
    obj_ff = fourier(obj_t.reshape(B * T, obj_dim)).reshape(B, T, -1)
    rest_t = th.as_tensor(rest, device=device)
    ag_t = th.as_tensor(obs_ag, device=device)
    dg_t = th.as_tensor(obs_dg, device=device)
    ap_t = th.as_tensor(a_prev, device=device)
    m_t = th.as_tensor(m_obj, device=device)
    x = th.cat([obj_ff, rest_t, ag_t, dg_t, ap_t, m_t], dim=-1)
    return x, obs_obs, obs_ag, obs_dg


def _run_paper_eval_at_step(
    *,
    env_step: int,
    paper_logger,
    writer,
    config: dict,
    env_id: str,
    run_id: str,
    method: str,
    paper_eval_settings: list[str],
    n_eval_paper: int,
    residual,
    base,
    device,
    ag_dim: int,
    dg_dim: int,
    base_obs_dim: int,
    obj_idx: list[int],
    obj_dim: int,
    act_dim: int,
    fourier,
    action_scale: th.Tensor,
    action_bias: th.Tensor,
) -> None:
    if paper_logger is None:
        return
    eval_seed_base = int(config["seed"]) + 1000
    alpha_eval = 1.0
    det_residual = True
    action_scale_np = action_scale.detach().cpu().numpy().astype(np.float32)

    for setting_id in paper_eval_settings:
        eval_env = _make_eval_env_residual(setting_id, seed=eval_seed_base, env_id=env_id)
        successes: list[float] = []
        returns: list[float] = []
        lengths: list[int] = []

        for ep in range(int(n_eval_paper)):
            obs, info = eval_env.reset(seed=eval_seed_base + ep)
            prev_a = np.zeros((act_dim,), dtype=np.float32)
            h = residual.init_hidden(1, device=device)

            done = False
            truncated = False
            ep_r = 0.0
            ep_l = 0

            while not (done or truncated):
                o_batch = _as_batch_obs_dict(obs)
                a0_env = _compute_base_a0_env(
                    base=base, device=device, action_bias=action_bias, action_scale=action_scale, o_batch=o_batch
                )[0]
                delta_env, h = _compute_residual_delta_eval(
                    o_batch=o_batch,
                    info=info if isinstance(info, dict) else {},
                    h_in=h,
                    prev_action_eval=prev_a,
                    deterministic=det_residual,
                    device=device,
                    ag_dim=ag_dim,
                    dg_dim=dg_dim,
                    base_obs_dim=base_obs_dim,
                    obj_idx=obj_idx,
                    obj_dim=obj_dim,
                    fourier=fourier,
                    residual=residual,
                    action_scale_np=action_scale_np,
                )
                act = a0_env + float(alpha_eval) * delta_env[0]
                act = np.clip(act, eval_env.action_space.low, eval_env.action_space.high).astype(np.float32, copy=False)

                step_out = eval_env.step(act)
                if len(step_out) == 5:
                    obs, reward, terminated, truncated, info = step_out
                    done = bool(terminated)
                    truncated = bool(truncated)
                else:
                    obs, reward, done, info = step_out
                    truncated = False

                if isinstance(info, dict) and "executed_action" in info:
                    try:
                        prev_a = np.asarray(info["executed_action"], dtype=np.float32).reshape(-1)
                    except Exception:
                        prev_a = act
                else:
                    prev_a = act

                ep_r += float(reward)
                ep_l += 1

            success = 0.0
            if isinstance(info, dict) and "is_success" in info:
                try:
                    success = float(info.get("is_success", 0.0))
                except Exception:
                    success = 0.0
            successes.append(success)
            returns.append(ep_r)
            lengths.append(int(ep_l))

        eval_env.close()

        sr = float(np.mean(successes)) if successes else float("nan")
        r_mean = float(np.mean(returns)) if returns else float("nan")
        r_std = float(np.std(returns)) if returns else float("nan")
        l_mean = float(np.mean(lengths)) if lengths else float("nan")
        l_std = float(np.std(lengths)) if lengths else float("nan")

        paper_logger.log_eval_summary(
            {
                "run_id": run_id,
                "method": method,
                "eval_setting_id": setting_id,
                "train_seed": int(config["seed"]),
                "env_step": int(env_step),
                "num_eval_episodes": int(n_eval_paper),
                "success_rate": sr,
                "return_mean": r_mean,
                "return_std": r_std,
                "len_mean": l_mean,
                "len_std": l_std,
            }
        )

        writer.add_scalar(f"eval/{setting_id}/success_rate", sr, int(env_step))
        writer.add_scalar(f"eval/{setting_id}/mean_reward", r_mean, int(env_step))
        writer.add_scalar(f"eval/{setting_id}/mean_ep_length", l_mean, int(env_step))
        if setting_id == "Nominal":
            writer.add_scalar("eval/success_rate", sr, int(env_step))

    print(
        f"[residual][paper_eval] step={int(env_step)} wrote eval_summary.csv for settings={paper_eval_settings} "
        f"(episodes/setting={int(n_eval_paper)})",
        flush=True,
    )


def train_pretrained_sac_with_residual_mem(args, CONFIG: dict, env, run_dir: str):
    if int(getattr(args, "history_k", 0)) != 0:
        raise ValueError("Residual-mem version requires --history-k 0 (base SAC input dim must not change).")
    if getattr(args, "pretrained", None) in (None, ""):
        raise ValueError("Residual-mem version requires --pretrained pointing to a SAC .zip model.")

    # Load pretrained SAC (base actor + critics). We'll train critics; base actor is frozen by default.
    base = SAC.load(args.pretrained, env=env, device="auto")
    device = base.device
    base.policy.set_training_mode(True)

    _freeze_module(base.policy.actor, train_last_layer_only=bool(getattr(args, "base_actor_unfreeze_last_layer", False)))

    # Action scaling helpers (torch), matching SB3 conventions:
    # - actor outputs "scaled" actions in [-1, 1]
    # - env uses Box(low, high)
    low = th.as_tensor(env.action_space.low, device=device).float()
    high = th.as_tensor(env.action_space.high, device=device).float()
    action_scale = 0.5 * (high - low)
    action_bias = 0.5 * (high + low)

    action_scale_np = action_scale.detach().cpu().numpy().astype(np.float32)

    # Entropy coefficient (fixed)
    ent_coef = None
    if isinstance(getattr(base, "ent_coef", None), str) and getattr(base, "ent_coef") == "auto":
        ent_coef = th.exp(base.log_ent_coef.detach())
    else:
        ent_coef = th.as_tensor(float(getattr(base, "ent_coef", 0.0)), device=device)

    # Residual input dims
    obj_idx = parse_indices(getattr(args, "residual_obj_idx", None))
    if obj_idx is None:
        obj_idx = [3, 4, 5]

    obs_space = env.observation_space
    base_obs_dim = int(obs_space["observation"].shape[0])
    ag_dim = int(obs_space["achieved_goal"].shape[0]) if "achieved_goal" in obs_space.spaces else 0
    dg_dim = int(obs_space["desired_goal"].shape[0]) if "desired_goal" in obs_space.spaces else 0
    act_dim = int(env.action_space.shape[0])

    obj_dim = len(obj_idx)
    rest_dim = base_obs_dim - obj_dim
    fourier = FourierFeatures(obj_dim, num_bands=int(args.fourier_bands), max_freq=float(args.fourier_max_freq)).to(device)
    input_dim = int(fourier.out_dim + rest_dim + ag_dim + dg_dim + act_dim + obj_dim)

    residual = ResidualMemPolicy(
        input_dim=input_dim,
        action_dim=act_dim,
        hidden_size=int(args.residual_hidden),
        mem_type=str(args.residual_mem_type),
    ).to(device)

    critic_opt = th.optim.Adam([p for p in base.policy.critic.parameters() if p.requires_grad], lr=float(CONFIG["learning_rate"]))
    residual_opt = th.optim.Adam(residual.parameters(), lr=float(CONFIG["learning_rate"]))

    # schedules
    alpha_schedule = AlphaSchedule(
        alpha_start=float(args.alpha_start),
        alpha_mid=float(args.alpha_mid),
        alpha_final=float(args.alpha_final),
        step_mid=int(args.alpha_step_mid),
        step_final=int(args.alpha_step_final),
    )

    use_curr = bool(getattr(args, "use_disturbance_curriculum", False))
    curr = None
    if use_curr:
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

    use_adv = bool(getattr(args, "use_adversary", False))
    adversary = None
    if use_adv:
        # Initialize adversary using args
        adv_cfg = AdversaryConfig(
            enabled=True,
            adv_lambda=float(getattr(args, "adv_lambda", 0.5)),
            w_act=float(getattr(args, "adv_w_act", 2.0)),
            update_every=int(getattr(args, "adv_update_every", 10)),
            batch_size=int(getattr(args, "adv_update_every", 10)),
            stage_steps=getattr(args, "adv_stage_steps", [200000, 400000]),
            k_obs_max=int(getattr(args, "adv_k_obs_max", 3)),
            k_act_max=int(getattr(args, "adv_k_act_max", 1)),
            lr=float(getattr(args, "adv_lr", 0.05)),
        )
        adversary = BudgetedAdversary(adv_cfg, device=device).to(device)
        print(f"[residual] Enabled Budgeted Adversary: lambda={adv_cfg.adv_lambda} k_obs_max={adv_cfg.k_obs_max}")

    n_envs = int(getattr(env, "num_envs", 1))

    # replay buffer (sequence-capable)
    rb = SequenceReplayBuffer(max_transitions=int(CONFIG["buffer_size"]), seed=int(CONFIG["seed"]), n_envs=n_envs)

    # tensorboard
    writer = SummaryWriter(log_dir=CONFIG["tensorboard_log_dir"])

    # ---------------------------------------------------------------------
    # Periodic "paper eval" for STRICTLY aligned learning curves
    # (baseline path logs eval curves via PaperExperimentCallback; residual path needs its own)
    #
    # We log to:
    # - run_dir/csv/eval_summary.csv  (same schema as PaperLogger)
    # - TensorBoard: eval/<setting_id>/success_rate (+ returns/len)
    # ---------------------------------------------------------------------
    try:
        gym.register_envs(gymnasium_robotics)
    except Exception:
        # Already registered or gym version doesn't require explicit registration
        pass

    eval_freq_paper = int(getattr(args, "eval_freq_paper", 0) or 0)
    n_eval_paper = int(getattr(args, "n_eval_paper", 0) or 0)
    paper_eval_settings = [
        s.strip() for s in str(getattr(args, "paper_eval_settings", "Nominal,ObsOnly3,ActOnly1,MixedHard")).split(",") if s.strip()
    ]

    csv_dir = os.path.join(run_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    run_id = os.path.basename(os.path.normpath(run_dir))
    method = str(getattr(args, "run_name", "") or getattr(args, "method_tag", "") or "ours")
    paper_logger = None
    next_eval_step = None
    if eval_freq_paper > 0 and n_eval_paper > 0:
        paper_logger = PaperLogger(log_dir=str(csv_dir), run_id=run_id, config=CONFIG)
        next_eval_step = int(eval_freq_paper)

    # Save metadata
    with open(os.path.join(run_dir, "residual_mem_meta.yaml"), "w", encoding="utf-8") as f:
        import yaml

        yaml.safe_dump(
            {
                "algo": "pretrained_sac_residual_mem",
                "obj_idx": obj_idx,
                "fourier": {"bands": int(args.fourier_bands), "max_freq": float(args.fourier_max_freq)},
                "alpha_schedule": asdict(alpha_schedule),
                "disturbance_curriculum": None if curr is None else asdict(curr),
                "seq_len": int(args.seq_len),
                "burn_in": int(args.burn_in),
                "pretrained": str(args.pretrained),
            },
            f,
            sort_keys=False,
        )

    # env init (Gym env returns (obs, info); VecEnv returns obs only)
    reset_out = None
    if hasattr(env, "reset"):
        try:
            reset_out = env.reset(seed=int(CONFIG["seed"]))
        except TypeError:
            reset_out = env.reset()
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        obs, info = reset_out
        infos = [info] * n_envs
    else:
        obs = reset_out
        infos = [{} for _ in range(n_envs)]

    prev_action = np.zeros((n_envs, act_dim), dtype=np.float32)
    # recurrent hidden for rollout
    h_roll = residual.init_hidden(n_envs, device=device)

    ep_reward = np.zeros((n_envs,), dtype=np.float32)
    ep_len = np.zeros((n_envs,), dtype=np.int32)
    recent_success: list[float] = []
    last_print_step = -1

    total_timesteps = int(CONFIG["total_timesteps"])
    batch_size = int(getattr(args, "rm_batch_size", None) or int(CONFIG["batch_size"]))
    burn_in = int(args.burn_in)
    seq_len = int(args.seq_len)
    train_freq = int(max(1, getattr(args, "rm_train_freq", 10)))
    gradient_steps = int(max(1, getattr(args, "rm_gradient_steps", 1)))

    global_step = 0  # count of transitions collected (SB3-like semantics)
    updates = 0
    next_train_step = int(train_freq)
    next_print_step = 0

    # Always record (and idempotently apply) the initial disturbance settings once,
    # so get_disturbance() reflects fixed-disturbance runs even when curriculum is OFF.
    init_ds = {"k_obs": int(args.obs_delay_k), "k_act": int(args.action_delay_k), "bias_b": float(args.obs_bias_b)}
    if getattr(args, "obs_dropout_p", None) is not None:
        init_ds["dropout_p"] = float(args.obs_dropout_p)
    else:
        init_ds["dropout_level"] = str(getattr(args, "obs_dropout_level", "none"))
    _maybe_set_disturbance(env, spec=init_ds)

    while global_step < total_timesteps:
        # curriculum update (disturbances)
        if curr is not None:
            spec = curr.spec(global_step)
            ds = {"k_obs": spec.k_obs, "k_act": spec.k_act, "bias_b": spec.bias_b}
            if spec.dropout_p is not None:
                ds["dropout_p"] = float(spec.dropout_p)
            else:
                ds["dropout_level"] = str(spec.dropout_level or "none")
                ds["dropout_level"] = str(spec.dropout_level or "none")
            _maybe_set_disturbance(env, spec=ds)
        
        if use_adv and global_step == 0:
             # Initial sample for all envs
             for i in range(n_envs):
                 k_obs, k_act, lp_obs, lp_act, stage = adversary.sample_disturbance(global_step)
                 if not hasattr(adversary, "env_context"): adversary.env_context = {}
                 adversary.env_context[i] = {"k_obs": k_obs, "k_act": k_act, "lp_obs": lp_obs, "lp_act": lp_act, "stage": stage}
                 # Apply
                 if hasattr(env, "env_method"):
                     env.env_method("set_disturbance", indices=[i], k_obs=int(k_obs), k_act=int(k_act))
                 else:
                     _maybe_set_disturbance(env, spec={"k_obs": int(k_obs), "k_act": int(k_act)})

        
        # adversary update (START of episode logic effectively happens after previous done, 
        # but here we need to ensure disturbances are set before step() if new episode started.
        # Actually our loop is Step-based. We need to detect "New Episode" or just rely on the fact 
        # that we only need to change dist when an episode finishes?
        # Standard SB3 callback does: _on_step -> if done: sample NEW dist for NEXT episode.
        # So we should put adversarial sampling in the DONE block below.
        # BUT: For the very first episode, we need to sample ONCE.
        if use_adv and global_step == 0:
             # Initial sample for all envs
             for i in range(n_envs):
                 k_obs, k_act, lp_obs, lp_act, stage = adversary.sample_disturbance(global_step)
                 # We need to store these to use them in store_episode later.
                 # Let's attach them to info or a separate buffer.
                 if not hasattr(adversary, "env_context"): adversary.env_context = {}
                 adversary.env_context[i] = {"k_obs": k_obs, "k_act": k_act, "lp_obs": lp_obs, "lp_act": lp_act, "stage": stage}
                 _maybe_set_disturbance(env, spec={"k_obs": int(k_obs), "k_act": int(k_act)})
                 # Note: VecEnv set_disturbance usually sets consistent prop. 
                 # If we have multiple envs with different dist, we need env_method with indices.
                 if hasattr(env, "env_method"):
                     env.env_method("set_disturbance", indices=[i], k_obs=int(k_obs), k_act=int(k_act))


        alpha = float(alpha_schedule.value(global_step))

        o_batch = _as_batch_obs_dict(obs)
        a0_env = _compute_base_a0_env(
            base=base, device=device, action_bias=action_bias, action_scale=action_scale, o_batch=o_batch
        )  # (n_envs, act_dim)
        delta_env, h_roll = _compute_residual_delta_rollout(
            o_batch=o_batch,
            infos_list=infos,
            h_in=h_roll,
            prev_action=prev_action,
            n_envs=n_envs,
            device=device,
            ag_dim=ag_dim,
            dg_dim=dg_dim,
            base_obs_dim=base_obs_dim,
            obj_idx=obj_idx,
            obj_dim=obj_dim,
            fourier=fourier,
            residual=residual,
            action_scale_np=action_scale_np,
        )
        act_batch = a0_env + float(alpha) * delta_env
        act_batch = np.clip(act_batch, env.action_space.low, env.action_space.high).astype(np.float32)

        next_obs, rewards, dones, infos2 = _step_env_vec(env, act_batch=act_batch, n_envs=n_envs)

        # store transitions per env
        o_batch_next = _as_batch_obs_dict(next_obs)
        for i in range(n_envs):
            o_i = {k: np.array(v[i], copy=True) for k, v in o_batch.items()}
            # terminal_observation if provided by VecEnv
            next_i = None
            if isinstance(infos2[i], dict) and "terminal_observation" in infos2[i]:
                tobs = infos2[i]["terminal_observation"]
                next_i = {k: np.array(np.asarray(tobs[k]).reshape(-1), copy=True) for k in tobs} if isinstance(tobs, dict) else o_i
            else:
                next_i = {k: np.array(v[i], copy=True) for k, v in o_batch_next.items()}

            obs_mask_i = None
            if isinstance(infos[i], dict) and "obs_mask" in infos[i]:
                obs_mask_i = np.asarray(infos[i]["obs_mask"], dtype=np.float32)

            # If action delay is enabled, store the actually executed action for correctness.
            act_store = act_batch[i]
            if isinstance(infos2[i], dict) and "executed_action" in infos2[i]:
                try:
                    act_store = np.asarray(infos2[i]["executed_action"], dtype=np.float32).reshape(-1)
                except Exception:
                    act_store = act_batch[i]

            rb.add(
                env_idx=i,
                obs=o_i,
                action=act_store,
                reward=float(rewards[i]),
                done=bool(dones[i]),
                next_obs=next_i,
                obs_mask=obs_mask_i,
            )

            ep_reward[i] += float(rewards[i])
            ep_len[i] += 1

            if bool(dones[i]):
                is_success = None
                if isinstance(infos2[i], dict) and "is_success" in infos2[i]:
                    try:
                        is_success = float(infos2[i]["is_success"])
                    except Exception:
                        is_success = None
                if is_success is not None:
                    recent_success.append(is_success)
                    if len(recent_success) > 100:
                        recent_success = recent_success[-100:]
                
                # ADVERSARY: Logic needs `ep_ret_val` before reset.
                ep_ret_val = float(ep_reward[i])
                
                # ADVERSARY STORE (Injecting here to access ep_ret_val)
                if use_adv and adversary is not None:
                     ctx = getattr(adversary, "env_context", {}).get(i)
                     if ctx:
                         # Use return or success? Standard is return. 
                         # But for Fetch, return is -50..0. Success is 0/1.
                         # Adversary minimizes agent return.
                         adversary.store_episode(
                            log_prob_obs=ctx["lp_obs"],
                            log_prob_act=ctx["lp_act"],
                            ep_return=ep_ret_val,
                            k_obs=ctx["k_obs"],
                            k_act=ctx["k_act"]
                        )

                writer.add_scalar("rollout/ep_reward", float(ep_reward[i]), global_step)
                writer.add_scalar("rollout/ep_len", int(ep_len[i]), global_step)
                if is_success is not None and recent_success:
                    writer.add_scalar("rollout/success_rate_100ep", float(np.mean(recent_success)), global_step)

                ds = _maybe_get_disturbance(env)
                sr = float(np.mean(recent_success)) if recent_success else None
                print(
                    f"[residual][episode] env={i} step={global_step} ep_len={int(ep_len[i])} ep_reward={float(ep_reward[i]):.3f} "
                    f"ep_success={is_success} success_100={sr} alpha={alpha:.3f} dist={ds}",
                    flush=True,
                )

                # reset per-env rollout state
                ep_reward_val = float(ep_reward[i])
                ep_reward[i] = 0.0
                ep_len[i] = 0
                prev_action[i, :] = 0.0
                h_roll[:, i, :] = 0.0
                
                # ADVERSARY: Episode Done
                if use_adv and adversary is not None:
                    # Store previous episode
                    ctx = getattr(adversary, "env_context", {}).get(i)
                    if ctx:
                        adversary.store_episode(
                            log_prob_obs=ctx["lp_obs"],
                            log_prob_act=ctx["lp_act"],
                            ep_return=ep_reward_val,
                            k_obs=ctx["k_obs"],
                            k_act=ctx["k_act"]
                        )

                    # Sample NEW disturbance for next episode
                    k_obs_new, k_act_new, lp_obs_new, lp_act_new, stage_new = adversary.sample_disturbance(global_step)
                    adversary.env_context[i] = {
                        "k_obs": k_obs_new, "k_act": k_act_new, 
                        "lp_obs": lp_obs_new, "lp_act": lp_act_new, "stage": stage_new
                    }
                    if hasattr(env, "env_method"):
                        env.env_method("set_disturbance", indices=[i], k_obs=int(k_obs_new), k_act=int(k_act_new))
                    else:
                        _maybe_set_disturbance(env, spec={"k_obs": int(k_obs_new), "k_act": int(k_act_new)})
                    
                    # Update
                    infos_adv = adversary.update()
                    if infos_adv:
                        for k,v in infos_adv.items():
                            writer.add_scalar(f"adversary/{k}", v, global_step)


        # update prev_action with executed action when available (action delay correctness)
        prev_action[:, :] = act_batch
        for i in range(n_envs):
            if isinstance(infos2[i], dict) and "executed_action" in infos2[i]:
                try:
                    prev_action[i, :] = np.asarray(infos2[i]["executed_action"], dtype=np.float32).reshape(-1)
                except Exception:
                    prev_action[i, :] = act_batch[i]
        for i in range(n_envs):
            if bool(dones[i]):
                prev_action[i, :] = 0.0
        obs = next_obs
        infos = infos2
        global_step += n_envs

        # periodic paper eval (strictly aligned metric for learning curves)
        if next_eval_step is not None and global_step >= int(next_eval_step):
            _run_paper_eval_at_step(
                env_step=int(global_step),
                paper_logger=paper_logger,
                writer=writer,
                config=CONFIG,
                env_id=CONFIG["env_id"],
                run_id=run_id,
                method=method,
                paper_eval_settings=paper_eval_settings,
                n_eval_paper=int(n_eval_paper),
                residual=residual,
                base=base,
                device=device,
                ag_dim=ag_dim,
                dg_dim=dg_dim,
                base_obs_dim=base_obs_dim,
                obj_idx=obj_idx,
                obj_dim=obj_dim,
                act_dim=act_dim,
                fourier=fourier,
                action_scale=action_scale,
                action_bias=action_bias,
            )
            next_eval_step = int(next_eval_step) + int(eval_freq_paper)

        # progress print (so terminal has continuous output; based on transition count)
        if global_step >= next_print_step:
            ds = _maybe_get_disturbance(env)
            sr = float(np.mean(recent_success)) if recent_success else None
            print(
                f"[residual] step={global_step}/{total_timesteps} updates={updates} alpha={alpha:.3f} "
                f"success_100={sr} dist={ds}",
                flush=True,
            )
            last_print_step = global_step
            next_print_step = int(last_print_step) + 1000

        # training update (sequence sampling + burn-in)
        min_ready = (burn_in + seq_len) * max(1, batch_size // 4)
        if rb.size < min_ready:
            continue

        # Train periodically (like SB3 train_freq), not every step.
        if global_step < next_train_step:
            continue
        next_train_step = int(next_train_step) + int(train_freq)

        for _ in range(gradient_steps):
            batch = rb.sample_sequences(batch_size=batch_size, seq_len=seq_len, burn_in=burn_in)
            obs_seq_list = batch["obs_seq"]  # list[B][T]
            actions_np = batch["actions"]  # (B,L,act)
            rewards_np = batch["rewards"]  # (B,L)
            dones_np = batch["dones"]  # (B,L)
            masks_np = batch["masks"]  # (B,T,obs_dim) or None

            # Build x_seq and obs tensors
            L = burn_in + seq_len  # transitions per sampled fragment
            T = L + 1  # obs steps
            x_seq, obs_obs_np, obs_ag_np, obs_dg_np = _build_x_seq(
                obs_seq_list=obs_seq_list,
                actions_np=actions_np,
                masks_np=masks_np,
                base_obs_dim=base_obs_dim,
                ag_dim=ag_dim,
                dg_dim=dg_dim,
                obj_idx=obj_idx,
                obj_dim=obj_dim,
                act_dim=act_dim,
                device=device,
                fourier=fourier,
            )

            # Slice to the "main" part: obs steps for training are burn_in..burn_in+seq_len (length seq_len+1)
            obs_main_obs = obs_obs_np[:, burn_in : burn_in + seq_len + 1, :]
            obs_main_ag = obs_ag_np[:, burn_in : burn_in + seq_len + 1, :]
            obs_main_dg = obs_dg_np[:, burn_in : burn_in + seq_len + 1, :]

            # Base actions for each MAIN obs step (torch)
            obs_flat = {
                "observation": th.as_tensor(obs_main_obs.reshape(batch_size * (seq_len + 1), base_obs_dim), device=device),
                "achieved_goal": th.as_tensor(obs_main_ag.reshape(batch_size * (seq_len + 1), ag_dim), device=device) if ag_dim > 0 else th.zeros((batch_size * (seq_len + 1), 0), device=device),
                "desired_goal": th.as_tensor(obs_main_dg.reshape(batch_size * (seq_len + 1), dg_dim), device=device) if dg_dim > 0 else th.zeros((batch_size * (seq_len + 1), 0), device=device),
            }
            with th.no_grad():
                a0_scaled = base.policy.actor(obs_flat, deterministic=True)
                a0_env = _unscale_action_torch(action_bias, action_scale, a0_scaled)
            a0_env = a0_env.view(batch_size, seq_len + 1, act_dim)

            # Residual policy with burn-in (no-grad) -> then compute outputs on main window with grad.
            h0 = residual.init_hidden(batch_size, device=device)
            if burn_in > 0:
                with th.no_grad():
                    _, h_b = residual.rnn(x_seq[:, :burn_in, :], h0)
            else:
                h_b = h0
            x_main = x_seq[:, burn_in:, :]  # (B, seq_len+1, input_dim)
            out_main, _ = residual.rnn(x_main, h_b)

            mu = residual.mu(out_main)
            log_std = residual.log_std(out_main).clamp(residual.log_std_min, residual.log_std_max)
            std = th.exp(log_std)
            u = mu + std * th.randn_like(mu)
            delta_main = th.tanh(u)  # (B, seq_len+1, act_dim) in [-1,1]

            # log prob of squashed Gaussian
            log_prob_u = -0.5 * (((u - mu) / (std + 1e-8)).pow(2) + 2.0 * log_std + float(np.log(2.0 * np.pi)))
            log_prob_u = log_prob_u.sum(dim=-1, keepdim=True)
            corr = th.log(1.0 - delta_main.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
            logp_main = log_prob_u - corr  # (B, seq_len+1, 1)

            alpha_t = float(alpha_schedule.value(global_step))
            alpha_clamped = float(max(1e-3, alpha_t))

            # final action samples for MAIN obs steps (delta normalized -> env delta)
            delta_env = delta_main * action_scale.view(1, 1, act_dim)
            a_pi_raw = a0_env + alpha_t * delta_env
            a_pi = th.max(th.min(a_pi_raw, high.view(1, 1, act_dim)), low.view(1, 1, act_dim))
            # adjust logp by alpha scaling (best-effort)
            logp_adj = logp_main - (act_dim * float(np.log(alpha_clamped)))

            # Transitions for loss: MAIN transitions are burn_in..burn_in+seq_len-1 (length seq_len)
            act_np_main = actions_np[:, burn_in : burn_in + seq_len, :]
            rew_np_main = rewards_np[:, burn_in : burn_in + seq_len]
            done_np_main = dones_np[:, burn_in : burn_in + seq_len]

            obs_t = {
                "observation": th.as_tensor(obs_main_obs[:, :seq_len, :].reshape(batch_size * seq_len, base_obs_dim), device=device),
                "achieved_goal": th.as_tensor(obs_main_ag[:, :seq_len, :].reshape(batch_size * seq_len, ag_dim), device=device) if ag_dim > 0 else th.zeros((batch_size * seq_len, 0), device=device),
                "desired_goal": th.as_tensor(obs_main_dg[:, :seq_len, :].reshape(batch_size * seq_len, dg_dim), device=device) if dg_dim > 0 else th.zeros((batch_size * seq_len, 0), device=device),
            }
            obs_tp1 = {
                "observation": th.as_tensor(obs_main_obs[:, 1:, :].reshape(batch_size * seq_len, base_obs_dim), device=device),
                "achieved_goal": th.as_tensor(obs_main_ag[:, 1:, :].reshape(batch_size * seq_len, ag_dim), device=device) if ag_dim > 0 else th.zeros((batch_size * seq_len, 0), device=device),
                "desired_goal": th.as_tensor(obs_main_dg[:, 1:, :].reshape(batch_size * seq_len, dg_dim), device=device) if dg_dim > 0 else th.zeros((batch_size * seq_len, 0), device=device),
            }

            act_b = th.as_tensor(act_np_main.reshape(batch_size * seq_len, act_dim), device=device)
            rew_t = th.as_tensor(rew_np_main.reshape(batch_size * seq_len, 1), device=device)
            done_t = th.as_tensor(done_np_main.reshape(batch_size * seq_len, 1), device=device)

            a_pi_tp1 = a_pi[:, 1:, :].reshape(batch_size * seq_len, act_dim)
            logp_tp1 = logp_adj[:, 1:, :].reshape(batch_size * seq_len, 1)

            act_b_s = _scale_action_torch(action_bias, action_scale, act_b)
            a_pi_tp1_s = _scale_action_torch(action_bias, action_scale, a_pi_tp1)

            with th.no_grad():
                q_targ_list = base.policy.critic_target(obs_tp1, a_pi_tp1_s)
                q_targ_min = th.min(th.cat(q_targ_list, dim=1), dim=1, keepdim=True).values
                target_q = rew_t + float(CONFIG["gamma"]) * (1.0 - done_t) * (q_targ_min - ent_coef * logp_tp1)

            q_curr_list = base.policy.critic(obs_t, act_b_s)
            critic_loss = 0.0
            for q in q_curr_list:
                critic_loss = critic_loss + F.mse_loss(q, target_q)

            critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_opt.step()

            # actor (residual) update: freeze critic params for stability
            for p in base.policy.critic.parameters():
                p.requires_grad_(False)

            a_pi_t = a_pi[:, :seq_len, :].reshape(batch_size * seq_len, act_dim)
            logp_t = logp_adj[:, :seq_len, :].reshape(batch_size * seq_len, 1)
            a_pi_t_s = _scale_action_torch(action_bias, action_scale, a_pi_t)
            q_pi_list = base.policy.critic(obs_t, a_pi_t_s)
            q_pi_min = th.min(th.cat(q_pi_list, dim=1), dim=1, keepdim=True).values
            actor_loss = (ent_coef * logp_t - q_pi_min).mean()

            residual_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            residual_opt.step()

            for p in base.policy.critic.parameters():
                p.requires_grad_(True)

            # soft update target critic
            tau = float(CONFIG["tau"])
            with th.no_grad():
                for p, p_targ in zip(base.policy.critic.parameters(), base.policy.critic_target.parameters()):
                    p_targ.data.mul_(1.0 - tau).add_(tau * p.data)

            updates += 1
        if updates % 50 == 0:
            writer.add_scalar("train/critic_loss", float(critic_loss.item()), global_step)
            writer.add_scalar("train/actor_loss", float(actor_loss.item()), global_step)
            writer.add_scalar("curriculum/alpha", float(alpha_t), global_step)
            ds = _maybe_get_disturbance(env)
            if ds is not None:
                if ds.get("dropout_p") is not None:
                    writer.add_scalar("curriculum/dropout_p", float(ds["dropout_p"] or 0.0), global_step)
                if ds.get("k_obs") is not None:
                    writer.add_scalar("curriculum/k_obs", float(ds["k_obs"] or 0.0), global_step)
                if ds.get("k_act") is not None:
                    writer.add_scalar("curriculum/k_act", float(ds["k_act"] or 0.0), global_step)
                if ds.get("bias_b") is not None:
                    writer.add_scalar("curriculum/bias_b", float(ds["bias_b"] or 0.0), global_step)

        if updates % 200 == 0:
            # additional (less frequent) diagnostic print
            ds = _maybe_get_disturbance(env)
            print(
                f"[residual][diag] step={global_step} updates={updates} alpha={alpha_t:.3f} ent_coef={float(ent_coef):.4f} dist={ds}",
                flush=True,
            )

    # Save artifacts
    base_out = os.path.join(run_dir, "base_sac")
    base.save(base_out)
    th.save({"state_dict": residual.state_dict(), "obj_idx": obj_idx}, os.path.join(run_dir, "residual_mem.pt"))
    writer.close()
    print(f"[residual] done. saved base_sac={base_out}.zip residual_mem.pt in {run_dir}")
