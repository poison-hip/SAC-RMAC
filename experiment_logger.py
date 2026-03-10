import os
import csv
import time
from typing import Dict, Any, List, Optional
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class PaperLogger:
    def __init__(self, log_dir: str, run_id: str, config: Dict[str, Any]):
        self.log_dir = log_dir
        self.run_id = run_id
        self.config = config
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Save config
        # (Assuming config is already saved by train.py, but we can save extra metadata if needed)
        
        # Initialize CSVs
        self.train_csv_path = os.path.join(self.log_dir, "train_episodes.csv")
        self.eval_csv_path = os.path.join(self.log_dir, "eval_summary.csv")
        
        # Define Headers
        self.train_headers = [
            "run_id", "method", "scenario_tag", "train_seed", 
            "env_step_end", "episode_idx", "episode_return", "success", "episode_len",
            "k_obs", "k_act", "adv_stage", "adv_cost", "adv_lambda",
            "alpha_residual", "residual_norm", "notes"
        ]
        
        self.eval_headers = [
            "run_id", "method", "eval_setting_id", "train_seed", "env_step",
            "num_eval_episodes", "success_rate", "return_mean", "return_std",
            "len_mean", "len_std"
        ]
        
        # Create CSVs with headers if they don't exist
        self._init_csv(self.train_csv_path, self.train_headers)
        self._init_csv(self.eval_csv_path, self.eval_headers)

    def _init_csv(self, path: str, headers: List[str]):
        # If file exists but is empty (e.g., pre-created), still write headers.
        if (not os.path.exists(path)) or (os.path.exists(path) and os.path.getsize(path) == 0):
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log_train_episode(self, row_dict: Dict[str, Any]):
        # Fill missing keys with sensible defaults
        row = [row_dict.get(h, "") for h in self.train_headers]
        with open(self.train_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def log_eval_summary(self, row_dict: Dict[str, Any]):
        row = [row_dict.get(h, "") for h in self.eval_headers]
        with open(self.eval_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

class PaperExperimentCallback(BaseCallback):
    """
    Handles paper-grade logging (A1) and periodic fixed-setting evaluation (B2).
    """
    def __init__(self, 
                 logger: PaperLogger, 
                 eval_freq: int, 
                 n_eval_episodes: int,
                 eval_settings: List[str],
                 make_eval_env_fn,  # Function that returns a fresh env for a given setting_id
                 train_seed: int,
                 method: str,
                 scenario_tag: str,
                 verbose: int = 0):
        super().__init__(verbose)
        self.paper_logger = logger
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_settings = eval_settings
        self.make_eval_env_fn = make_eval_env_fn
        self.train_seed = train_seed
        self.method = method
        self.scenario_tag = scenario_tag
        
        self.last_eval_step = 0
        self.episode_counter = 0

    def _on_step(self) -> bool:
        # A1: Training Log
        # We rely on 'dones' and 'infos' from the training env.
        # Ensure training env has RecordEpisodeStatistics or Monitor wrapper.
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        
        for i, done in enumerate(dones):
            if done:
                self.episode_counter += 1
                info = infos[i]
                
                # Extract Episode Stats
                ep_ret = info.get("episode", {}).get("r", 0.0)
                ep_len = info.get("episode", {}).get("l", 0)
                # Success is usually in info['is_success'] or info['success']. 
                # Gymnasium-Robotics: info['is_success']
                success = float(info.get("is_success", 0.0))
                
                # Extract Adversary/Disturbance Stats
                # If AdversarialTrainingCallback is running, it might have injected stats into info
                # Or we can read from the env wrapper if accessible.
                # Since we are in DummyVecEnv/SubprocVecEnv, we can't easily access the exact wrapper instance for *this* step
                # unless we stored it. 
                # However, AdversarialTrainingCallback logs into TensorBoard.
                # A reliable way is to trust the info dict if we ensure wrappers inject it there.
                
                # Try to get disturbance params from info (we need to ensure wrappers inject them)
                k_obs = info.get("k_obs", -1)
                k_act = info.get("k_act", -1)
                adv_stage = info.get("adv_stage", -1)
                adv_cost = info.get("adv_cost", 0.0)
                
                # Residual stats (if available)
                alpha = 0.0 # TODO: fetch from model/env if dynamic
                
                self.paper_logger.log_train_episode({
                    "run_id": self.paper_logger.run_id,
                    "method": self.method,
                    "scenario_tag": self.scenario_tag,
                    "train_seed": self.train_seed,
                    "env_step_end": self.num_timesteps,
                    "episode_idx": self.episode_counter,
                    "episode_return": ep_ret,
                    "success": int(success),
                    "episode_len": ep_len,
                    "k_obs": k_obs,
                    "k_act": k_act,
                    "adv_stage": adv_stage,
                    "adv_cost": adv_cost,
                    # "adv_lambda": ... (static from config)
                })

        # B2: Periodic Eval
        if self.eval_freq > 0 and self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            self._run_evals()
            
        return True

    def _run_evals(self):
        if self.verbose:
            print(f"[PaperEval] Running evaluation at step {self.num_timesteps}...")
            
        # We need the model to predict. 'self.model' is available in BaseCallback.
        model = self.model
        
        for setting_id in self.eval_settings:
            # Create fresh env for this setting
            eval_env = self.make_eval_env_fn(setting_id, seed=self.train_seed + 1000) # Offset seed for eval
            
            successes = []
            returns = []
            lengths = []
            
            obs, _ = eval_env.reset()
            
            # Reset residual memory if applicable (handled by env reset usually, but if model has state...)
            # The model policy (if recurrent) needs state reset.
            # SB3 predict handles state if we pass it back.
            
            # If we use residual memory, the "state" is internal to the policy (GRU hidden).
            # We usually need to reset it at episode start.
            
            for _ in range(self.n_eval_episodes):
                done = False
                truncated = False
                ep_r = 0.0
                ep_l = 0
                
                # Reset LSTM states for new episode
                # For SB3 SAC (non-recurrent), state is None.
                # For our ResidualMemPolicy, we might need to handle hidden state reset manually if it's not auto-handled.
                # Our ResidualMemPolicy usually handles auto-reset on `done` if properly integrated,
                # BUT `model.predict` on a single env requires us to manage `state`.
                
                # However, our ResidualMemPolicy likely hooks into observation history or just works step-by-step.
                # Let's assume `predict` works standardly.
                
                state = None 
                
                while not (done or truncated):
                    action, state = model.predict(obs, state=state, deterministic=True)
                    obs, reward, done, truncated, info = eval_env.step(action)
                    ep_r += reward
                    ep_l += 1
                
                # Episode done
                success = float(info.get("is_success", 0.0))
                successes.append(success)
                returns.append(ep_r)
                lengths.append(ep_l)
                
                obs, _ = eval_env.reset()
            
            eval_env.close()
            
            # Log summary
            self.paper_logger.log_eval_summary({
                "run_id": self.paper_logger.run_id,
                "method": self.method,
                "eval_setting_id": setting_id,
                "train_seed": self.train_seed,
                "env_step": self.num_timesteps,
                "num_eval_episodes": self.n_eval_episodes,
                "success_rate": np.mean(successes),
                "return_mean": np.mean(returns),
                "return_std": np.std(returns),
                "len_mean": np.mean(lengths),
                "len_std": np.std(lengths),
            })
            
            if self.verbose:
                print(f"  -> {setting_id}: SR={np.mean(successes):.2f}, R={np.mean(returns):.1f}")
