
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class AdversaryConfig:
    enabled: bool = False
    adv_lambda: float = 0.5  # budget penalty coeff
    w_act: float = 2.0       # cost weight for action delay
    update_every: int = 10   # update every N episodes
    batch_size: int = 20     # use last N episodes for update
    k_obs_max: int = 3
    k_act_max: int = 1
    lr: float = 0.05
    stage_steps: List[int] = None # e.g. [200000, 400000]

class BudgetedAdversary(nn.Module):
    def __init__(self, config: AdversaryConfig, device="cpu"):
        super().__init__()
        self.cfg = config
        self.device = device
        
        # Parameters: log-probabilities (unnormalized logits)
        # Independent distributions for simplicity and robustness
        # k_obs in [0, k_obs_max], k_act in [0, k_act_max]
        self.logits_obs = nn.Parameter(th.zeros(self.cfg.k_obs_max + 1))
        self.logits_act = nn.Parameter(th.zeros(self.cfg.k_act_max + 1))
        
        self.optimizer = th.optim.Adam(self.parameters(), lr=self.cfg.lr)
        
        # Buffer for REINFORCE: list of (log_prob, return, cost)
        # Actually we store (log_prob, reward) where reward = -ret + lambda*cost
        self.t_buffer = []  # format: (log_prob_obs, log_prob_act, ep_return, cost)
        
        self.current_stage = 0
        self.stage_steps = self.cfg.stage_steps or [200000, 400000]

    def get_stage(self, global_step: int) -> int:
        if global_step < self.stage_steps[0]:
            return 0
        elif global_step < self.stage_steps[1]:
            return 1
        else:
            return 2

    def get_action_mask(self, stage: int) -> Tuple[th.Tensor, th.Tensor]:
        # Stage 0: k_obs <= 1, k_act <= 0
        # Stage 1: k_obs <= 2, k_act <= 1
        # Stage 2: k_obs <= 3, k_act <= 1
        
        mask_obs = th.zeros_like(self.logits_obs, dtype=th.bool)
        mask_act = th.zeros_like(self.logits_act, dtype=th.bool)
        
        limit_obs = 1
        limit_act = 0
        
        if stage >= 1:
            limit_obs = 2
            limit_act = 1
        if stage >= 2:
            limit_obs = 3 # or self.cfg.k_obs_max
            limit_act = 1 # or self.cfg.k_act_max
            
        mask_obs[:limit_obs+1] = True
        mask_act[:limit_act+1] = True
        
        return mask_obs, mask_act

    def sample_disturbance(self, global_step: int) -> Tuple[int, int, th.Tensor, th.Tensor, int]:
        self.current_stage = self.get_stage(global_step)
        mask_obs, mask_act = self.get_action_mask(self.current_stage)
        
        # Masked softmax
        # We use a large negative number for masked elements
        inf_mask_obs = th.where(mask_obs, th.zeros_like(self.logits_obs), th.tensor(-1e9).to(self.device))
        inf_mask_act = th.where(mask_act, th.zeros_like(self.logits_act), th.tensor(-1e9).to(self.device))
        
        probs_obs = F.softmax(self.logits_obs + inf_mask_obs, dim=0)
        probs_act = F.softmax(self.logits_act + inf_mask_act, dim=0)
        
        dist_obs = th.distributions.Categorical(probs_obs)
        dist_act = th.distributions.Categorical(probs_act)
        
        idx_obs = dist_obs.sample()
        idx_act = dist_act.sample()
        # Debug print
        if global_step % 200 == 0:
            print(f"[ADV DEBUG] Step={global_step} Stage={self.current_stage} ObsLogits={self.logits_obs.data} ProbsObs={probs_obs.data} SampledObs={idx_obs.item()}", flush=True)

        return (
            idx_obs.item(), 
            idx_act.item(), 
            dist_obs.log_prob(idx_obs), 
            dist_act.log_prob(idx_act),
            self.current_stage
        )

    def store_episode(self, log_prob_obs, log_prob_act, ep_return, k_obs, k_act):
        cost = k_obs + self.cfg.w_act * k_act
        self.t_buffer.append({
            "lp_obs": log_prob_obs,
            "lp_act": log_prob_act,
            "ret": ep_return,
            "cost": cost
        })

    def update(self) -> dict:
        if len(self.t_buffer) < self.cfg.batch_size:
            return {}
            
        # Use last N episodes
        batch = self.t_buffer[-self.cfg.batch_size:]
        self.t_buffer = [] # Clear buffer after update (on-policyish) or keep sliding window? 
        # REINFORCE is on-policy, so clear is strict. User said "sliding window" in prompt? 
        # "Use recent N to update" -> REINFORCE usually discards. Let's discard.
        
        rewards = []
        for x in batch:
            # Adversary wants to Minimize Return => Maximize (-Return)
            # Penalty for cost: Maximize (-Return + lambda * cost) ??? 
            # Wait. "Minimize Protagonist Return BUT with budget penalty".
            # Obj = -Return + lambda * Cost (This implies we WANT high cost? No.)
            # If we want to minimize return usually implies strong attack.
            # Budget penalty implies we want LOW cost.
            # So we want to maximize: ( -Return ) - lambda * Cost
            # = minimize (Return + lambda * Cost)
            
            # Adv Reward = -1 * (ProtagonistReturn) - lambda * Cost
            # If lambda=0, we maximize negative return (i.e. minimize return). Correct.
            # If Cost is high, Adv Reward decreases. Correct.
            adv_reward = -1.0 * x['ret'] - self.cfg.adv_lambda * x['cost']
            rewards.append(adv_reward)
            
        rewards = th.tensor(rewards).to(self.device)
        # Normalize rewards for stability
        if rewards.std() > 1e-5:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
        loss = []
        for i, x in enumerate(batch):
            loss.append( -1.0 * (x['lp_obs'] + x['lp_act']) * rewards[i] )
            
        self.optimizer.zero_grad()
        final_loss = th.stack(loss).mean()
        final_loss.backward()
        self.optimizer.step()
        
        return {
            "adv_loss": final_loss.item(),
            "adv_reward": rewards.mean().item(),
            "avg_cost": sum(x['cost'] for x in batch) / len(batch)
        }
