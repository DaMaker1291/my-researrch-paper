"""
Single-Agent RL Network for Hurricane Drone Coverage
=====================================================
Conv1D encoder + MLP policy + value head.
Used as baseline for multi-agent comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Dict, Tuple


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)
    return layer


class ActorCritic(nn.Module):
    """
    Single-drone actor-critic network.
    
    Input: observation (472D)
    Output: action distribution + value
    """
    
    def __init__(self, obs_dim=472, action_dim=4, hidden_dim=256):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            orthogonal_init(nn.Linear(hidden_dim // 2, action_dim), gain=0.01),
        )
        
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            orthogonal_init(nn.Linear(hidden_dim // 2, 1), gain=1.0),
        )
        
        # Observation normalization
        self.obs_mean = nn.Parameter(torch.zeros(obs_dim), requires_grad=False)
        self.obs_var = nn.Parameter(torch.ones(obs_dim), requires_grad=False)
        self.obs_count = 0
        
    def get_action(self, obs, deterministic=False):
        """Get action from observation."""
        # Normalize
        obs_norm = (obs - self.obs_mean) / (self.obs_var + 1e-8).sqrt()
        
        # Features
        features = self.feature_extractor(obs_norm)
        
        # Actor
        action_mean = self.actor(features)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        
        dist = Normal(action_mean, action_std)
        action = action_mean if deterministic else dist.sample()
        action = torch.clamp(action, -1, 1)
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(features).squeeze(-1)
        
        return {
            'action': action,
            'log_prob': log_prob,
            'value': value,
        }
    
    def update_obs_stats(self, obs):
        """Update running observation statistics."""
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        batch = torch.FloatTensor(obs)
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        batch_count = float(batch.shape[0])
        
        if batch_count < 2:
            return
        
        total = self.obs_count + batch_count
        delta = batch_mean - self.obs_mean
        self.obs_mean += delta * batch_count / total
        self.obs_var = torch.clamp(
            (self.obs_var * self.obs_count + batch_var * batch_count +
             delta**2 * self.obs_count * batch_count / total) / total,
            min=1e-6
        )
        self.obs_count = total


class PPOTrainer:
    """PPO trainer for single-agent."""
    
    def __init__(self, model, lr=3e-4, gamma=0.99, lam=0.95,
                 clip_eps=0.2, entropy_coef=0.01):
        self.model = model
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=2000, eta_min=lr * 0.1)
    
    def compute_gae(self, rewards, values, dones):
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            next_val = values[t + 1] if t < len(rewards) - 1 else 0
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae
            returns[t] = advantages[t] + values[t]
        return advantages, returns
    
    def update(self, obs, actions, log_probs, rewards, dones, values):
        obs_t = torch.FloatTensor(obs)
        actions_t = torch.FloatTensor(actions)
        old_lp_t = torch.FloatTensor(log_probs)
        dones_t = torch.FloatTensor(dones)
        
        advantages, returns = self.compute_gae(rewards, values, dones)
        adv_t = torch.FloatTensor(advantages)
        ret_t = torch.FloatTensor(returns)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        
        total_pl, total_vl, total_ent, n = 0, 0, 0, 0
        batch_size = len(obs)
        mini_batch = min(256, batch_size)
        
        for _ in range(4):
            idx = np.random.permutation(batch_size)
            for start in range(0, batch_size, mini_batch):
                bi = idx[start:start + mini_batch]
                
                result = self.model.get_action(obs_t[bi])
                ratio = torch.exp(result['log_prob'] - old_lp_t[bi])
                
                s1 = ratio * adv_t[bi]
                s2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[bi]
                policy_loss = -torch.min(s1, s2).mean()
                
                value_loss = F.mse_loss(result['value'], ret_t[bi])
                
                dist = Normal(result['action'], self.model.actor_log_std.exp())
                entropy = dist.entropy().mean()
                
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
                total_pl += policy_loss.item()
                total_vl += value_loss.item()
                total_ent += entropy.item()
                n += 1
        
        return {
            'policy_loss': total_pl / max(n, 1),
            'value_loss': total_vl / max(n, 1),
            'entropy': total_ent / max(n, 1),
        }


class RolloutBuffer:
    """Simple rollout buffer."""
    
    def __init__(self):
        self.obs, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones, self.values = [], [], []
    
    def add(self, obs, action, log_prob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
    
    def get_all(self):
        return {
            'obs': np.array(self.obs),
            'actions': np.array(self.actions),
            'log_probs': np.array(self.log_probs),
            'rewards': np.array(self.rewards),
            'dones': np.array(self.dones),
            'values': np.array(self.values),
        }
    
    def reset(self):
        self.obs, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones, self.values = [], [], []
