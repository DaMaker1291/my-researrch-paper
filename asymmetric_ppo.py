"""
Asymmetric Actor-Critic with LSTM for Hurricane Drone Control
==============================================================

Architecture:
- Actor (LSTM): Sees only IMU data (what real drone sees)
  - Input: IMU history (accelerometer + gyroscope + motor commands)
  - LSTM: 128 hidden units, processes temporal patterns
  - Output: 4 motor RPM commands (continuous)
  
- Critic (MLP): Gets privileged information during training
  - Input: Full state (position, velocity, wind, mass, drag, etc.)
  - MLP: 256-256 hidden layers
  - Output: Scalar value estimate

Key insight: The critic sees "god-mode" data during training but is NOT used
at inference time. The actor must learn to infer wind conditions from IMU
vibration patterns alone.

Training: PPO with asymmetric actor-critic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class LSTMActor(nn.Module):
    """
    LSTM-based actor that sees only IMU history.
    
    The LSTM processes temporal patterns in the IMU data to infer
    wind conditions and motor dynamics. This is the only network
    used at inference time.
    
    Input: IMU history [H frames of (motor_commands(4) + accelerometer(3))]
    Output: 4 motor RPM commands (continuous, [-1, 1])
    """
    
    def __init__(self, obs_dim: int = 80, hidden_dim: int = 128,
                 action_dim: int = 4, history_length: int = 10):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.history_length = history_length
        
        # Feature extraction per frame
        self.frame_encoder = nn.Sequential(
            nn.Linear(7, 32),  # 7 = 4 motor + 3 accel per frame
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
        )
        
        # Position error + quaternion encoder (static features)
        self.static_encoder = nn.Sequential(
            nn.Linear(10, 32),  # 3 pos_error + 4 quat + 3 gyro
            nn.LayerNorm(32),
            nn.ELU(),
        )
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=32,  # frame encoding size
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        
        # Policy head (outputs mean and log_std for continuous actions)
        self.policy_mean = nn.Sequential(
            nn.Linear(hidden_dim + 32, 64),  # LSTM output + static features
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),  # output in [-1, 1]
        )
        
        self.policy_log_std = nn.Parameter(
            torch.zeros(action_dim) - 0.5  # initial std ~ 0.6
        )
        
        # Hidden state for LSTM
        self.hidden = None
    
    def init_hidden(self, batch_size: int = 1, device: str = 'cpu'):
        """Initialize LSTM hidden state."""
        self.hidden = (
            torch.zeros(2, batch_size, self.hidden_dim).to(device),
            torch.zeros(2, batch_size, self.hidden_dim).to(device),
        )
    
    def forward(self, obs: torch.Tensor, hidden: Optional[Tuple] = None) -> Dict:
        """
        Args:
            obs: (batch, obs_dim) or (batch, seq_len, obs_dim)
            hidden: optional LSTM hidden state
        
        Returns:
            dict with 'actions', 'log_probs', 'values'
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # add sequence dimension
        
        B, T, D = obs.shape
        
        # Extract static features (first 10 dims: pos_error + quat + gyro)
        static = obs[:, -1, :10]  # use last frame's static features
        static_feat = self.static_encoder(static)  # (B, 32)
        
        # Extract frame features (motor_history + accel_history)
        frame_feats = []
        for t in range(T):
            frame = obs[:, t, 10:17]  # 4 motor + 3 accel per frame
            frame_feat = self.frame_encoder(frame)  # (B, 32)
            frame_feats.append(frame_feat)
        
        frame_feats = torch.stack(frame_feats, dim=1)  # (B, T, 32)
        
        # LSTM processing
        if hidden is not None:
            lstm_out, new_hidden = self.lstm(frame_feats, hidden)
        else:
            lstm_out, new_hidden = self.lstm(frame_feats)
        
        # Use last hidden state
        lstm_feat = lstm_out[:, -1, :]  # (B, hidden_dim)
        
        # Combine with static features
        combined = torch.cat([lstm_feat, static_feat], dim=-1)  # (B, hidden_dim+32)
        
        # Policy
        action_mean = self.policy_mean(combined)  # (B, action_dim)
        action_std = torch.exp(self.policy_log_std).expand_as(action_mean)
        
        return {
            'action_mean': action_mean,
            'action_std': action_std,
            'hidden': new_hidden,
        }
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Dict:
        """Get action for inference (no gradient)."""
        with torch.no_grad():
            result = self.forward(obs, self.hidden)
            self.hidden = result['hidden']
        
        mean = result['action_mean']
        std = result['action_std']
        
        if deterministic:
            action = mean
        else:
            action = mean + std * torch.randn_like(mean)
        
        action = torch.clamp(action, -1, 1)
        
        # Log probability
        log_std = self.policy_log_std.expand_as(mean)
        log_prob = -0.5 * ((action - mean) / (std + 1e-8)) ** 2 - log_std - 0.5 * np.log(2 * np.pi)
        log_prob = log_prob.sum(dim=-1)
        
        return {
            'actions': action,
            'log_probs': log_prob,
        }


class PrivilegedCritic(nn.Module):
    """
    Critic that sees full privileged information during training.
    
    Input: Full state (position, velocity, wind vector, mass, drag, etc.)
    Output: Scalar value estimate
    
    This network is ONLY used during training and is discarded at inference.
    It provides "god-mode" value estimates to train the actor via advantage.
    """
    
    def __init__(self, privileged_dim: int = 20):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(privileged_dim, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, privileged_obs: torch.Tensor) -> torch.Tensor:
        """privileged_obs: (B, privileged_dim) → value: (B, 1)"""
        return self.net(privileged_obs)


class AsymmetricPPO:
    """
    Asymmetric PPO with LSTM actor and privileged critic.
    
    Key difference from standard PPO:
    - Actor sees only IMU data (realistic)
    - Critic sees full state (privileged info during training only)
    - This forces the actor to learn to infer hidden physics from limited observations
    """
    
    def __init__(self, actor: LSTMActor, critic: PrivilegedCritic,
                 actor_lr: float = 3e-4, critic_lr: float = 1e-3,
                 gamma: float = 0.99, lam: float = 0.95,
                 clip_eps: float = 0.2, entropy_coef: float = 0.01):
        
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        
        # Separate optimizers
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr, eps=1e-5)
        
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=1000, eta_min=actor_lr * 0.1
        )
        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer, T_max=1000, eta_min=critic_lr * 0.1
        )
    
    def compute_gae(self, rewards: np.ndarray, values: np.ndarray,
                    dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        T = len(rewards)
        advantages = np.zeros(T)
        returns = np.zeros(T)
        
        last_gae = 0
        for t in reversed(range(T)):
            next_val = values[t + 1] if t < T - 1 else 0
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def update(self, rollout_data: Dict) -> Dict:
        """Update both actor and critic from rollout data."""
        obs = rollout_data['obs']  # (T, obs_dim)
        privileged_obs = rollout_data['privileged_obs']  # (T, privileged_dim)
        actions = rollout_data['actions']  # (T, action_dim)
        old_log_probs = rollout_data['log_probs']  # (T,)
        rewards = rollout_data['rewards']  # (T,)
        dones = rollout_data['dones']  # (T,)
        
        # Compute advantages using privileged critic
        with torch.no_grad():
            values = self.critic(torch.FloatTensor(privileged_obs)).squeeze(-1).numpy()
        
        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        obs_t = torch.FloatTensor(obs)
        actions_t = torch.FloatTensor(actions)
        old_lp_t = torch.FloatTensor(old_log_probs)
        adv_t = torch.FloatTensor(advantages)
        ret_t = torch.FloatTensor(returns)
        
        # PPO update epochs
        total_stats = {'policy_loss': 0, 'value_loss': 0, 'entropy': 0}
        
        for _ in range(4):  # 4 PPO epochs
            # Shuffle data
            idx = np.random.permutation(len(obs))
            
            for start in range(0, len(obs), 64):
                batch_idx = idx[start:start + 64]
                
                # Actor update
                self.actor.init_hidden(batch_size=len(batch_idx))
                result = self.actor(obs_t[batch_idx])
                
                mean = result['action_mean']
                std = result['action_std']
                
                # New log probs
                log_std = self.actor.policy_log_std.expand_as(mean)
                new_lp = -0.5 * ((actions_t[batch_idx] - mean) / (std + 1e-8)) ** 2 - log_std - 0.5 * np.log(2 * np.pi)
                new_lp = new_lp.sum(dim=-1)
                
                # PPO clip
                ratio = torch.exp(new_lp - old_lp_t[batch_idx])
                s1 = ratio * adv_t[batch_idx]
                s2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[batch_idx]
                policy_loss = -torch.min(s1, s2).mean()
                
                # Entropy bonus
                entropy = 0.5 * (1 + torch.log(std ** 2 + 1e-8)).sum(dim=-1).mean()
                
                # Actor backward
                self.actor_optimizer.zero_grad()
                (policy_loss - self.entropy_coef * entropy).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                
                # Critic update
                value = self.critic(torch.FloatTensor(privileged_obs[batch_idx])).squeeze(-1)
                value_loss = F.mse_loss(value, ret_t[batch_idx])
                
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
                
                total_stats['policy_loss'] += policy_loss.item()
                total_stats['value_loss'] += value_loss.item()
                total_stats['entropy'] += entropy.item()
        
        # Step schedulers
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        
        n = max(1, len(obs) // 64 * 4)
        return {k: v / n for k, v in total_stats.items()}
    
    def save(self, path: str):
        """Save both actor and critic."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load both actor and critic."""
        checkpoint = torch.load(path, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
