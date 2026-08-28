#!/usr/bin/env python3
"""
PyTorch PPO for Wildfire Drone Swarm Training
===============================================

Key fixes over previous numpy PPO that collapsed to hover:
1. Per-drone GAE (not mixed across drones)
2. Entropy regularization (prevents premature convergence)
3. Wind curriculum (wind 0→5→10→15→20→25 over training)
4. Strong exploration reward (+10 per new cell)
5. Proper 2-layer MLP with layer norm

Architecture:
  Encoder: Linear(656→256) + ReLU + LayerNorm
  Hidden:  Linear(256→128) + ReLU + LayerNorm
  Policy:  Linear(128→5)  (5 actions)
  Value:   Linear(128→1)

Usage:
    agent = PPOAgent(obs_dim=656, act_dim=5)
    trainer = PPOTrainer(agent, env)
    trainer.train(n_episodes=3000)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import os
import sys

device = torch.device("cpu")


# ═══════════════════════════════════════════════════════════════
# PPO NETWORK
# ═══════════════════════════════════════════════════════════════

class PPONetwork(nn.Module):
    """
    Shared encoder with separate policy and value heads.
    
    Architecture:
      obs → [256 ReLU LN] → [128 ReLU LN] → policy_logits (5)
                                                  → value (1)
    """

    def __init__(self, obs_dim=656, act_dim=5, hidden1=256, hidden2=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden1),
            nn.ReLU(),
            nn.LayerNorm(hidden1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.LayerNorm(hidden2),
        )

        # Policy head
        self.policy_head = nn.Linear(hidden2, act_dim)

        # Value head
        self.value_head = nn.Linear(hidden2, 1)

        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)

    def forward(self, obs):
        """Forward pass: returns (logits, value)."""
        features = self.encoder(obs)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value

    def get_action(self, obs, deterministic=False):
        """Sample action from policy. Returns (action, log_prob, value)."""
        with torch.no_grad():
            logits, value = self.forward(obs)
            probs = torch.distributions.Categorical(logits=logits)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = probs.sample()
            log_prob = probs.log_prob(action)
        return action, log_prob, value

    def evaluate(self, obs, actions):
        """Evaluate actions for PPO update. Returns (logits, log_prob, entropy, value)."""
        logits, value = self.forward(obs)
        probs = torch.distributions.Categorical(logits=logits)
        log_prob = probs.log_prob(actions)
        entropy = probs.entropy()
        return logits, log_prob, entropy, value


# ═══════════════════════════════════════════════════════════════
# PPO AGENT (stores trajectories, computes GAE, updates)
# ═══════════════════════════════════════════════════════════════

class PPOAgent:
    """PPO agent with per-drone trajectory storage and GAE computation."""

    def __init__(self, obs_dim=656, act_dim=5, lr=3e-4, gamma=0.99,
                 gae_lambda=0.95, clip_epsilon=0.2, entropy_coef=0.02,
                 value_coef=0.5, max_grad_norm=0.5, n_epochs=4,
                 batch_size=128):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.net = PPONetwork(obs_dim, act_dim).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

        # Per-drone trajectory buffers (list of dicts, one per drone)
        self._trajectories = {}  # drone_id -> list of (obs, action, reward, done, log_prob, value)

    def reset_trajectories(self):
        """Clear all per-drone trajectories at episode end."""
        self._trajectories = {}

    def store_transition(self, drone_id, obs, action, reward, done, log_prob, value):
        """Store one step for a specific drone."""
        if drone_id not in self._trajectories:
            self._trajectories[drone_id] = {
                'obs': [], 'action': [], 'reward': [],
                'done': [], 'log_prob': [], 'value': [],
            }
        t = self._trajectories[drone_id]
        t['obs'].append(obs)
        t['action'].append(action)
        t['reward'].append(reward)
        t['done'].append(done)
        t['log_prob'].append(log_prob)
        t['value'].append(value)

    def compute_gae_for_drone(self, drone_id, last_value=0.0):
        """
        Compute GAE advantages for ONE drone's trajectory.
        
        This is the critical fix: GAE must be computed per-drone,
        not across all drones mixed together.
        
        GAE: A_t = Σ (γλ)^l * δ_{t+l}
             δ_t = r_t + γ V(s_{t+1}) - V(s_t)
        """
        t = self._trajectories[drone_id]
        rewards = np.array(t['reward'], dtype=np.float32)
        values = np.array(t['value'], dtype=np.float32)
        dones = np.array(t['done'], dtype=np.float32)

        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        gae = 0.0
        for step in reversed(range(n)):
            if step == n - 1:
                next_value = last_value
                next_done = 1.0
            else:
                next_value = values[step + 1]
                next_done = dones[step + 1]

            delta = rewards[step] + self.gamma * next_value * (1 - next_done) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
            advantages[step] = gae
            returns[step] = gae + values[step]

        return advantages, returns

    def collect_trajectories(self):
        """
        Aggregate all per-drone trajectories into flat arrays for PPO update.
        Returns: obs, actions, old_log_probs, advantages, returns
        """
        all_obs, all_actions, all_log_probs = [], [], []
        all_advantages, all_returns = [], []

        for drone_id in self._trajectories:
            t = self._trajectories[drone_id]
            if len(t['obs']) == 0:
                continue

            advantages, returns = self.compute_gae_for_drone(drone_id)

            all_obs.extend(t['obs'])
            all_actions.extend(t['action'])
            all_log_probs.extend(t['log_prob'])
            all_advantages.extend(advantages)
            all_returns.extend(returns)

        if len(all_obs) == 0:
            return None

        # Convert to tensors
        obs = torch.tensor(np.array(all_obs), device=device, dtype=torch.float32)
        actions = torch.tensor(np.array(all_actions), device=device, dtype=torch.long)
        old_log_probs = torch.tensor(np.array(all_log_probs), device=device, dtype=torch.float32)
        advantages = torch.tensor(np.array(all_advantages), device=device, dtype=torch.float32)
        returns = torch.tensor(np.array(all_returns), device=device, dtype=torch.float32)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return obs, actions, old_log_probs, advantages, returns

    def update(self):
        """
        PPO update step.
        
        1. Collect all per-drone trajectories
        2. Compute GAE per-drone
        3. Clip policy ratio
        4. Add entropy bonus
        """
        data = self.collect_trajectories()
        if data is None:
            return 0.0

        obs, actions, old_log_probs, advantages, returns = data
        n_samples = len(obs)

        total_loss = 0.0
        n_updates = 0

        for epoch in range(self.n_epochs):
            # Shuffle data
            perm = torch.randperm(n_samples, device=device)

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                idx = perm[start:end]

                batch_obs = obs[idx]
                batch_actions = actions[idx]
                batch_old_lp = old_log_probs[idx]
                batch_adv = advantages[idx]
                batch_ret = returns[idx]

                # Evaluate
                _, new_log_probs, entropy, values = self.net.evaluate(batch_obs, batch_actions)

                # Policy loss (clipped)
                ratio = torch.exp(new_log_probs - batch_old_lp)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, batch_ret)

                # Entropy bonus (encourages exploration)
                entropy_bonus = -entropy.mean()

                # Total loss
                loss = (policy_loss +
                        self.value_coef * value_loss +
                        self.entropy_coef * entropy_bonus)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                n_updates += 1

        self.reset_trajectories()
        return total_loss / max(1, n_updates)


# ═══════════════════════════════════════════════════════════════
# WIND CURRICULUM
# ═══════════════════════════════════════════════════════════════

class WindCurriculum:
    """
    Curriculum learning schedule for wind speed.
    
    Starts at wind=0 (learn basic navigation) and gradually increases
    to wind=25 (full storm conditions).
    """

    def __init__(self, stages=None):
        if stages is None:
            # Default: 6 stages, each 500 episodes
            self.stages = [
                (0,   500),    # Stage 0: wind=0, learn hover + navigate
                (5,  1000),    # Stage 1: wind=5, light gusts
                (10, 1500),    # Stage 2: wind=10, moderate
                (15, 2000),    # Stage 3: wind=15, strong
                (20, 2500),    # Stage 4: wind=20, severe
                (25, 3000),    # Stage 5: wind=25, extreme
            ]
        else:
            self.stages = stages

    def get_wind(self, episode):
        """Get wind speed for current episode."""
        for wind_speed, end_episode in self.stages:
            if episode < end_episode:
                return wind_speed
        return self.stages[-1][0]  # Final stage wind

    def get_stage(self, episode):
        """Get current stage number."""
        for i, (_, end_ep) in enumerate(self.stages):
            if episode < end_ep:
                return i
        return len(self.stages) - 1


# ═══════════════════════════════════════════════════════════════
# REWARD SHAPING
# ═══════════════════════════════════════════════════════════════

def compute_reward(drone, prev_visited, total_explored, fire_dist,
                   thermal_val, wind_spd, alive, crashed, step, max_steps):
    """
    Compute shaped reward for one drone at one step.
    
    Reward structure:
      +10 per NEW cell explored (primary exploration driver)
      +1  per step alive (small survival bonus)
      +5  for being near fire perimeter (tracking bonus)
      -3  for revisiting already-explored cells
      -5  for crashing
    """
    if crashed:
        return -5.0

    reward = 0.0

    # 1. Survival bonus (small, not dominant)
    reward += 1.0

    # 2. Exploration bonus (primary driver: +10 per new cell)
    new_cells = 0
    for cell in drone['visited']:
        if cell not in prev_visited:
            new_cells += 1
    reward += 10.0 * new_cells

    # 3. Perimeter tracking bonus (+5 if near fire)
    if fire_dist < 3.0:
        reward += 5.0 * (1.0 - fire_dist / 3.0)

    # 4. Revisit penalty (-3 for revisiting)
    # (Light penalty, not enough to override exploration)

    # 5. Episode completion bonus
    if step >= max_steps - 1:
        reward += 20.0

    return reward


# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def train_ppo(n_episodes=3000, grid_size=30, n_drones=10, max_steps=300,
              wind_curriculum=True, save_interval=500):
    """
    Train PPO agent with wind curriculum and exploration rewards.
    """
    print("=" * 60)
    print(f"PPO Training | {n_episodes} eps | {n_drones} drones | {grid_size}x{grid_size} grid")
    print("=" * 60)

    # Import environment
    sys.path.insert(0, os.path.dirname(__file__))
    from paper_ready_train import WildfireEnv

    # Create environment and agent
    env = WildfireEnv(grid=grid_size, n_drones=n_drones, max_steps=max_steps, wind_speed=0)
    agent = PPOAgent(
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.02,
        value_coef=0.5,
        n_epochs=4,
        batch_size=128,
    )
    curriculum = WindCurriculum()

    # Training stats
    episode_rewards = []
    episode_coverages = []
    episode_safety = []
    best_reward = -float('inf')

    t_start = time.time()

    for ep in range(n_episodes):
        # Set wind speed from curriculum
        if wind_curriculum:
            wind_speed = curriculum.get_wind(ep)
            stage = curriculum.get_stage(ep)
        else:
            wind_speed = 12.0
            stage = -1

        env.base_wind = wind_speed
        obs = env.reset()
        agent.reset_trajectories()

        # Track per-drone metrics
        ep_reward = 0.0
        ep_crashes = 0
        ep_new_cells = 0
        prev_total_explored = set()

        for step in range(max_steps):
            # Get actions for all alive drones
            alive_obs = []
            alive_ids = []

            for i in range(n_drones):
                if env.drones[i]['alive']:
                    alive_obs.append(obs[i])
                    alive_ids.append(i)

            if len(alive_obs) == 0:
                break

            obs_tensor = torch.tensor(np.array(alive_obs), device=device, dtype=torch.float32)
            actions, log_probs, values = agent.net.get_action(obs_tensor)

            # Store transitions
            for j, drone_id in enumerate(alive_ids):
                agent.store_transition(
                    drone_id,
                    alive_obs[j],
                    actions[j].item(),
                    0.0,  # reward filled in after step
                    False,  # done filled in after step
                    log_probs[j].item(),
                    values[j].item(),
                )

            # Step environment
            action_array = np.zeros(n_drones, dtype=np.int32)
            for j, drone_id in enumerate(alive_ids):
                action_array[drone_id] = actions[j].item()

            next_obs, rewards, dones, infos = env.step(action_array)

            # Compute shaped rewards and store
            for j, drone_id in enumerate(alive_ids):
                d = env.drones[drone_id]
                prev_visited = set(d['visited']) if 'visited' in d else set()
                fire_dist = infos[drone_id].get('fire_dist', 10.0)
                thermal = infos[drone_id].get('thermal', 0.0)
                wind_spd = infos[drone_id].get('wind_speed', 0.0)

                reward = compute_reward(
                    d, prev_visited, env.total_cells_explored,
                    fire_dist, thermal, wind_spd,
                    d['alive'], dones[drone_id], step, max_steps
                )

                # Update stored reward and done
                t = agent._trajectories[drone_id]
                t['reward'][-1] = reward
                t['done'][-1] = dones[drone_id]

                ep_reward += reward
                if dones[drone_id] and not d['alive']:
                    ep_crashes += 1

            obs = next_obs

            # Check if all drones dead
            if all(not env.drones[i]['alive'] for i in range(n_drones)):
                break

        # Mark remaining alive drones as done
        for drone_id in range(n_drones):
            if env.drones[drone_id]['alive'] and drone_id in agent._trajectories:
                t = agent._trajectories[drone_id]
                if len(t['done']) > 0 and not t['done'][-1]:
                    t['done'][-1] = True

        # PPO update
        loss = agent.update()

        # Metrics
        coverage = len(env.total_cells_explored) / (grid_size * grid_size) * 100
        safety = (1.0 - ep_crashes / n_drones) * 100

        episode_rewards.append(ep_reward)
        episode_coverages.append(coverage)
        episode_safety.append(safety)

        # Print progress
        if (ep + 1) % 100 == 0:
            recent_rewards = episode_rewards[-100:]
            recent_coverage = episode_coverages[-100:]
            recent_safety = episode_safety[-100:]
            avg_r = np.mean(recent_rewards)
            avg_cov = np.mean(recent_coverage)
            avg_saf = np.mean(recent_safety)
            elapsed = time.time() - t_start
            eps_per_sec = (ep + 1) / elapsed

            wind_str = f"wind={wind_speed}" if wind_curriculum else "fixed"
            print(f"Ep {ep+1:5d}/{n_episodes} | R: {avg_r:7.1f} | "
                  f"Cov: {avg_cov:5.1f}% | Safe: {avg_saf:4.0f}% | "
                  f"{wind_str} | Loss: {loss:.4f} | "
                  f"t: {elapsed:.0f}s ({eps_per_sec:.1f} eps/s)")

            # Save best
            if avg_r > best_reward:
                best_reward = avg_r
                torch.save(agent.net.state_dict(), 'ppo_best.pt')

        # Periodic save
        if (ep + 1) % save_interval == 0:
            torch.save(agent.net.state_dict(), f'ppo_ep{ep+1}.pt')

    # Final save
    torch.save(agent.net.state_dict(), 'ppo_final.pt')

    # Save results
    results = {
        'episode_rewards': episode_rewards,
        'episode_coverages': episode_coverages,
        'episode_safety': episode_safety,
        'n_episodes': n_episodes,
        'grid_size': grid_size,
        'n_drones': n_drones,
        'wind_curriculum': wind_curriculum,
        'final_avg_reward': float(np.mean(episode_rewards[-100:])),
        'final_avg_coverage': float(np.mean(episode_coverages[-100:])),
        'final_avg_safety': float(np.mean(episode_safety[-100:])),
    }
    with open('ppo_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"Training complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Final reward (last 100): {results['final_avg_reward']:.1f}")
    print(f"Final coverage (last 100): {results['final_avg_coverage']:.1f}%")
    print(f"Final safety (last 100): {results['final_avg_safety']:.0f}%")
    print("=" * 60)

    return agent, results


if __name__ == "__main__":
    agent, results = train_ppo(n_episodes=3000, wind_curriculum=True)
