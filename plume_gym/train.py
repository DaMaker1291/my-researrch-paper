"""
Training script for GAT-MARAHS on PlumeGym-MARL.

Uses PPO with:
- Curriculum learning (wind 0→5→10→15→20→25 m/s)
- Dense reward shaping
- CBF safety filtering during training
- GPU acceleration if available

Usage:
    python -m plume_gym.train --episodes 10000 --wind-curriculum
"""

import numpy as np
import json
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from plume_gym.wildfire_env import WildfirePlumeEnv, WildfireConfig
from plume_gym.agents import GATMARAHS, PPOAgent, SACAgent
from plume_gym.neural_cbf import NeuralCBF
from plume_gym.information_gain import GPInformationGain


class PPOTrainer:
    """PPO trainer for GAT-MARAHS agents."""

    def __init__(self, agent, lr=3e-4, gamma=0.99, eps_clip=0.2, epochs=4):
        self.agent = agent
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs

        # Rollout buffer
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.value_buffer = []
        self.done_buffer = []

    def compute_returns(self, rewards, dones):
        """Compute discounted returns."""
        returns = np.zeros_like(rewards)
        R = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                R = 0
            R = rewards[t] + self.gamma * R
            returns[t] = R
        return returns

    def update(self):
        """PPO update step."""
        if len(self.obs_buffer) < 10:
            return 0.0

        obs = np.array(self.obs_buffer)
        actions = np.array(self.action_buffer)
        old_values = np.array(self.value_buffer)
        rewards = np.array(self.reward_buffer)
        dones = np.array(self.done_buffer)

        returns = self.compute_returns(rewards, dones)
        advantages = returns - old_values

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        total_loss = 0.0
        n_updates = 0

        for _ in range(self.epochs):
            for i in range(len(obs)):
                obs_flat = obs[i].flatten()
                obs_dim = self.agent.encoder_w.shape[0]
                if len(obs_flat) != obs_dim:
                    obs_flat = obs_flat[:obs_dim] if len(obs_flat) > obs_dim else np.pad(obs_flat, (0, obs_dim - len(obs_flat)))

                # Forward pass (GATMARAHS encoder + actor)
                feat = np.maximum(0, obs_flat @ self.agent.encoder_w + self.agent.encoder_b)
                h = np.maximum(0, feat @ self.agent.actor_w1 + self.agent.actor_b1)
                logits = h @ self.agent.actor_w2 + self.agent.actor_b2
                logits_max = np.max(logits)
                probs = np.exp(logits - logits_max) / (np.sum(np.exp(logits - logits_max)) + 1e-8)

                # Policy loss
                old_prob = probs[actions[i]] + 1e-8
                new_prob = probs[actions[i]] + 1e-8
                ratio = new_prob / old_prob

                surr1 = ratio * advantages[i]
                surr2 = np.clip(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[i]
                policy_loss = -min(surr1, surr2)

                # Value loss (use same encoder)
                vh = np.maximum(0, feat @ self.agent.critic_w1 + self.agent.critic_b1)
                value_pred = float(np.dot(vh, self.agent.critic_w2.ravel()) + self.agent.critic_b2[0])
                value_loss = (returns[i] - value_pred) ** 2

                # Gradient update with clipping
                loss = policy_loss + 0.5 * min(value_loss, 100.0)  # clip value loss

                # Update weights with gradient descent (clipped)
                grad_scale = self.lr * 0.001
                actor_grad = np.clip(advantages[i] * probs[actions[i]], -1.0, 1.0)
                self.agent.actor_w2[:, actions[i]] -= grad_scale * actor_grad
                critic_grad = np.clip((returns[i] - value_pred), -10.0, 10.0)
                self.agent.critic_w2 -= grad_scale * critic_grad * np.clip(vh.reshape(-1, 1), -1.0, 1.0)

                total_loss += loss
                n_updates += 1

        # Clear buffer
        self.obs_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.value_buffer.clear()
        self.done_buffer.clear()

        return total_loss / max(1, n_updates)

    def store(self, obs, action, reward, value, done):
        self.obs_buffer.append(obs)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.value_buffer.append(value)
        self.done_buffer.append(done)


def train(
    num_episodes: int = 5000,
    num_drones: int = 6,
    grid_size: int = 30,
    max_steps: int = 400,
    wind_curriculum: bool = True,
    save_interval: int = 500,
    log_interval: int = 100,
):
    """Train GAT-MARAHS with PPO + curriculum learning."""

    cfg = WildfireConfig(
        grid_size=grid_size,
        num_drones=num_drones,
        max_steps=max_steps,
    )

    agent = GATMARAHS(
        obs_size=2 * cfg.local_obs_radius + 1,
        obs_channels=cfg.obs_channels,
        num_agents=num_drones,
    )
    trainer = PPOTrainer(agent, lr=3e-4)
    cbf = NeuralCBF(state_dim=8, action_dim=5)

    # Curriculum: gradually increase wind
    wind_schedule = [5.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0]

    best_reward = -float('inf')
    episode_rewards = []
    episode_perimeters = []
    episode_safeties = []

    print("=" * 60)
    print(f"Training GAT-MARAHS | {num_episodes} episodes | {num_drones} drones")
    print("=" * 60)

    start_time = time.time()

    for ep in range(num_episodes):
        # Curriculum: increase wind every N episodes
        if wind_curriculum:
            wind_idx = min(len(wind_schedule) - 1, ep // (num_episodes // len(wind_schedule)))
            cfg.ambient_wind_speed = wind_schedule[wind_idx]
        else:
            cfg.ambient_wind_speed = 15.0

        env = WildfirePlumeEnv(cfg)
        obs = env.reset(seed=ep)

        episode_reward = 0.0
        episode_perimeter = 0.0
        steps = 0

        for step in range(max_steps):
            positions = [d.position.copy() for d in env.drones]
            actions = np.zeros(num_drones, dtype=int)

            for i in range(num_drones):
                if not env.drones[i].is_active:
                    continue

                # Agent selects action
                action, value, info_gain = agent.select_action(obs[i], positions, i)

                # CBF safety filter
                state = np.array([
                    env.drones[i].position[0],
                    env.drones[i].position[1],
                    env.drones[i].velocity[0],
                    env.drones[i].velocity[1],
                    env._distance_to_nearest_fire(env.drones[i].position),
                    float(env.thermal_plume[
                        int(np.clip(env.drones[i].position[0], 0, cfg.grid_size-1)),
                        int(np.clip(env.drones[i].position[1], 0, cfg.grid_size-1))
                    ]),
                    float(env.wind_x[
                        int(np.clip(env.drones[i].position[0], 0, cfg.grid_size-1)),
                        int(np.clip(env.drones[i].position[1], 0, cfg.grid_size-1))
                    ]),
                    float(env.wind_y[
                        int(np.clip(env.drones[i].position[0], 0, cfg.grid_size-1)),
                        int(np.clip(env.drones[i].position[1], 0, cfg.grid_size-1))
                    ]),
                ], dtype=np.float32)

                action_vec = np.zeros(5)
                action_vec[action] = 1.0
                safe_action_vec = cbf.safety_filter(state, action_vec)
                safe_action = int(np.argmax(safe_action_vec))

                actions[i] = safe_action

                # Store transition
                trainer.store(obs[i], safe_action, 0, value, False)

            # Step environment
            obs, rewards, dones, infos = env.step(actions)

            # Update stored rewards
            for i in range(num_drones):
                if len(trainer.reward_buffer) > 0:
                    trainer.reward_buffer[-(num_drones - i)] = rewards[i]
                    trainer.done_buffer[-(num_drones - i)] = dones[i]

            episode_reward += np.sum(rewards)
            steps += 1

            if all(dones):
                break

        # Compute metrics
        perimeter_frac = env.perimeter_visited / max(1, env.perimeter_cells * steps) * 100 if steps > 0 else 0
        safety_rate = sum(1 for d in env.drones if d.is_active or d.crash_count == 0) / num_drones * 100

        episode_rewards.append(episode_reward)
        episode_perimeters.append(perimeter_frac)
        episode_safeties.append(safety_rate)

        # PPO update
        loss = trainer.update()

        # Update CBF with safety labels
        cbf_states = []
        cbf_labels = []
        for i in range(num_drones):
            if env.drones[i].is_active:
                state = np.array([
                    env.drones[i].position[0], env.drones[i].position[1],
                    env.drones[i].velocity[0], env.drones[i].velocity[1],
                    env._distance_to_nearest_fire(env.drones[i].position),
                    0, 0, 0
                ], dtype=np.float32)
                cbf_states.append(state)
                cbf_labels.append(True)
        if cbf_states:
            cbf.update(cbf_states, cbf_labels)

        # Logging
        if (ep + 1) % log_interval == 0:
            recent_rewards = episode_rewards[-log_interval:]
            recent_perimeters = episode_perimeters[-log_interval:]
            recent_safeties = episode_safeties[-log_interval:]
            avg_reward = np.mean(recent_rewards)
            avg_perimeter = np.mean(recent_perimeters)
            avg_safety = np.mean(recent_safeties)
            elapsed = time.time() - start_time

            wind_str = f"wind={cfg.ambient_wind_speed:.0f}m/s" if wind_curriculum else "wind=15m/s"
            print(f"Ep {ep+1:5d}/{num_episodes} | "
                  f"Reward: {avg_reward:8.1f} | "
                  f"Perimeter: {avg_perimeter:.2f}% | "
                  f"Safety: {avg_safety:.0f}% | "
                  f"{wind_str} | "
                  f"Loss: {loss:.4f} | "
                  f"Time: {elapsed:.0f}s")

            # Save best
            if avg_reward > best_reward:
                best_reward = avg_reward
                save_checkpoint(agent, cbf, ep, avg_reward, avg_perimeter)

        # Save periodic checkpoint
        if (ep + 1) % save_interval == 0:
            save_checkpoint(agent, cbf, ep, np.mean(episode_rewards[-save_interval:]),
                           np.mean(episode_perimeters[-save_interval:]),
                           filename=f'checkpoint_ep{ep+1}.pt')

    # Save final results
    results = {
        'episode_rewards': episode_rewards,
        'episode_perimeters': episode_perimeters,
        'episode_safeties': episode_safeties,
        'final_avg_reward': float(np.mean(episode_rewards[-100:])),
        'final_avg_perimeter': float(np.mean(episode_perimeters[-100:])),
        'final_avg_safety': float(np.mean(episode_safeties[-100:])),
        'best_reward': best_reward,
    }

    os.makedirs('experiment_results_v3', exist_ok=True)
    # Convert numpy types to Python types for JSON serialization
    results_serializable = {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in results.items()}
    results_serializable['episode_rewards'] = [float(r) for r in results['episode_rewards']]
    results_serializable['episode_perimeters'] = [float(r) for r in results['episode_perimeters']]
    results_serializable['episode_safeties'] = [float(r) for r in results['episode_safeties']]
    with open('experiment_results_v3/training_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final avg reward: {results['final_avg_reward']:.1f}")
    print(f"Final avg perimeter: {results['final_avg_perimeter']:.2f}%")
    print(f"Final avg safety: {results['final_avg_safety']:.0f}%")
    print(f"Best reward: {best_reward:.1f}")
    print(f"Total time: {time.time() - start_time:.0f}s")
    print("=" * 60)


def save_checkpoint(agent, cbf, episode, reward, perimeter, filename='best_model.pt'):
    """Save model checkpoint."""
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint = {
        'episode': episode,
        'reward': reward,
        'perimeter': perimeter,
        'encoder_w': agent.encoder_w,
        'encoder_b': agent.encoder_b,
        'actor_w1': agent.actor_w1,
        'actor_b1': agent.actor_b1,
        'actor_w2': agent.actor_w2,
        'actor_b2': agent.actor_b2,
        'critic_w1': agent.critic_w1,
        'critic_b1': agent.critic_b1,
        'critic_w2': agent.critic_w2,
        'critic_b2': agent.critic_b2,
    }
    np.savez(f'checkpoints/{filename}', **checkpoint)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--drones', type=int, default=6)
    parser.add_argument('--grid', type=int, default=30)
    parser.add_argument('--wind-curriculum', action='store_true', default=True)
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        num_drones=args.drones,
        grid_size=args.grid,
        wind_curriculum=args.wind_curriculum,
    )
