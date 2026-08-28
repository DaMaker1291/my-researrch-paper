"""
Train PPO Baseline for Hurricane Drone Coverage
=================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import time


class PPOActorCritic(nn.Module):
    """PPO actor-critic for continuous control."""
    
    def __init__(self, obs_dim=41, act_dim=4, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.Tanh(),
            nn.Linear(hidden//2, act_dim), nn.Tanh(),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.Tanh(),
            nn.Linear(hidden//2, 1),
        )
    
    def forward(self, obs):
        x = self.shared(obs)
        mean = self.actor_mean(x)
        std = self.actor_log_std.exp().expand_as(mean)
        value = self.critic(x)
        return mean, std, value
    
    def get_action(self, obs, deterministic=False):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        action = mean if deterministic else dist.sample()
        action = torch.clamp(action, -1, 1)
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value.squeeze(-1)
    
    def evaluate(self, obs, actions):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value.squeeze(-1)


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = np.zeros_like(rewards)
    returns = np.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(len(rewards))):
        next_val = values[t+1] if t < len(rewards)-1 else 0
        delta = rewards[t] + gamma * next_val * (1-dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (1-dones[t]) * last_gae
        returns[t] = advantages[t] + values[t]
    return advantages, returns


def train_ppo():
    """Train PPO on hurricane environment."""
    from hurricane_env import HurricaneStationKeepingEnv, HurricaneConfig
    from real_wind_provider import RealWindProvider
    
    print("="*60)
    print("TRAINING PPO BASELINE")
    print("="*60)
    
    # Create env
    config = HurricaneConfig(wind_provider='katrina')
    env = HurricaneStationKeepingEnv(config=config)
    wind = RealWindProvider('katrina')
    env.set_wind_provider(wind)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    print(f"Obs dim: {obs_dim}, Act dim: {act_dim}")
    
    # Create model
    device = torch.device('cpu')
    model = PPOActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    
    # Training config
    total_steps = 10000
    rollout_len = 100
    n_epochs = 4
    batch_size = 64
    gamma = 0.99
    lam = 0.95
    clip_eps = 0.2
    
    # Training loop
    global_step = 0
    episode_rewards = []
    episode_coverages = []
    best_coverage = 0
    
    print(f"\nTraining for {total_steps} steps...")
    start_time = time.time()
    
    while global_step < total_steps:
        # Collect rollout
        obs_list, actions_list, log_probs_list = [], [], []
        rewards_list, dones_list, values_list = [], [], []
        
        obs_np, _ = env.reset()
        obs = torch.FloatTensor(obs_np)
        
        ep_reward = 0
        ep_coverage = 0
        
        for step in range(rollout_len):
            with torch.no_grad():
                action, log_prob, value = model.get_action(obs)
            
            obs_np_new, reward, terminated, truncated, info = env.step(action.numpy())
            done = terminated or truncated
            
            obs_list.append(obs)
            actions_list.append(action)
            log_probs_list.append(log_prob)
            rewards_list.append(reward)
            dones_list.append(float(done))
            values_list.append(value.item())
            
            ep_reward += reward
            ep_coverage = info.get('coverage_pct', 0)
            
            obs = torch.FloatTensor(obs_np_new)
            global_step += 1
            
            if done:
                break
        
        episode_rewards.append(ep_reward)
        episode_coverages.append(ep_coverage)
        
        if ep_coverage > best_coverage:
            best_coverage = ep_coverage
        
        # Compute GAE
        obs_tensor = torch.stack(obs_list)
        actions_tensor = torch.stack(actions_list)
        old_log_probs_tensor = torch.stack(log_probs_list)
        
        with torch.no_grad():
            _, _, next_value = model.get_action(obs)
            values_list.append(next_value.item())
        
        advantages, returns = compute_gae(
            np.array(rewards_list), np.array(values_list), np.array(dones_list),
            gamma, lam
        )
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_pg_loss = 0
        total_v_loss = 0
        n_updates = 0
        
        for _ in range(n_epochs):
            indices = np.random.permutation(len(obs_list))
            for start in range(0, len(obs_list), batch_size):
                end = min(start + batch_size, len(obs_list))
                batch_idx = indices[start:end]
                
                batch_obs = obs_tensor[batch_idx]
                batch_actions = actions_tensor[batch_idx]
                batch_old_lp = old_log_probs_tensor[batch_idx]
                batch_adv = advantages[batch_idx]
                batch_ret = returns[batch_idx]
                
                new_lp, entropy, new_value = model.evaluate(batch_obs, batch_actions)
                
                ratio = torch.exp(new_lp - batch_old_lp)
                s1 = ratio * batch_adv
                s2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * batch_adv
                pg_loss = -torch.min(s1, s2).mean()
                
                v_loss = nn.MSELoss()(new_value, batch_ret)
                
                loss = pg_loss + 0.5 * v_loss - 0.01 * entropy.mean()
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                
                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                n_updates += 1
        
        # Print progress
        if len(episode_rewards) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_coverage = np.mean(episode_coverages[-10:])
            elapsed = time.time() - start_time
            fps = global_step / max(elapsed, 0.01)
            
            print(f"Step {global_step:>6} | "
                  f"Reward {avg_reward:>8.1f} | "
                  f"Coverage {avg_coverage:>5.1f}% | "
                  f"Best {best_coverage:.1f}% | "
                  f"FPS {fps:.0f}")
    
    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Final best coverage: {best_coverage:.1f}%")
    
    # Save model
    torch.save(model.state_dict(), 'ppo_hurricane.pt')
    print("Model saved to ppo_hurricane.pt")
    
    return model, best_coverage


def evaluate_ppo(model, num_episodes=20):
    """Evaluate trained PPO."""
    from hurricane_env import HurricaneStationKeepingEnv, HurricaneConfig
    from real_wind_provider import RealWindProvider
    
    print("\n" + "="*60)
    print("EVALUATING TRAINED PPO")
    print("="*60)
    
    config = HurricaneConfig(wind_provider='katrina')
    env = HurricaneStationKeepingEnv(config=config)
    wind = RealWindProvider('katrina')
    env.set_wind_provider(wind)
    
    coverages = []
    crashes = 0
    
    for ep in range(num_episodes):
        obs_np, _ = env.reset()
        obs = torch.FloatTensor(obs_np)
        
        for step in range(600):
            with torch.no_grad():
                action, _, _ = model.get_action(obs, deterministic=True)
            obs_np, reward, terminated, truncated, info = env.step(action.numpy())
            obs = torch.FloatTensor(obs_np)
            if terminated or truncated:
                if reward < -50:
                    crashes += 1
                break
        
        coverages.append(info.get('coverage_pct', 0))
    
    print(f"PPO Results ({num_episodes} episodes):")
    print(f"  Coverage: {np.mean(coverages):.1f}% ± {np.std(coverages):.1f}%")
    print(f"  Crashes: {crashes}/{num_episodes}")
    print(f"  Safety: {100*(1-crashes/num_episodes):.1f}%")
    
    return np.mean(coverages), 100*(1-crashes/num_episodes)


def compare_all():
    """Compare all methods with proper numbers."""
    import numpy as np
    from hurricane_env import HurricaneStationKeepingEnv, HurricaneConfig
    from real_wind_provider import RealWindProvider
    from safe_adaptive_controller import SafeAdaptiveController
    
    print("\n" + "#"*60)
    print("# FINAL COMPARISON: All Methods (600 steps)")
    print("#"*60)
    
    config = HurricaneConfig(wind_provider='katrina')
    env = HurricaneStationKeepingEnv(config=config)
    wind = RealWindProvider('katrina')
    env.set_wind_provider(wind)
    
    results = {}
    
    # 1. Random
    coverages = []
    crashes = 0
    for seed in range(20):
        np.random.seed(seed)
        obs, _ = env.reset()
        for step in range(600):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                if reward < -50: crashes += 1
                break
        coverages.append(info.get('coverage_pct', 0))
    results['Random'] = {'coverage': np.mean(coverages), 'std': np.std(coverages), 
                         'safety': 100*(1-crashes/20), 'crashes': crashes}
    print(f"Random:     {np.mean(coverages):.1f}% ± {np.std(coverages):.1f}% | Safety: {100*(1-crashes/20):.0f}%")
    
    # 2. Greedy
    coverages = []
    crashes = 0
    for seed in range(20):
        np.random.seed(seed)
        obs, _ = env.reset()
        for step in range(600):
            target_dir = obs[27:29]
            target_dist = obs[29]
            alt_err = obs[31]
            if target_dist > 0.01:
                action = np.array([-alt_err*0.3, -target_dir[1]*0.5, target_dir[0]*0.5, 0])
            else:
                action = np.array([-alt_err*0.3, 0, 0, 0])
            action = np.clip(action, -1, 1)
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                if reward < -50: crashes += 1
                break
        coverages.append(info.get('coverage_pct', 0))
    results['Greedy'] = {'coverage': np.mean(coverages), 'std': np.std(coverages),
                         'safety': 100*(1-crashes/20), 'crashes': crashes}
    print(f"Greedy:     {np.mean(coverages):.1f}% ± {np.std(coverages):.1f}% | Safety: {100*(1-crashes/20):.0f}%")
    
    # 3. PPO (trained)
    try:
        model = PPOActorCritic(obs_dim=41, act_dim=4)
        model.load_state_dict(torch.load('ppo_hurricane.pt', weights_only=True))
        model.eval()
        
        coverages = []
        crashes = 0
        for seed in range(20):
            np.random.seed(seed)
            obs_np, _ = env.reset()
            obs = torch.FloatTensor(obs_np)
            for step in range(600):
                with torch.no_grad():
                    action, _, _ = model.get_action(obs, deterministic=True)
                obs_np, reward, term, trunc, info = env.step(action.numpy())
                obs = torch.FloatTensor(obs_np)
                if term or trunc:
                    if reward < -50: crashes += 1
                    break
            coverages.append(info.get('coverage_pct', 0))
        results['PPO'] = {'coverage': np.mean(coverages), 'std': np.std(coverages),
                          'safety': 100*(1-crashes/20), 'crashes': crashes}
        print(f"PPO:        {np.mean(coverages):.1f}% ± {np.std(coverages):.1f}% | Safety: {100*(1-crashes/20):.0f}%")
    except:
        print("PPO:        (not trained yet)")
        results['PPO'] = {'coverage': 0, 'std': 0, 'safety': 0, 'crashes': 20}
    
    # 4. MARAHS
    controller = SafeAdaptiveController()
    coverages = []
    crashes = 0
    for seed in range(20):
        np.random.seed(seed)
        obs, _ = env.reset()
        controller.reset()
        for step in range(600):
            state = {
                'position': env.dynamics.position,
                'velocity': env.dynamics.velocity,
                'quaternion': env.dynamics.orientation,
                'motor_rpms': env.dynamics.motor_rpms,
                'mass': 1.5,
            }
            imu = {'acceleration': np.array([0,0,9.81]), 'gyroscope': np.zeros(3), 
                   'quaternion': np.array([1,0,0,0])}
            result = controller.get_action(obs, state=state, imu_data=imu)
            action = result['action']
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                if reward < -50: crashes += 1
                break
        coverages.append(info.get('coverage_pct', 0))
    results['MARAHS'] = {'coverage': np.mean(coverages), 'std': np.std(coverages),
                         'safety': 100*(1-crashes/20), 'crashes': crashes}
    print(f"MARAHS:     {np.mean(coverages):.1f}% ± {np.std(coverages):.1f}% | Safety: {100*(1-crashes/20):.0f}%")
    
    # Summary table
    print("\n" + "="*65)
    print("FINAL RESULTS TABLE")
    print("="*65)
    print(f"{'Method':<12} {'Coverage':>12} {'Safety':>10} {'vs MARAHS':>12}")
    print("-"*65)
    for method in ['Random', 'Greedy', 'PPO', 'MARAHS']:
        r = results[method]
        delta = r['coverage'] - results['MARAHS']['coverage']
        delta_str = f"{delta:+.1f}%" if method != 'MARAHS' else "—"
        print(f"{method:<12} {r['coverage']:>5.1f}% ± {r['std']:.1f} {r['safety']:>8.0f}% {delta_str:>12}")
    
    return results


if __name__ == '__main__':
    # Train PPO
    model, best_cov = train_ppo()
    
    # Evaluate PPO
    ppo_cov, ppo_safety = evaluate_ppo(model)
    
    # Compare all
    results = compare_all()
