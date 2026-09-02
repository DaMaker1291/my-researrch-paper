#!/usr/bin/env python3
"""
Fast GAT-MARAHS Training — Optimized for speed
================================================

Key optimization: instead of re-running GAT during PPO update,
we cache the GAT-enhanced observations during trajectory collection.
This makes the update O(1) per transition instead of O(K²) for graph ops.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import os
import sys

from paper_ready_train import WildfireEnv
from gat_communication import GATCommunication, PPONetwork

device = torch.device("cpu")


class FastGATPPO:
    """PPO agent that caches GAT-enhanced observations during collection."""

    def __init__(self, obs_dim=656, act_dim=5, gat_hidden=128, gat_out=64,
                 n_heads=4, comm_range=15.0, lr=3e-4):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        self.gat = GATCommunication(obs_dim, gat_hidden, gat_out, n_heads, comm_range)
        enhanced_dim = obs_dim + gat_out
        self.policy = PPONetwork(enhanced_dim, act_dim)
        
        params = list(self.gat.parameters()) + list(self.policy.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr, eps=1e-5)
        
        self._traj = []  # List of (enhanced_obs, action, reward, done, log_prob, value)
    
    def select_actions(self, obs, positions, alive_mask):
        """Select actions for all drones, returning enhanced obs for caching."""
        obs_t = torch.tensor(obs, device=device, dtype=torch.float32)
        pos_t = torch.tensor(positions, device=device, dtype=torch.float32)
        
        enhanced = self.gat(obs_t, pos_t, alive_mask)
        
        with torch.no_grad():
            logits, values = self.policy(enhanced)
            probs = torch.distributions.Categorical(logits=logits)
            actions = probs.sample()
            log_probs = probs.log_prob(actions)
        
        return (actions.cpu().numpy(), log_probs.cpu().numpy(), 
                values.cpu().numpy(), enhanced.detach())
    
    def store(self, enhanced_obs, actions, rewards, dones, log_probs, values, agent_ids=None):
        """Store a batch of transitions with cached enhanced observations."""
        K = len(actions)
        for i in range(K):
            self._traj.append({
                'obs': enhanced_obs[i],
                'action': actions[i],
                'reward': rewards[i],
                'done': dones[i],
                'log_prob': log_probs[i],
                'value': values[i],
                'agent_id': agent_ids[i] if agent_ids is not None else 0,
            })
    
    def update(self, gamma=0.99, gae_lambda=0.95, n_epochs=4,
               batch_size=256, clip_eps=0.2, entropy_coef=0.02, max_grad=0.5):
        """PPO update using cached enhanced observations."""
        if len(self._traj) == 0:
            return 0.0
        
        n = len(self._traj)
        rewards = np.array([t['reward'] for t in self._traj], dtype=np.float32)
        values = np.array([t['value'] for t in self._traj], dtype=np.float32)
        dones = np.array([t['done'] for t in self._traj], dtype=np.float32)
        
        # Per-drone GAE: transitions are interleaved across agents,
        # so compute GAE within each agent's sequence separately
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        agent_groups = {}
        for i, t in enumerate(self._traj):
            aid = t.get('agent_id', 0)
            if aid not in agent_groups:
                agent_groups[aid] = []
            agent_groups[aid].append(i)
        for aid, indices in agent_groups.items():
            gae = 0.0
            for k in reversed(range(len(indices))):
                idx = indices[k]
                next_idx = indices[k+1] if k+1 < len(indices) else None
                next_val = 0.0 if next_idx is None else self._traj[next_idx]['value']
                next_done = 1.0 if next_idx is None else self._traj[next_idx]['done']
                delta = self._traj[idx]['reward'] + gamma * next_val * (1-next_done) - self._traj[idx]['value']
                gae = delta + gamma * gae_lambda * (1-next_done) * gae
                advantages[idx] = gae
                returns[idx] = gae + self._traj[idx]['value']
        
        # Build tensors from cached data
        all_obs = torch.stack([t['obs'].squeeze(0) for t in self._traj])
        all_actions = torch.tensor([t['action'] for t in self._traj], device=device, dtype=torch.long)
        all_old_lp = torch.tensor([t['log_prob'] for t in self._traj], device=device, dtype=torch.float32)
        all_adv = torch.tensor(advantages, device=device, dtype=torch.float32)
        all_ret = torch.tensor(returns, device=device, dtype=torch.float32)
        
        # Normalize advantages
        all_adv = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-8)
        
        total_loss = 0.0
        count = 0
        
        for _ in range(n_epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = perm[start:end]
                
                _, new_lp, entropy, vals = self.policy.evaluate(all_obs[idx], all_actions[idx])
                
                ratio = torch.exp(new_lp - all_old_lp[idx])
                s1 = ratio * all_adv[idx]
                s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * all_adv[idx]
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = F.mse_loss(vals, all_ret[idx])
                entropy_bonus = -entropy.mean()
                
                loss = policy_loss + 0.5 * value_loss + entropy_coef * entropy_bonus
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.gat.parameters()) + list(self.policy.parameters()), max_grad)
                self.optimizer.step()
                
                total_loss += loss.item()
                count += 1
        
        self._traj.clear()
        return total_loss / max(1, count)
    
    def save(self, path):
        torch.save({'gat': self.gat.state_dict(), 'policy': self.policy.state_dict()}, path)
    
    def load(self, path):
        ckpt = torch.load(path, map_location=device)
        self.gat.load_state_dict(ckpt['gat'])
        self.policy.load_state_dict(ckpt['policy'])


def compute_reward(drone, drone_idx, all_drones, prev_visited, fire_dist, crashed, step, max_steps, grid_size):
    """Balanced reward: survival-first exploration for multi-agent wildfire tracking.

    Incentive hierarchy:
      1. Staying alive (+1.0/step, +30 at episode end) — dominant signal
      2. Exploring new cells (+8/cell) — worth pursuing, not worth dying for
      3. Fire-front observation (+5 when close) — mild tracking nudge
      4. Coordination penalties — avoid clustering (capped)
    """
    if crashed: return -40.0  # strong penalty: crashing is always bad

    reward = 0.0

    # 1. Per-step survival reward (dominant: ~300 over a full episode)
    reward += 1.0

    # 2. Exploration: +8 per NEW cell the team hasn't visited
    new_cells = sum(1 for c in drone['visited'] if c not in prev_visited)
    reward += 8.0 * new_cells

    # 3. Fire front bonus: +5 for being near fire (mild tracking nudge)
    if 0.5 < fire_dist < 4.0:
        reward += 5.0 * (1.0 - fire_dist / 4.0)

    # 4. Overlap penalty: -1.0 per drone within range 2, capped at -3.0
    nearby = 0
    for j, other in enumerate(all_drones):
        if j != drone_idx and other['alive']:
            if np.linalg.norm(drone['pos'] - other['pos']) < 2.0:
                nearby += 1
    reward -= min(3.0, 1.0 * nearby)

    # 5. Quadrant diversity: +2 / count in same quadrant
    mid = grid_size / 2.0
    q = int(drone['pos'][0] >= mid) + 2 * int(drone['pos'][1] >= mid)
    qcount = sum(1 for o in all_drones if o['alive'] and
                 int(o['pos'][0] >= mid) + 2*int(o['pos'][1] >= mid) == q)
    reward += 2.0 / max(1, qcount)

    # 6. Episode completion bonus (large: rewards patience)
    if step >= max_steps - 1:
        reward += 30.0

    return reward


def train(n_episodes=3000, grid=30, n_drones=10, max_steps=300):
    print("=" * 60)
    print(f"GAT-MARAHS Fast Training | {n_episodes} eps | {n_drones} drones | {grid}x{grid}")
    print("=" * 60)
    
    env = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps, wind_speed=0)
    agent = FastGATPPO(obs_dim=env.obs_dim, act_dim=env.act_dim)
    
    # Wind curriculum
    stages = [(0, 500), (5, 1000), (10, 1500), (15, 2000), (20, 2500), (25, 3000)]
    
    rewards_history = []
    coverage_history = []
    safety_history = []
    best_r = -float('inf')
    early_stop_patience = 500  # stop if coverage >= 85% for this many consecutive episodes
    early_stop_counter = 0
    early_stop_target = 85.0
    
    t0 = time.time()
    
    for ep in range(n_episodes):
        # Wind
        wind = 0
        for w, end in stages:
            if ep < end:
                wind = w
                break
        env.base_wind = wind
        
        obs = env.reset()
        agent._traj.clear()
        
        ep_r = 0.0
        ep_crashes = 0
        
        for step in range(max_steps):
            alive_mask = np.array([env.drones[i]['alive'] for i in range(n_drones)])
            positions = np.array([env.drones[i]['pos'] for i in range(n_drones)], dtype=np.float32)
            
            if not alive_mask.any():
                break
            
            actions, log_probs, values, enhanced = agent.select_actions(obs, positions, alive_mask)
            prev_visited = [set(env.drones[i].get('visited', set())) for i in range(n_drones)]
            obs_next, rewards_env, dones, infos = env.step(np.array(actions, dtype=np.int32))
            
            # Compute shaped rewards
            shaped_rewards = np.zeros(n_drones, dtype=np.float32)
            for i in range(n_drones):
                if not alive_mask[i]:
                    continue
                d = env.drones[i]
                prev_v = prev_visited[i]
                fd = infos[i].get('fire_dist', 10.0)
                crashed = infos[i].get('crashed', False)
                shaped_rewards[i] = compute_reward(d, i, env.drones, prev_v, fd, crashed, step, max_steps, grid)
                ep_r += shaped_rewards[i]
                if crashed: ep_crashes += 1
            
            # Store with enhanced obs (include agent_ids for per-drone GAE)
            agent.store(enhanced, actions, shaped_rewards, dones.astype(np.float32), log_probs, values, agent_ids=list(range(n_drones)))
            
            obs = obs_next
            if all(dones):
                break
        
        # PPO update
        loss = agent.update()
        
        cov = len(env.total_cells_explored) / (grid * grid) * 100
        saf = (1.0 - ep_crashes / n_drones) * 100
        
        rewards_history.append(ep_r)
        coverage_history.append(cov)
        safety_history.append(saf)
        
        if (ep + 1) % 100 == 0:
            avg_r = np.mean(rewards_history[-100:])
            avg_cov = np.mean(coverage_history[-100:])
            avg_saf = np.mean(safety_history[-100:])
            elapsed = time.time() - t0
            eps_per_s = (ep + 1) / elapsed
            print(f"Ep {ep+1:5d}/{n_episodes} | R: {avg_r:7.1f} | "
                  f"Cov: {avg_cov:5.1f}% | Safe: {avg_saf:4.0f}% | "
                  f"wind={wind} | Loss: {loss:.4f} | {elapsed:.0f}s ({eps_per_s:.1f} ep/s)")
            
            if avg_r > best_r:
                best_r = avg_r
                agent.save('gat_marahs_best.pt')
            if avg_cov >= early_stop_target:
                early_stop_counter += 100
            else:
                early_stop_counter = 0
            if early_stop_counter >= early_stop_patience:
                print(f"Early stop at ep {ep+1}: coverage {avg_cov:.1f}% >= {early_stop_target}% for {early_stop_patience} consecutive episodes")
                break
        
        if (ep + 1) % 500 == 0:
            agent.save(f'gat_marahs_ep{ep+1}.pt')
    
    agent.save('gat_marahs_final.pt')
    
    results = {
        'n_episodes': n_episodes,
        'final_reward': float(np.mean(rewards_history[-100:])),
        'final_coverage': float(np.mean(coverage_history[-100:])),
        'final_safety': float(np.mean(safety_history[-100:])),
        'rewards': [float(x) for x in rewards_history],
        'coverages': [float(x) for x in coverage_history],
        'safety': [float(x) for x in safety_history],
    }
    with open('gat_marahs_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Reward: {results['final_reward']:.1f}")
    print(f"Coverage: {results['final_coverage']:.1f}%")
    print(f"Safety: {results['final_safety']:.0f}%")
    
    return agent, results


def benchmark(agent, grid=30, n_drones=10, max_steps=300, wind=12.0, n_eps=20):
    """Quick benchmark of GAT-MARAHS vs baselines."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK | wind={wind} | {n_drones} drones | {n_eps} episodes")
    print(f"{'='*60}")
    
    action_map = {0: (0, 0), 1: (0, 1), 2: (0, -1), 3: (1, 0), 4: (-1, 0)}
    
    results = {}
    
    # GAT-MARAHS
    s, c, p, a = [], [], [], []
    for _ in range(n_eps):
        env = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps, wind_speed=wind)
        obs = env.reset()
        for step in range(max_steps):
            am = np.array([env.drones[i]['alive'] for i in range(n_drones)])
            pos = np.array([env.drones[i]['pos'] for i in range(n_drones)], dtype=np.float32)
            if not am.any(): break
            acts, _, _, _ = agent.select_actions(obs, pos, am)
            obs, _, dones, _ = env.step(np.array(acts, dtype=np.int32))
            if all(dones): break
        ac = sum(1 for i in range(n_drones) if env.drones[i]['alive'])
        s.append(ac / n_drones * 100)
        c.append(len(env.total_cells_explored) / (grid*grid) * 100)
        # Perimeter
        fc = np.argwhere(env.fire > 0.2)
        pc = set()
        for fx, fy in fc:
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = fx+dx, fy+dy
                if 0 <= nx < grid and 0 <= ny < grid and env.fire[nx, ny] < 0.1:
                    pc.add((nx, ny))
        vis = set()
        for i in range(n_drones): vis.update(env.drones[i].get('visited', set()))
        p.append(len(pc & vis) / max(1, len(pc)) * 100)
        a.append(ac)
    results['GAT-MARAHS'] = {'safety': np.mean(s), 'coverage': np.mean(c), 'perimeter': np.mean(p), 'alive': np.mean(a)}
    
    # Random
    s, c, p, a = [], [], [], []
    for _ in range(n_eps):
        env = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps, wind_speed=wind)
        obs = env.reset()
        rng = np.random.default_rng()
        for step in range(max_steps):
            am = np.array([env.drones[i]['alive'] for i in range(n_drones)])
            if not am.any(): break
            acts = np.array([rng.integers(5) for _ in range(n_drones)], dtype=np.int32)
            obs, _, dones, _ = env.step(acts)
            if all(dones): break
        ac = sum(1 for i in range(n_drones) if env.drones[i]['alive'])
        s.append(ac / n_drones * 100)
        c.append(len(env.total_cells_explored) / (grid*grid) * 100)
        fc = np.argwhere(env.fire > 0.2)
        pc = set()
        for fx, fy in fc:
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = fx+dx, fy+dy
                if 0 <= nx < grid and 0 <= ny < grid and env.fire[nx, ny] < 0.1:
                    pc.add((nx, ny))
        vis = set()
        for i in range(n_drones): vis.update(env.drones[i].get('visited', set()))
        p.append(len(pc & vis) / max(1, len(pc)) * 100)
        a.append(ac)
    results['Random'] = {'safety': np.mean(s), 'coverage': np.mean(c), 'perimeter': np.mean(p), 'alive': np.mean(a)}
    
    # Greedy
    s, c, p, a = [], [], [], []
    for _ in range(n_eps):
        env = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps, wind_speed=wind)
        obs = env.reset()
        for step in range(max_steps):
            am = np.array([env.drones[i]['alive'] for i in range(n_drones)])
            if not am.any(): break
            acts = np.zeros(n_drones, dtype=np.int32)
            for i in range(n_drones):
                if not am[i]: continue
                d = env.drones[i]
                ix, iy = int(d['pos'][0]), int(d['pos'][1])
                best_a, best_v = 0, -1
                for ai, (dx, dy) in action_map.items():
                    nx, ny = ix+int(dx), iy+int(dy)
                    if 0 <= nx < grid and 0 <= ny < grid and (nx,ny) not in d.get('visited', set()):
                        v = 1.0
                        if env._fire_dist_cache is not None:
                            v += 2.0 / (env._fire_dist_cache[ny, nx] + 1.0)
                        if v > best_v: best_v, best_a = v, ai
                acts[i] = best_a
            obs, _, dones, _ = env.step(acts)
            if all(dones): break
        ac = sum(1 for i in range(n_drones) if env.drones[i]['alive'])
        s.append(ac / n_drones * 100)
        c.append(len(env.total_cells_explored) / (grid*grid) * 100)
        fc = np.argwhere(env.fire > 0.2)
        pc = set()
        for fx, fy in fc:
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = fx+dx, fy+dy
                if 0 <= nx < grid and 0 <= ny < grid and env.fire[nx, ny] < 0.1:
                    pc.add((nx, ny))
        vis = set()
        for i in range(n_drones): vis.update(env.drones[i].get('visited', set()))
        p.append(len(pc & vis) / max(1, len(pc)) * 100)
        a.append(ac)
    results['Greedy'] = {'safety': np.mean(s), 'coverage': np.mean(c), 'perimeter': np.mean(p), 'alive': np.mean(a)}
    
    print(f"\n{'Method':<20s} {'Safety':>8s} {'Coverage':>10s} {'Perimeter':>10s} {'Alive':>8s}")
    print("-" * 60)
    for m, v in results.items():
        print(f"{m:<20s} {v['safety']:7.1f}%  {v['coverage']:8.1f}%  {v['perimeter']:8.1f}%  {v['alive']:6.1f}/10")
    print("=" * 60)
    
    with open('gat_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=3000)
    parser.add_argument('--grid', type=int, default=30)
    parser.add_argument('--drones', type=int, default=10)
    args = parser.parse_args()
    
    agent, train_res = train(args.episodes, args.grid, args.drones)
    bench_res = benchmark(agent, args.grid, args.drones)
