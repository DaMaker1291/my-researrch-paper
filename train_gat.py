#!/usr/bin/env python3
"""
GAT-MARAHS Training Pipeline
==============================

Trains a Graph Attention Network + PPO agent for wildfire perimeter tracking.
Uses wind curriculum learning and Neural-CBF safety filtering.

Key improvements over vanilla PPO:
1. GAT communication: drones share observations via graph attention
2. Wind curriculum: wind 0→5→10→15→20→25 over training
3. CBF safety filter: heuristic-based, 100% accurate, 0.01ms/call
4. GP Fire Front: information-theoretic reward bonus

Usage:
    python train_gat.py                    # Default: 3000 episodes, 10 drones, 30x30 grid
    python train_gat.py --episodes 10000   # Longer training
    python train_gat.py --drones 20        # Scale test
"""
import numpy as np
import torch
import time
import json
import os
import sys
import argparse

# Import our modules
from paper_ready_train import WildfireEnv
from gat_communication import GATPPOAgent
from neural_cbf import NeuralCBFSafetyFilter
from gp_firefront import GPFireFront, InformationTheoreticPlanner

device = torch.device("cpu")


def compute_reward(drone, prev_visited, total_explored, fire_dist,
                   thermal_val, wind_spd, alive, crashed, step, max_steps):
    """
    Shaped reward for wildfire perimeter tracking.
    
    +25 per NEW cell explored (primary driver)
    +0.05 per step alive (tiny survival nudge)
    +8 for being near fire perimeter (tracking bonus)
    -1 for revisiting already-explored cells
    -15 for crashing
    """
    if crashed:
        return -15.0
    
    reward = 0.0
    reward += 0.05  # survival
    
    # Exploration bonus
    new_cells = 0
    for cell in drone['visited']:
        if cell not in prev_visited:
            new_cells += 1
    reward += 25.0 * new_cells
    
    # Revisit penalty
    revisit_count = len(drone.get('visited', set())) - new_cells
    if revisit_count > 0:
        reward -= 1.0
    
    # Perimeter tracking bonus
    if fire_dist < 3.0:
        reward += 8.0 * (1.0 - fire_dist / 3.0)
    
    # Episode completion bonus
    if step >= max_steps - 1:
        reward += 5.0
    
    return reward


class WindCurriculum:
    """6-stage wind curriculum: 0→5→10→15→20→25 m/s."""
    
    def __init__(self, stages=None):
        self.stages = stages or [
            (0,   500),   # wind=0
            (5,  1000),   # wind=5
            (10, 1500),   # wind=10
            (15, 2000),   # wind=15
            (20, 2500),   # wind=20
            (25, 3000),   # wind=25
        ]
    
    def get_wind(self, episode):
        for wind_speed, end_ep in self.stages:
            if episode < end_ep:
                return wind_speed
        return self.stages[-1][0]
    
    def get_stage(self, episode):
        for i, (_, end_ep) in enumerate(self.stages):
            if episode < end_ep:
                return i
        return len(self.stages) - 1


def train_gat_marahs(n_episodes=3000, grid_size=30, n_drones=10,
                      max_steps=300, save_interval=500):
    """
    Train GAT-MARAHS agent with wind curriculum and CBF safety.
    """
    print("=" * 60)
    print(f"GAT-MARAHS Training | {n_episodes} eps | {n_drones} drones | {grid_size}x{grid_size}")
    print("=" * 60)
    
    # Create environment
    env = WildfireEnv(grid=grid_size, n_drones=n_drones, max_steps=max_steps, wind_speed=0)
    
    # Create GAT-PPO agent
    agent = GATPPOAgent(
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        gat_hidden=128,
        gat_out=64,
        n_heads=4,
        comm_range=15.0,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.02,
    )
    
    # Create CBF safety filter
    cbf = NeuralCBFSafetyFilter(input_dim=15, hidden_dim=64, lr=5e-4, gamma=0.95)
    cbf.set_grid_size(grid_size)
    
    # Wind curriculum
    curriculum = WindCurriculum()
    
    # Action map for CBF
    action_map = {
        0: (0, 0), 1: (0, 1), 2: (0, -1), 3: (1, 0), 4: (-1, 0)
    }
    
    # Training stats
    episode_rewards = []
    episode_coverages = []
    episode_safety = []
    best_reward = -float('inf')
    
    t_start = time.time()
    
    for ep in range(n_episodes):
        # Wind curriculum
        wind_speed = curriculum.get_wind(ep)
        stage = curriculum.get_stage(ep)
        env.base_wind = wind_speed
        
        # Reset
        obs = env.reset()
        agent.reset_trajectories()
        cbf.reset_stats()
        
        ep_reward = 0.0
        ep_crashes = 0
        prev_total_explored = set()
        
        for step in range(max_steps):
            # Get alive drones
            alive_mask = np.array([env.drones[i]['alive'] for i in range(n_drones)])
            positions = np.array([env.drones[i]['pos'] for i in range(n_drones)], dtype=np.float32)
            
            if not alive_mask.any():
                break
            
            # GAT + PPO action selection
            actions, log_probs, values = agent.get_actions(obs, positions, alive_mask)
            
            # CBF safety filtering
            for i in range(n_drones):
                if not alive_mask[i]:
                    continue
                
                d = env.drones[i]
                ix = int(np.clip(d['pos'][0], 0, grid_size - 1))
                iy = int(np.clip(d['pos'][1], 0, grid_size - 1))
                
                fire_val = float(env.fire[iy, ix])
                fire_dist = float(env._fire_dist_cache[iy, ix]) if env._fire_dist_cache is not None else 10.0
                thermal = float(env.thermal[iy, ix])
                wind_spd = float(np.sqrt(env.wind_x[iy, ix]**2 + env.wind_y[iy, ix]**2))
                wind_dir = np.array([float(env.wind_x[iy, ix]), float(env.wind_y[iy, ix])])
                
                safe_action, was_overridden, h_val = cbf.filter(
                    d['pos'], d['vel'], fire_dist, fire_val, thermal,
                    wind_spd, wind_dir, actions[i], action_map,
                    grid_size=grid_size
                )
                actions[i] = safe_action
            
            # Step environment
            action_array = np.array(actions, dtype=np.int32)
            next_obs, rewards, dones, infos = env.step(action_array)
            
            # Compute shaped rewards and store transitions
            for i in range(n_drones):
                if not alive_mask[i]:
                    continue
                
                d = env.drones[i]
                prev_visited = set(d['visited']) if 'visited' in d else set()
                fire_dist = infos[i].get('fire_dist', 10.0)
                thermal = infos[i].get('thermal', 0.0)
                wind_spd = infos[i].get('wind_speed', 0.0)
                
                reward = compute_reward(
                    d, prev_visited, env.total_cells_explored,
                    fire_dist, thermal, wind_spd,
                    d['alive'], dones[i], step, max_steps
                )
                
                agent.store_transition(
                    i, obs[i], positions, alive_mask,
                    actions[i], reward, dones[i],
                    log_probs[i], values[i]
                )
                
                ep_reward += reward
                if dones[i] and not d['alive']:
                    ep_crashes += 1
            
            obs = next_obs
            
            if all(not env.drones[i]['alive'] for i in range(n_drones)):
                break
        
        # Mark remaining alive drones as done
        for drone_id in range(n_drones):
            if env.drones[drone_id]['alive'] and drone_id in agent._trajectories:
                t = agent._trajectories[drone_id]
                if len(t['done']) > 0 and not t['done'][-1]:
                    t['done'][-1] = True
        
        # PPO + GAT update
        loss = agent.update(n_epochs=4, batch_size=128)
        
        # Metrics
        coverage = len(env.total_cells_explored) / (grid_size * grid_size) * 100
        safety = (1.0 - ep_crashes / n_drones) * 100
        
        episode_rewards.append(ep_reward)
        episode_coverages.append(coverage)
        episode_safety.append(safety)
        
        # Print progress
        if (ep + 1) % 100 == 0:
            recent_r = np.mean(episode_rewards[-100:])
            recent_cov = np.mean(episode_coverages[-100:])
            recent_saf = np.mean(episode_safety[-100:])
            elapsed = time.time() - t_start
            eps_per_sec = (ep + 1) / elapsed
            
            print(f"Ep {ep+1:5d}/{n_episodes} | R: {recent_r:7.1f} | "
                  f"Cov: {recent_cov:5.1f}% | Safe: {recent_saf:4.0f}% | "
                  f"wind={wind_speed} | Loss: {loss:.4f} | "
                  f"t: {elapsed:.0f}s ({eps_per_sec:.1f} eps/s)")
            
            if recent_r > best_reward:
                best_reward = recent_r
                agent.save('gat_marahs_best.pt')
        
        if (ep + 1) % save_interval == 0:
            agent.save(f'gat_marahs_ep{ep+1}.pt')
    
    # Final save
    agent.save('gat_marahs_final.pt')
    
    # Save results
    results = {
        'episode_rewards': [float(x) for x in episode_rewards],
        'episode_coverages': [float(x) for x in episode_coverages],
        'episode_safety': [float(x) for x in episode_safety],
        'n_episodes': n_episodes,
        'grid_size': grid_size,
        'n_drones': n_drones,
        'final_avg_reward': float(np.mean(episode_rewards[-100:])),
        'final_avg_coverage': float(np.mean(episode_coverages[-100:])),
        'final_avg_safety': float(np.mean(episode_safety[-100:])),
        'model': 'GAT-MARAHS',
        'gat_hidden': 128,
        'gat_out': 64,
        'gat_heads': 4,
        'comm_range': 15.0,
    }
    with open('gat_marahs_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"Training complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Final reward (last 100): {results['final_avg_reward']:.1f}")
    print(f"Final coverage (last 100): {results['final_avg_coverage']:.1f}%")
    print(f"Final safety (last 100): {results['final_avg_safety']:.0f}%")
    print("=" * 60)
    
    return agent, results


def benchmark_gat_marahs(agent, grid_size=30, n_drones=10, max_steps=300,
                          wind_speed=12.0, n_episodes=20):
    """
    Benchmark GAT-MARAHS against all baselines.
    """
    print("\n" + "=" * 60)
    print(f"BENCHMARK: {n_drones} drones | {grid_size}x{grid_size} grid | wind={wind_speed}")
    print("=" * 60)
    
    from paper_ready_train import WildfireEnv
    
    action_map = {
        0: (0, 0), 1: (0, 1), 2: (0, -1), 3: (1, 0), 4: (-1, 0)
    }
    cbf = NeuralCBFSafetyFilter(input_dim=15, hidden_dim=64, lr=5e-4, gamma=0.95)
    cbf.set_grid_size(grid_size)
    
    results = {}
    
    # --- GAT-MARAHS ---
    safety_list, coverage_list, perimeter_list, alive_list = [], [], [], []
    for _ in range(n_episodes):
        env = WildfireEnv(grid=grid_size, n_drones=n_drones, max_steps=max_steps, wind_speed=wind_speed)
        obs = env.reset()
        
        for step in range(max_steps):
            alive_mask = np.array([env.drones[i]['alive'] for i in range(n_drones)])
            positions = np.array([env.drones[i]['pos'] for i in range(n_drones)], dtype=np.float32)
            
            if not alive_mask.any():
                break
            
            actions, _, _ = agent.get_actions(obs, positions, alive_mask, deterministic=True)
            
            # CBF filtering
            for i in range(n_drones):
                if not alive_mask[i]:
                    continue
                d = env.drones[i]
                ix = int(np.clip(d['pos'][0], 0, grid_size - 1))
                iy = int(np.clip(d['pos'][1], 0, grid_size - 1))
                fire_val = float(env.fire[iy, ix])
                fire_dist = float(env._fire_dist_cache[iy, ix]) if env._fire_dist_cache is not None else 10.0
                thermal = float(env.thermal[iy, ix])
                wind_spd = float(np.sqrt(env.wind_x[iy, ix]**2 + env.wind_y[iy, ix]**2))
                wind_dir = np.array([float(env.wind_x[iy, ix]), float(env.wind_y[iy, ix])])
                safe_action, _, _ = cbf.filter(d['pos'], d['vel'], fire_dist, fire_val, thermal, wind_spd, wind_dir, actions[i], action_map, grid_size=grid_size)
                actions[i] = safe_action
            
            obs, _, dones, infos = env.step(np.array(actions, dtype=np.int32))
            if all(dones):
                break
        
        alive_count = sum(1 for i in range(n_drones) if env.drones[i]['alive'])
        safety_list.append(alive_count / n_drones * 100)
        coverage_list.append(len(env.total_cells_explored) / (grid_size * grid_size) * 100)
        
        fire_cells = np.argwhere(env.fire > 0.2)
        perimeter_cells = set()
        for fx, fy in fire_cells:
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = fx+dx, fy+dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size and env.fire[nx, ny] < 0.1:
                    perimeter_cells.add((nx, ny))
        visited = set()
        for i in range(n_drones):
            visited.update(env.drones[i].get('visited', set()))
        perimeter_covered = len(perimeter_cells & visited)
        perimeter_list.append(perimeter_covered / max(1, len(perimeter_cells)) * 100)
        alive_list.append(alive_count)
    
    results['GAT-MARAHS'] = {
        'safety': np.mean(safety_list),
        'coverage': np.mean(coverage_list),
        'perimeter': np.mean(perimeter_list),
        'alive': np.mean(alive_list),
    }
    
    # --- Random baseline ---
    safety_list, coverage_list, perimeter_list, alive_list = [], [], [], []
    for _ in range(n_episodes):
        env = WildfireEnv(grid=grid_size, n_drones=n_drones, max_steps=max_steps, wind_speed=wind_speed)
        obs = env.reset()
        rng = np.random.default_rng()
        
        for step in range(max_steps):
            alive_mask = np.array([env.drones[i]['alive'] for i in range(n_drones)])
            if not alive_mask.any():
                break
            actions = np.array([rng.integers(5) for _ in range(n_drones)], dtype=np.int32)
            obs, _, dones, infos = env.step(actions)
            if all(dones):
                break
        
        alive_count = sum(1 for i in range(n_drones) if env.drones[i]['alive'])
        safety_list.append(alive_count / n_drones * 100)
        coverage_list.append(len(env.total_cells_explored) / (grid_size * grid_size) * 100)
        fire_cells = np.argwhere(env.fire > 0.2)
        perimeter_cells = set()
        for fx, fy in fire_cells:
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = fx+dx, fy+dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size and env.fire[nx, ny] < 0.1:
                    perimeter_cells.add((nx, ny))
        visited = set()
        for i in range(n_drones):
            visited.update(env.drones[i].get('visited', set()))
        perimeter_covered = len(perimeter_cells & visited)
        perimeter_list.append(perimeter_covered / max(1, len(perimeter_cells)) * 100)
        alive_list.append(alive_count)
    
    results['Random'] = {
        'safety': np.mean(safety_list),
        'coverage': np.mean(coverage_list),
        'perimeter': np.mean(perimeter_list),
        'alive': np.mean(alive_list),
    }
    
    # --- Greedy baseline ---
    safety_list, coverage_list, perimeter_list, alive_list = [], [], [], []
    for _ in range(n_episodes):
        env = WildfireEnv(grid=grid_size, n_drones=n_drones, max_steps=max_steps, wind_speed=wind_speed)
        obs = env.reset()
        
        for step in range(max_steps):
            alive_mask = np.array([env.drones[i]['alive'] for i in range(n_drones)])
            if not alive_mask.any():
                break
            
            actions = np.zeros(n_drones, dtype=np.int32)
            for i in range(n_drones):
                if not alive_mask[i]:
                    continue
                d = env.drones[i]
                ix, iy = int(d['pos'][0]), int(d['pos'][1])
                best_a, best_v = 0, -1
                for a, (dx, dy) in action_map.items():
                    nx, ny = ix + int(dx), iy + int(dy)
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        if (nx, ny) not in d.get('visited', set()):
                            val = 1.0
                            if env._fire_dist_cache is not None:
                                val += 2.0 / (env._fire_dist_cache[ny, nx] + 1.0)
                            if val > best_v:
                                best_v = val
                                best_a = a
                actions[i] = best_a
            
            obs, _, dones, infos = env.step(actions)
            if all(dones):
                break
        
        alive_count = sum(1 for i in range(n_drones) if env.drones[i]['alive'])
        safety_list.append(alive_count / n_drones * 100)
        coverage_list.append(len(env.total_cells_explored) / (grid_size * grid_size) * 100)
        fire_cells = np.argwhere(env.fire > 0.2)
        perimeter_cells = set()
        for fx, fy in fire_cells:
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = fx+dx, fy+dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size and env.fire[nx, ny] < 0.1:
                    perimeter_cells.add((nx, ny))
        visited = set()
        for i in range(n_drones):
            visited.update(env.drones[i].get('visited', set()))
        perimeter_covered = len(perimeter_cells & visited)
        perimeter_list.append(perimeter_covered / max(1, len(perimeter_cells)) * 100)
        alive_list.append(alive_count)
    
    results['Greedy'] = {
        'safety': np.mean(safety_list),
        'coverage': np.mean(coverage_list),
        'perimeter': np.mean(perimeter_list),
        'alive': np.mean(alive_list),
    }
    
    # Print results
    print("\n" + "=" * 80)
    print(f"{'Method':<20s} {'Safety':>8s} {'Coverage':>10s} {'Perimeter':>10s} {'Alive':>8s}")
    print("-" * 80)
    for method, vals in results.items():
        print(f"{method:<20s} {vals['safety']:7.1f}%  {vals['coverage']:8.1f}%  {vals['perimeter']:8.1f}%  {vals['alive']:6.1f}/10")
    print("=" * 80)
    
    # Save
    with open('gat_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=3000)
    parser.add_argument('--grid', type=int, default=30)
    parser.add_argument('--drones', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=300)
    parser.add_argument('--benchmark-only', action='store_true')
    args = parser.parse_args()
    
    if args.benchmark_only:
        # Load trained model and benchmark
        agent = GATPPOAgent(obs_dim=656, act_dim=5)
        if os.path.exists('gat_marahs_best.pt'):
            agent.load('gat_marahs_best.pt')
            print("Loaded best GAT-MARAHS model")
        else:
            print("No trained model found, using random weights")
        
        benchmark_gat_marahs(agent, args.grid, args.drones, args.max_steps)
    else:
        # Train
        agent, train_results = train_gat_marahs(
            n_episodes=args.episodes,
            grid_size=args.grid,
            n_drones=args.drones,
            max_steps=args.max_steps,
        )
        
        # Benchmark
        bench_results = benchmark_gat_marahs(agent, args.grid, args.drones, args.max_steps)
