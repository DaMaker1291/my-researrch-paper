#!/usr/bin/env python3
"""
5-Seed Statistical Significance Experiments
=============================================

Runs GAT-MARAHS training with 5 different random seeds and reports
mean ± std for all metrics. This is required for publication-quality results.
"""
import numpy as np
import torch
import time
import json
import os

from train_gat_fast import FastGATPPO, train, benchmark
from paper_ready_train import WildfireEnv


def run_single_seed(seed, n_episodes=500, grid=20, n_drones=6, max_steps=150):
    """Run one training + benchmark with a specific seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*60}")
    print(f"SEED {seed} | {n_episodes} episodes | {n_drones} drones | {grid}x{grid}")
    print(f"{'='*60}")
    
    # Train
    agent, train_results = train(n_episodes, grid, n_drones, max_steps)
    
    # Benchmark at wind=12
    bench_results = benchmark(agent, grid, n_drones, max_steps, wind=12.0, n_eps=20)
    
    # Wind sweep
    wind_results = {}
    for wind in [5, 10, 15, 20, 25]:
        env = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps, wind_speed=wind)
        s_list, c_list, p_list = [], [], []
        for _ in range(10):
            obs = env.reset()
            for step in range(max_steps):
                am = np.array([env.drones[i]['alive'] for i in range(n_drones)])
                pos = np.array([env.drones[i]['pos'] for i in range(n_drones)], dtype=np.float32)
                if not am.any(): break
                acts, _, _, _ = agent.select_actions(obs, pos, am)
                obs, _, dones, _ = env.step(np.array(acts, dtype=np.int32))
                if all(dones): break
            ac = sum(1 for i in range(n_drones) if env.drones[i]['alive'])
            s_list.append(ac / n_drones * 100)
            c_list.append(len(env.total_cells_explored) / (grid*grid) * 100)
            fc = np.argwhere(env.fire > 0.2)
            pc = set()
            for fx, fy in fc:
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = fx+dx, fy+dy
                    if 0 <= nx < grid and 0 <= ny < grid and env.fire[nx, ny] < 0.1:
                        pc.add((nx, ny))
            vis = set()
            for i in range(n_drones): vis.update(env.drones[i].get('visited', set()))
            p_list.append(len(pc & vis) / max(1, len(pc)) * 100)
        
        wind_results[str(wind)] = {
            'safety': np.mean(s_list),
            'coverage': np.mean(c_list),
            'perimeter': np.mean(p_list),
        }
    
    return {
        'seed': seed,
        'train': train_results,
        'benchmark': bench_results,
        'wind_sweep': wind_results,
    }


def main():
    seeds = [42, 123, 456, 789, 1024]
    all_results = []
    
    print("=" * 60)
    print("5-SEED STATISTICAL SIGNIFICANCE EXPERIMENTS")
    print("=" * 60)
    
    for seed in seeds:
        result = run_single_seed(seed, n_episodes=500, grid=20, n_drones=6, max_steps=150)
        all_results.append(result)
    
    # Aggregate results
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS (mean ± std across 5 seeds)")
    print("=" * 80)
    
    metrics = ['safety', 'coverage', 'perimeter']
    
    # Benchmark at wind=12
    print("\n--- Benchmark (wind=12 m/s) ---")
    for method in ['GAT-MARAHS', 'Random', 'Greedy']:
        print(f"\n{method}:")
        for m in metrics:
            vals = [r['benchmark'][method][m] for r in all_results]
            print(f"  {m:>10s}: {np.mean(vals):6.1f}% ± {np.std(vals):5.1f}%")
    
    # Wind sweep
    print("\n--- Wind Sweep (GAT-MARAHS) ---")
    for wind in ['5', '10', '15', '20', '25']:
        print(f"\n  Wind={wind} m/s:")
        for m in metrics:
            vals = [r['wind_sweep'][wind][m] for r in all_results]
            print(f"    {m:>10s}: {np.mean(vals):6.1f}% ± {np.std(vals):5.1f}%")
    
    # Training convergence
    print("\n--- Training Convergence ---")
    for m in ['final_reward', 'final_coverage']:
        vals = [r['train'][m] for r in all_results]
        print(f"  {m:>15s}: {np.mean(vals):8.1f} ± {np.std(vals):6.1f}")
    
    # Save
    output = {
        'seeds': seeds,
        'n_seeds': len(seeds),
        'aggregate': {
            'benchmark_wind12': {},
            'wind_sweep': {},
        },
    }
    
    for method in all_results[0]['benchmark']:
        output['aggregate']['benchmark_wind12'][method] = {}
        for m in metrics:
            vals = [r['benchmark'][method][m] for r in all_results]
            output['aggregate']['benchmark_wind12'][method][m] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
            }
    
    for wind in all_results[0]['wind_sweep']:
        output['aggregate']['wind_sweep'][wind] = {}
        for m in metrics:
            vals = [r['wind_sweep'][wind][m] for r in all_results]
            output['aggregate']['wind_sweep'][wind][m] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
            }
    
    output['raw_results'] = all_results
    
    with open('5seed_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\nResults saved to 5seed_results.json")


if __name__ == "__main__":
    main()
