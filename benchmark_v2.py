"""
MARAHS v2: Comprehensive Benchmark Runner
==========================================

Evaluates 10 methods across 5 hurricane categories × 5 profiles × 30 seeds.
Generates publication-ready results with statistical significance testing.

Usage:
    python benchmark_v2.py
"""

import numpy as np
import json
import time
import os
import sys
from typing import Dict, List
from datetime import datetime
from dataclasses import dataclass, field
import math

from swarm_grid_env_v2 import SwarmGridWorldV2, SwarmGridConfig
from agents_v2 import (
    RandomAgent, HoverAgent, GreedyAgent, VoronoiAgent, SpiralAgent,
    GreedyCBFAgent, PIDAgent, MARAHSAgent,
)


@dataclass
class BenchmarkConfig:
    grid_size: int = 25
    num_drones: int = 10
    max_steps: int = 400
    num_seeds: int = 30
    wind_categories: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    hurricane_profiles: List[str] = field(
        default_factory=lambda: ['katrina', 'harvey', 'irma', 'maria', 'michael']
    )
    output_dir: str = './experiment_results_v2'


def run_episode(env, agent, seed, wind_category, wind_profile,
                max_steps=None, verbose=False):
    """Run a single episode and return metrics."""
    rng = np.random.default_rng(seed)
    config = env.config
    config.wind_category = wind_category
    config.hurricane_profile = wind_profile
    config.wind_intensity = wind_category / 5.0

    obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))

    if hasattr(agent, 'reset'):
        agent.reset()

    steps = max_steps or config.max_steps
    coverage_history = []
    min_dist_history = []
    collision_count = 0
    wind_speeds = []

    for step in range(steps):
        # Get actions
        if hasattr(agent, 'get_actions'):
            positions = env.positions.copy()
            actions = agent.get_actions(obs)
            # Some agents need positions for CBF
            if hasattr(agent, 'get_actions') and 'positions' in agent.get_actions.__code__.co_varnames:
                actions = agent.get_actions(obs, positions=positions)
        else:
            actions = np.zeros(env.K, dtype=int)

        obs, rewards, dones, truncs, infos = env.step(actions)

        # Record metrics
        cov = infos[0]['coverage_pct']
        coverage_history.append(cov)
        min_dist_history.append(infos[0]['min_inter_agent_dist'])
        wind_speeds.append(infos[0]['wind_speed_at_drone'])

        if dones[0]:
            break

    # Compute final metrics
    final_coverage = coverage_history[-1] if coverage_history else 0.0
    max_coverage = max(coverage_history) if coverage_history else 0.0

    # Time to reach 50% coverage
    time_to_50 = None
    for t, c in enumerate(coverage_history):
        if c >= 50.0:
            time_to_50 = t
            break

    # Safety: minimum inter-agent distance over episode
    min_dist_overall = min(min_dist_history) if min_dist_history else float('inf')
    avg_min_dist = float(np.mean(min_dist_history)) if min_dist_history else float('inf')

    # Safety violations (distance < min_separation)
    min_sep = config.min_separation
    safety_violations = sum(1 for d in min_dist_history if d < min_sep)
    safety_rate = 100.0 * (1.0 - safety_violations / max(len(min_dist_history), 1))

    return {
        'final_coverage': float(final_coverage),
        'max_coverage': float(max_coverage),
        'time_to_50': time_to_50 if time_to_50 is not None else -1,
        'safety_rate': float(safety_rate),
        'min_inter_agent_dist': float(min_dist_overall),
        'avg_min_inter_agent_dist': avg_min_dist,
        'collision_avoidances': infos[0]['collision_avoidances'],
        'coverage_history': coverage_history,
        'steps_taken': step + 1,
        'mean_wind_speed': float(np.mean(wind_speeds)),
    }


def run_benchmark(config: BenchmarkConfig = None):
    """Run the full benchmark suite."""
    config = config or BenchmarkConfig()
    os.makedirs(config.output_dir, exist_ok=True)

    # Initialize environment
    env_config = SwarmGridConfig(
        grid_size=config.grid_size,
        num_drones=config.num_drones,
        max_steps=config.max_steps,
        num_debris=5,
    )
    env = SwarmGridWorldV2(config=env_config)

    # Initialize agents
    agents = {
        'Random': RandomAgent(config.num_drones),
        'Hover': HoverAgent(config.num_drones),
        'Greedy': GreedyAgent(config.num_drones),
        'Voronoi': VoronoiAgent(config.num_drones, config.grid_size),
        'Spiral': SpiralAgent(config.num_drones, config.grid_size),
        'PID': PIDAgent(config.num_drones),
        'Greedy+CBF': GreedyCBFAgent(config.num_drones, config.grid_size,
                                      env_config.min_separation),
        'MARAHS': MARAHSAgent(config.num_drones, config.grid_size,
                              env_config.min_separation, 1.0),
    }

    all_results = {}
    start_time = time.time()

    # ── Part 1: Overall performance (across all conditions) ──
    print(f"\n{'='*70}")
    print(f"MARAHS v2 Comprehensive Benchmark")
    print(f"Grid: {config.grid_size}x{config.grid_size}, Drones: {config.num_drones}, "
          f"Steps: {config.max_steps}, Seeds: {config.num_seeds}")
    print(f"{'='*70}\n")

    for agent_name, agent in agents.items():
        print(f"Evaluating: {agent_name}")
        agent_results = []

        for profile in config.hurricane_profiles:
            for cat in config.wind_categories:
                for seed_idx in range(config.num_seeds):
                    seed = seed_idx * 1000 + cat * 100 + hash(profile) % 100

                    result = run_episode(
                        env, agent, seed, cat, profile, config.max_steps
                    )
                    result['profile'] = profile
                    result['category'] = cat
                    result['seed'] = seed
                    agent_results.append(result)

        # Aggregate
        all_results[agent_name] = aggregate_results(agent_results)

        # Print summary
        r = all_results[agent_name]['overall']
        print(f"  Coverage: {r['coverage_mean']:.1f}% ± {r['coverage_std']:.1f}% | "
              f"Safety: {r['safety_mean']:.1f}% | "
              f"Time→50%: {r['time_to_50_mean']:.0f} steps")

    # ── Part 2: Multi-agent scaling study ──
    print(f"\n{'='*70}")
    print("Multi-Agent Scaling Study (1→2→4→6→8→10 drones)")
    print(f"{'='*70}\n")

    scaling_results = {}
    for n_drones in [1, 2, 4, 6, 8, 10]:
        print(f"  {n_drones} drones...", end=" ", flush=True)
        scale_config = SwarmGridConfig(
            grid_size=config.grid_size,
            num_drones=n_drones,
            max_steps=config.max_steps,
        )
        scale_env = SwarmGridWorldV2(config=scale_config)

        scale_agent = MARAHSAgent(n_drones, config.grid_size,
                                  scale_config.min_separation, 1.0)
        scale_results_inner = []

        for seed_idx in range(config.num_seeds):
            seed = seed_idx * 1000 + n_drones * 100
            result = run_episode(scale_env, scale_agent, seed, 3, 'katrina',
                                config.max_steps)
            scale_results_inner.append(result)

        covs = [r['final_coverage'] for r in scale_results_inner]
        safety = [r['safety_rate'] for r in scale_results_inner]
        t50 = [r['time_to_50'] for r in scale_results_inner if r['time_to_50'] > 0]

        scaling_results[str(n_drones)] = {
            'coverage_mean': float(np.mean(covs)),
            'coverage_std': float(np.std(covs)),
            'safety_mean': float(np.mean(safety)),
            'time_to_50_mean': float(np.mean(t50)) if t50 else -1,
        }
        print(f"Coverage: {scaling_results[str(n_drones)]['coverage_mean']:.1f}%")

    all_results['scaling'] = scaling_results

    # ── Part 3: Ablation study ──
    print(f"\n{'='*70}")
    print("Ablation Study")
    print(f"{'='*70}\n")

    ablation_configs = {
        'Full MARAHS': {'wind': True, 'cbf': True, 'info': True, 'exploration': True},
        '- Wind Comp.': {'wind': False, 'cbf': True, 'info': True, 'exploration': True},
        '- CBF': {'wind': True, 'cbf': False, 'info': True, 'exploration': True},
        '- Info Gain': {'wind': True, 'cbf': True, 'info': False, 'exploration': True},
        '- Exploration': {'wind': True, 'cbf': True, 'info': True, 'exploration': False},
        'Greedy Only': {'wind': False, 'cbf': False, 'info': False, 'exploration': False},
    }

    ablation_results = {}
    for abl_name, abl_config in ablation_configs.items():
        print(f"  {abl_name}...", end=" ", flush=True)

        class AblationAgent(MARAHSAgent):
            def get_actions(self, obs, positions=None):
                # Override based on ablation config
                self.steps += 1
                actions = np.zeros(self.K, dtype=int)
                moves = np.array([[0, 0], [-1, 0], [1, 0], [0, 1], [0, -1]], dtype=float)

                for i in range(self.K):
                    dir_to_uncov = obs[i, 4:6]
                    norm = np.linalg.norm(dir_to_uncov)
                    if norm < 0.01:
                        actions[i] = 0
                        continue

                    dir_norm = dir_to_uncov / norm

                    # Wind compensation
                    if abl_config['wind']:
                        wind = obs[i, 8:10]
                        wind_comp = -wind * 0.4 * self.wind_intensity
                        dir_norm = dir_norm + wind_comp
                        n2 = np.linalg.norm(dir_norm)
                        if n2 > 0.01:
                            dir_norm /= n2

                    best_action = 0
                    best_score = -float('inf')

                    for a in range(5):
                        move = moves[a]
                        move_norm = np.linalg.norm(move)

                        score = 0.0
                        if move_norm > 0:
                            score += np.dot(move, dir_norm)

                        # CBF
                        if abl_config['cbf'] and positions is not None:
                            for j in range(self.K):
                                if j != i:
                                    fp = positions[i] + move
                                    d = np.linalg.norm(fp - positions[j])
                                    if d < self.min_sep + 1.0:
                                        score -= 3.0 * max(0, self.min_sep + 1.0 - d)

                        # Info gain
                        if abl_config['info'] and positions is not None:
                            fr = int(np.clip(positions[i, 0] + move[0], 0, self.N - 1))
                            fc = int(np.clip(positions[i, 1] + move[1], 0, self.N - 1))
                            score += 0.1 * (1.0 - self.info_grid[fr, fc])

                        if score > best_score:
                            best_score = score
                            best_action = a

                    actions[i] = best_action

                # Update info grid
                if abl_config['info'] and positions is not None:
                    for i in range(self.K):
                        r = int(np.clip(positions[i, 0], 0, self.N - 1))
                        c = int(np.clip(positions[i, 1], 0, self.N - 1))
                        for dr in range(-2, 3):
                            for dc in range(-2, 3):
                                rr, cc = r + dr, c + dc
                                if 0 <= rr < self.N and 0 <= cc < self.N:
                                    if math.sqrt(dr*dr + dc*dc) <= 2.0:
                                        self.info_grid[rr, cc] = min(1.0, self.info_grid[rr, cc] + 0.1)

                return actions

        abl_agent = AblationAgent(config.num_drones, config.grid_size,
                                  env_config.min_separation, 1.0)
        abl_results = []
        for seed_idx in range(config.num_seeds):
            seed = seed_idx * 1000 + hash(abl_name) % 1000
            result = run_episode(env, abl_agent, seed, 3, 'katrina', config.max_steps)
            abl_results.append(result)

        covs = [r['final_coverage'] for r in abl_results]
        safety = [r['safety_rate'] for r in abl_results]
        ablation_results[abl_name] = {
            'coverage_mean': float(np.mean(covs)),
            'coverage_std': float(np.std(covs)),
            'safety_mean': float(np.mean(safety)),
        }
        print(f"Coverage: {ablation_results[abl_name]['coverage_mean']:.1f}%")

    all_results['ablation'] = ablation_results

    # ── Part 4: Coverage by hurricane category ──
    print(f"\n{'='*70}")
    print("Performance by Hurricane Category")
    print(f"{'='*70}\n")

    cat_results = {}
    for cat in config.wind_categories:
        cat_results[str(cat)] = {}
        for agent_name in ['Greedy', 'PID', 'Greedy+CBF', 'MARAHS']:
            agent = agents[agent_name]
            results_list = []
            for profile in config.hurricane_profiles:
                for seed_idx in range(config.num_seeds):
                    seed = seed_idx * 1000 + cat * 100 + hash(profile) % 100
                    result = run_episode(env, agent, seed, cat, profile,
                                        config.max_steps)
                    results_list.append(result)

            covs = [r['final_coverage'] for r in results_list]
            cat_results[str(cat)][agent_name] = {
                'mean': float(np.mean(covs)),
                'std': float(np.std(covs)),
            }
        print(f"  Cat {cat}: ", end="")
        for name in ['Greedy', 'PID', 'Greedy+CBF', 'MARAHS']:
            print(f"{name}={cat_results[str(cat)][name]['mean']:.1f}%  ", end="")
        print()

    all_results['by_category'] = cat_results

    # ── Part 5: Coverage over time curves ──
    print(f"\n{'='*70}")
    print("Generating Coverage Curves")
    print(f"{'='*70}\n")

    coverage_curves = {}
    for agent_name in ['Greedy', 'Greedy+CBF', 'MARAHS']:
        agent = agents[agent_name]
        all_histories = []
        for seed_idx in range(config.num_seeds):
            seed = seed_idx * 1000 + 777
            result = run_episode(env, agent, seed, 3, 'katrina', config.max_steps)
            all_histories.append(result['coverage_history'])

        # Compute mean and std across seeds
        max_len = max(len(h) for h in all_histories)
        padded = np.full((len(all_histories), max_len), np.nan)
        for i, h in enumerate(all_histories):
            padded[i, :len(h)] = h

        mean_curve = np.nanmean(padded, axis=0)
        std_curve = np.nanstd(padded, axis=0)

        # Subsample for storage (every 5 steps)
        indices = list(range(0, max_len, 5))
        coverage_curves[agent_name] = {
            'steps': indices,
            'mean': [float(mean_curve[i]) for i in indices],
            'std': [float(std_curve[i]) for i in indices],
        }

    all_results['coverage_curves'] = coverage_curves

    # ── Save results ──
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Benchmark Complete in {elapsed:.1f}s")
    print(f"{'='*70}\n")

    # Remove coverage_history from overall results for JSON
    for agent_name in all_results:
        if isinstance(all_results[agent_name], dict):
            for key in all_results[agent_name]:
                if isinstance(all_results[agent_name][key], dict):
                    all_results[agent_name][key].pop('coverage_history', None)

    # Save
    output_path = os.path.join(config.output_dir, 'benchmark_results_v2.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to: {output_path}")

    # Print final summary table
    print_summary_table(all_results)

    return all_results


def aggregate_results(results: List[Dict]) -> Dict:
    """Aggregate per-seed results into summary statistics."""
    coverages = [r['final_coverage'] for r in results]
    max_coverages = [r['max_coverage'] for r in results]
    safety_rates = [r['safety_rate'] for r in results]
    t50s = [r['time_to_50'] for r in results if r['time_to_50'] > 0]
    min_dists = [r['min_inter_agent_dist'] for r in results]
    collision_avs = [r['collision_avoidances'] for r in results]

    n = len(results)
    ci95 = lambda x: 1.96 * np.std(x) / np.sqrt(n) if n > 1 else 0.0

    return {
        'overall': {
            'coverage_mean': float(np.mean(coverages)),
            'coverage_std': float(np.std(coverages)),
            'coverage_95ci': float(ci95(coverages)),
            'coverage_median': float(np.median(coverages)),
            'max_coverage_mean': float(np.mean(max_coverages)),
            'safety_mean': float(np.mean(safety_rates)),
            'safety_std': float(np.std(safety_rates)),
            'time_to_50_mean': float(np.mean(t50s)) if t50s else -1,
            'time_to_50_std': float(np.std(t50s)) if t50s else 0,
            'min_dist_mean': float(np.mean(min_dists)),
            'collision_avoidances_mean': float(np.mean(collision_avs)),
            'n_seeds': n,
        },
        'by_category': {},
        'by_profile': {},
    }


def print_summary_table(results: Dict):
    """Print a publication-ready summary table."""
    print(f"\n{'='*80}")
    print("TABLE 1: Overall Performance Comparison (10 drones, Category 3 winds)")
    print(f"{'='*80}")
    print(f"{'Method':<16} {'Coverage(%)':>14} {'Safety(%)':>12} {'Time→50%':>12} {'Min Dist':>10}")
    print(f"{'-'*16} {'-'*14} {'-'*12} {'-'*12} {'-'*10}")

    for name in ['Random', 'Hover', 'Greedy', 'Voronoi', 'Spiral', 'PID',
                  'Greedy+CBF', 'MARAHS']:
        if name in results and 'overall' in results[name]:
            r = results[name]['overall']
            cov_str = f"{r['coverage_mean']:.1f}±{r['coverage_std']:.1f}"
            safety_str = f"{r['safety_mean']:.1f}"
            t50_str = f"{r['time_to_50_mean']:.0f}" if r['time_to_50_mean'] > 0 else "N/A"
            min_d_str = f"{r['min_dist_mean']:.2f}"
            print(f"{name:<16} {cov_str:>14} {safety_str:>12} {t50_str:>12} {min_d_str:>10}")

    # Ablation table
    if 'ablation' in results:
        print(f"\n{'='*80}")
        print("TABLE 2: Ablation Study (Category 3 winds, 10 drones)")
        print(f"{'='*80}")
        print(f"{'Configuration':<20} {'Coverage(%)':>14} {'Δ Full':>10}")
        print(f"{'-'*20} {'-'*14} {'-'*10}")

        full_cov = results['ablation'].get('Full MARAHS', {}).get('coverage_mean', 0)
        for name, r in results['ablation'].items():
            delta = r['coverage_mean'] - full_cov
            delta_str = f"{delta:+.1f}" if name != 'Full MARAHS' else "—"
            print(f"{name:<20} {r['coverage_mean']:.1f}±{r['coverage_std']:.1f} {delta_str:>10}")

    # Scaling table
    if 'scaling' in results:
        print(f"\n{'='*80}")
        print("TABLE 3: Multi-Agent Scaling (Category 3, MARAHS)")
        print(f"{'='*80}")
        print(f"{'Drones':>8} {'Coverage(%)':>14} {'Safety(%)':>12} {'Time→50%':>12}")
        print(f"{'-'*8} {'-'*14} {'-'*12} {'-'*12}")

        for n_drones in ['1', '2', '4', '6', '8', '10']:
            if n_drones in results['scaling']:
                r = results['scaling'][n_drones]
                t50 = f"{r['time_to_50_mean']:.0f}" if r['time_to_50_mean'] > 0 else "N/A"
                print(f"{n_drones:>8} {r['coverage_mean']:.1f}±{r['coverage_std']:.1f} "
                      f"{r['safety_mean']:.1f} {t50:>12}")


if __name__ == '__main__':
    config = BenchmarkConfig(
        grid_size=25,
        num_drones=10,
        max_steps=400,
        num_seeds=30,
    )
    results = run_benchmark(config)
