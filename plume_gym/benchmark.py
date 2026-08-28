"""
Benchmark runner for PlumeGym-MARL wildfire tracking experiments.

Compares:
1. GAT-MARAHS (our method)
2. Greedy fire tracking
3. PID fire tracking
4. Random actions
5. PPO (randomly initialized)
6. SAC (randomly initialized)

Metrics:
- Perimeter tracking rate (%)
- Total cells explored
- Safety rate (%)
- Min distance to fire
- Crashes per episode
- Information gain
- Wind resistance
"""

import numpy as np
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from plume_gym.wildfire_env import WildfirePlumeEnv, WildfireConfig
from plume_gym.agents import GATMARAHS, PPOAgent, SACAgent, GreedyTracker, PIDTracker, RandomTracker
from plume_gym.neural_cbf import NeuralCBF
from plume_gym.information_gain import GPInformationGain


def run_episode(env, agents, use_cbf=False, cbf=None, info_gain=None, num_drones=6, max_steps=600):
    """Run a single episode and return metrics."""
    obs = env.reset()
    episode_reward = np.zeros(num_drones)
    steps_completed = 0
    crashes = 0
    safety_violations = 0
    cells_visited = set()
    perimeter_covered = set()
    info_gains = []

    positions = [d.position.copy() for d in env.drones]

    for step in range(max_steps):
        actions = np.zeros(num_drones, dtype=int)

        for i in range(num_drones):
            if not env.drones[i].is_active:
                continue

            if hasattr(agents[i], 'select_action'):
                if isinstance(agents[i], GreedyTracker):
                    action = agents[i].select_action(env.drones[i].position, env.fire_grid)
                elif isinstance(agents[i], PIDTracker):
                    action = agents[i].select_action(env.drones[i].position, env.fire_grid)
                elif isinstance(agents[i], RandomTracker):
                    action = agents[i].select_action()
                elif isinstance(agents[i], (GATMARAHS,)):
                    action, _, info = agents[i].select_action(obs[i], positions, i)
                    if info_gain is not None:
                        ig = info_gain.compute_expected_information_gain(
                            env.drones[i].position, action, env.cfg.drone_speed
                        )
                        info_gains.append(ig)
                elif isinstance(agents[i], (PPOAgent, SACAgent)):
                    action, _ = agents[i].select_action(obs[i])
                else:
                    action = 0

                # Apply CBF safety filter
                if use_cbf and cbf is not None and env.drones[i].is_active:
                    state = _drone_state_vector(env, i)
                    action_vec = np.zeros(5)
                    action_vec[action] = 1.0
                    safe_action_vec = cbf.safety_filter(state, action_vec)
                    action = int(np.argmax(safe_action_vec))

                actions[i] = action

        # Step environment
        obs, rewards, dones, infos = env.step(actions)
        episode_reward += rewards
        steps_completed += 1

        # Track metrics
        for i in range(num_drones):
            if env.drones[i].is_active:
                pos = env.drones[i].position
                ix, iy = int(pos[0]), int(pos[1])
                cells_visited.add((ix, iy))

                if env._is_on_perimeter(pos):
                    perimeter_covered.add((ix, iy))

                if dones[i]:
                    crashes += 1

        if all(dones):
            break

    # Compute final metrics
    active_drones = sum(1 for d in env.drones if d.is_active)
    total_cells = len(cells_visited)
    total_perimeter = max(1, env.perimeter_cells)
    perimeter_frac = len(perimeter_covered) / total_perimeter * 100 if total_perimeter > 0 else 0

    metrics = {
        'total_reward': float(np.sum(episode_reward)),
        'perimeter_tracking_pct': perimeter_frac,
        'cells_visited': total_cells,
        'coverage_pct': total_cells / env.cfg.grid_area * 100,
        'crashes': crashes,
        'safety_rate': (1 - crashes / num_drones) * 100,
        'active_drones': active_drones,
        'steps': steps_completed,
        'avg_reward_per_step': float(np.mean(episode_reward)) / max(1, steps_completed),
        'mean_wind_speed': float(np.sqrt(env.wind_x**2 + env.wind_y**2).mean()),
        'max_thermal': float(env.thermal_plume.max()),
        'fire_cells': env.total_fire_cells,
    }

    if info_gains:
        metrics['avg_info_gain'] = float(np.mean(info_gains))
        metrics['total_info_gain'] = float(np.sum(info_gains))

    if use_cbf and cbf is not None:
        metrics['cbf_active'] = True

    return metrics


def _drone_state_vector(env, drone_idx):
    """Create state vector for CBF."""
    d = env.drones[drone_idx]
    pos = d.position

    # [px, py, vx, vy, fire_dist, thermal, wind_x, wind_y]
    fire_dist = env._distance_to_nearest_fire(pos)
    ix = int(np.clip(pos[0], 0, env.cfg.grid_size-1))
    iy = int(np.clip(pos[1], 0, env.cfg.grid_size-1))
    thermal = float(env.thermal_plume[ix, iy])
    wx = float(env.wind_x[ix, iy])
    wy = float(env.wind_y[ix, iy])

    return np.array([pos[0], pos[1], d.velocity[0], d.velocity[1],
                     fire_dist, thermal, wx, wy], dtype=np.float32)


def run_benchmark(
    num_episodes: int = 15,
    num_drones: int = 6,
    grid_size: int = 40,
    max_steps: int = 600,
    wind_speeds: list = None,
):
    """Run full benchmark comparing all methods."""
    if wind_speeds is None:
        wind_speeds = [5.0, 10.0, 15.0, 20.0, 25.0]

    cfg = WildfireConfig(
        grid_size=grid_size,
        num_drones=num_drones,
        max_steps=max_steps,
    )

    methods = {
        'Random': lambda: [RandomTracker() for _ in range(num_drones)],
        'Greedy': lambda: [GreedyTracker() for _ in range(num_drones)],
        'PID': lambda: [PIDTracker() for _ in range(num_drones)],
        'PPO': lambda: [PPOAgent(cfg.observation_space_size, 5) for _ in range(num_drones)],
        'SAC': lambda: [SACAgent(cfg.observation_space_size, 5) for _ in range(num_drones)],
        'GAT-MARAHS': lambda: [GATMARAHS(
            obs_size=2*cfg.local_obs_radius+1,
            obs_channels=cfg.obs_channels,
            num_agents=num_drones,
        ) for _ in range(num_drones)],
        'MARAHS+CBF': lambda: [GATMARAHS(
            obs_size=2*cfg.local_obs_radius+1,
            obs_channels=cfg.obs_channels,
            num_agents=num_drones,
        ) for _ in range(num_drones)],
        'MARAHS+CBF+Info': lambda: [GATMARAHS(
            obs_size=2*cfg.local_obs_radius+1,
            obs_channels=cfg.obs_channels,
            num_agents=num_drones,
        ) for _ in range(num_drones)],
    }

    all_results = {}
    cbf = NeuralCBF(state_dim=8, action_dim=5)
    info_gain = GPInformationGain(grid_size=grid_size)

    for method_name, agent_fn in methods.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*60}")

        use_cbf = 'CBF' in method_name
        use_info = 'Info' in method_name

        method_results = {
            'overall': {},
            'by_wind': {},
            'episodes': [],
        }

        episode_metrics = []

        for ep in range(num_episodes):
            # Vary wind speed across episodes
            wind_speed = wind_speeds[ep % len(wind_speeds)]
            cfg.ambient_wind_speed = wind_speed

            env = WildfirePlumeEnv(cfg)
            agents = agent_fn()

            if use_info:
                info_gain.reset()

            metrics = run_episode(
                env, agents,
                use_cbf=use_cbf,
                cbf=cbf if use_cbf else None,
                info_gain=info_gain if use_info else None,
                num_drones=num_drones,
                max_steps=max_steps,
            )

            metrics['wind_speed'] = wind_speed
            metrics['episode'] = ep
            episode_metrics.append(metrics)

            # Track wind-specific results
            ws_key = f"wind_{wind_speed:.0f}"
            if ws_key not in method_results['by_wind']:
                method_results['by_wind'][ws_key] = []
            method_results['by_wind'][ws_key].append(metrics)

            print(f"  Ep {ep:2d} (wind={wind_speed:.0f}m/s): "
                  f"Perimeter={metrics['perimeter_tracking_pct']:.1f}% "
                  f"Safety={metrics['safety_rate']:.0f}% "
                  f"Cells={metrics['cells_visited']} "
                  f"Crashes={metrics['crashes']}")

        # Aggregate overall metrics
        all_rewards = [m['total_reward'] for m in episode_metrics]
        all_perimeter = [m['perimeter_tracking_pct'] for m in episode_metrics]
        all_safety = [m['safety_rate'] for m in episode_metrics]
        all_cells = [m['cells_visited'] for m in episode_metrics]
        all_crashes = [m['crashes'] for m in episode_metrics]
        all_coverage = [m['coverage_pct'] for m in episode_metrics]

        method_results['overall'] = {
            'reward_mean': float(np.mean(all_rewards)),
            'reward_std': float(np.std(all_rewards)),
            'perimeter_mean': float(np.mean(all_perimeter)),
            'perimeter_std': float(np.std(all_perimeter)),
            'safety_mean': float(np.mean(all_safety)),
            'safety_std': float(np.std(all_safety)),
            'cells_mean': float(np.mean(all_cells)),
            'cells_std': float(np.std(all_cells)),
            'crashes_mean': float(np.mean(all_crashes)),
            'crashes_std': float(np.std(all_crashes)),
            'coverage_mean': float(np.mean(all_coverage)),
            'coverage_std': float(np.std(all_coverage)),
            'n_episodes': num_episodes,
        }

        # Aggregate by wind speed
        for ws_key, ws_episodes in method_results['by_wind'].items():
            method_results['by_wind'][ws_key] = {
                'perimeter_mean': float(np.mean([e['perimeter_tracking_pct'] for e in ws_episodes])),
                'safety_mean': float(np.mean([e['safety_rate'] for e in ws_episodes])),
                'coverage_mean': float(np.mean([e['coverage_pct'] for e in ws_episodes])),
                'crashes_mean': float(np.mean([e['crashes'] for e in ws_episodes])),
            }

        method_results['episodes'] = episode_metrics
        all_results[method_name] = method_results

        ov = method_results['overall']
        print(f"\n  SUMMARY: Perimeter={ov['perimeter_mean']:.1f}% "
              f"Safety={ov['safety_mean']:.1f}% "
              f"Coverage={ov['coverage_mean']:.1f}% "
              f"Crashes={ov['crashes_mean']:.1f}")

    return all_results


def print_results_table(results):
    """Print formatted results table."""
    print("\n" + "=" * 90)
    print(f"{'Method':<18} {'Perimeter%':>12} {'Safety%':>10} {'Coverage%':>11} {'Crashes':>9} {'Reward':>10}")
    print("-" * 90)

    for method, data in results.items():
        ov = data['overall']
        print(f"{method:<18} "
              f"{ov['perimeter_mean']:>8.1f}±{ov['perimeter_std']:<3.1f} "
              f"{ov['safety_mean']:>6.1f}±{ov['safety_std']:<3.1f} "
              f"{ov['coverage_mean']:>7.1f}±{ov['coverage_std']:<3.1f} "
              f"{ov['crashes_mean']:>5.1f}±{ov['crashes_std']:<3.1f} "
              f"{ov['reward_mean']:>7.1f}")

    print("=" * 90)


def main():
    """Run the full benchmark."""
    print("=" * 70)
    print("PlumeGym-MARL Benchmark: Wildfire Perimeter Tracking")
    print("=" * 70)

    start = time.time()

    results = run_benchmark(
        num_episodes=15,
        num_drones=6,
        grid_size=40,
        max_steps=600,
    )

    elapsed = time.time() - start

    # Print results
    print_results_table(results)
    print(f"\nTotal time: {elapsed:.1f}s")

    # Save results
    os.makedirs('experiment_results_v3', exist_ok=True)

    # Remove non-serializable episodes for JSON
    results_clean = {}
    for method, data in results.items():
        results_clean[method] = {
            'overall': data['overall'],
            'by_wind': data['by_wind'],
        }

    with open('experiment_results_v3/benchmark_results.json', 'w') as f:
        json.dump(results_clean, f, indent=2)

    print(f"\nResults saved to experiment_results_v3/benchmark_results.json")


if __name__ == '__main__':
    main()
