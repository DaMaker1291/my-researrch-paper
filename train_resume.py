#!/usr/bin/env python3
"""
Resumable PPO training - saves checkpoint every 500 episodes.
Run multiple times to complete all 3000 episodes.

Usage:
    python3 train_resume.py          # Train from scratch or resume
    python3 train_resume.py --eval   # Skip training, run benchmark
"""
import numpy as np
import torch
import time
import json
import os
import sys

# Import from the existing ppo_train.py
from ppo_train import PPONetwork, PPOAgent, WindCurriculum, compute_reward
from paper_ready_train import WildfireEnv

CHECKPOINT = "ppo_checkpoint.json"
BEST_MODEL = "ppo_best.pt"
FINAL_MODEL = "ppo_final.pt"
RESULTS_FILE = "ppo_train_results.json"
EPISODES_PER_RUN = 500  # Each run completes 500 episodes
TOTAL_EPISODES = 3000


def load_checkpoint():
    """Load training state from disk."""
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, 'r') as f:
            return json.load(f)
    return {"episode": 0, "best_reward": -float('inf'), "episode_rewards": [],
            "episode_coverages": [], "episode_safety": []}


def save_checkpoint(state):
    """Save training state to disk."""
    with open(CHECKPOINT, 'w') as f:
        json.dump(state, f)


def train_batch(n_episodes=EPISODES_PER_RUN):
    """Train for n_episodes, resuming from checkpoint."""
    state = load_checkpoint()
    start_ep = state["episode"]

    if start_ep >= TOTAL_EPISODES:
        print(f"All {TOTAL_EPISODES} episodes already completed!")
        print(f"Final: reward={state['episode_rewards'][-100:]}" if state['episode_rewards'] else "")
        return state

    end_ep = min(start_ep + n_episodes, TOTAL_EPISODES)
    print(f"=" * 60)
    print(f"PPO Training: episodes {start_ep+1} → {end_ep} / {TOTAL_EPISODES}")
    print(f"=" * 60)

    env = WildfireEnv(grid=30, n_drones=10, max_steps=300, wind_speed=0)
    agent = PPOAgent(
        obs_dim=env.obs_dim, act_dim=env.act_dim,
        lr=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_epsilon=0.2, entropy_coef=0.02,
        value_coef=0.5, n_epochs=4, batch_size=128,
    )

    # Load best model if exists
    if os.path.exists(BEST_MODEL):
        try:
            agent.net.load_state_dict(torch.load(BEST_MODEL, weights_only=True))
            print(f"Loaded best model from {BEST_MODEL}")
        except Exception:
            pass

    curriculum = WindCurriculum()
    episode_rewards = state.get("episode_rewards", [])
    episode_coverages = state.get("episode_coverages", [])
    episode_safety = state.get("episode_safety", [])
    best_reward = state.get("best_reward", -float('inf'))

    t_start = time.time()

    for ep in range(start_ep, end_ep):
        wind_speed = curriculum.get_wind(ep)
        env.base_wind = wind_speed
        obs = env.reset()
        agent.reset_trajectories()

        ep_reward = 0.0
        ep_crashes = 0

        for step in range(300):
            alive_obs = []
            alive_ids = []
            for i in range(10):
                if env.drones[i]['alive']:
                    alive_obs.append(obs[i])
                    alive_ids.append(i)
            if len(alive_obs) == 0:
                break

            obs_tensor = torch.tensor(np.array(alive_obs), dtype=torch.float32)
            actions, log_probs, values = agent.net.get_action(obs_tensor)

            for j, drone_id in enumerate(alive_ids):
                agent.store_transition(
                    drone_id, alive_obs[j], actions[j].item(),
                    0.0, False, log_probs[j].item(), values[j].item(),
                )

            action_array = np.zeros(10, dtype=np.int32)
            for j, drone_id in enumerate(alive_ids):
                action_array[drone_id] = actions[j].item()

            next_obs, rewards, dones, infos = env.step(action_array)

            for j, drone_id in enumerate(alive_ids):
                d = env.drones[drone_id]
                prev_visited = set(d.get('visited', set()))
                fire_dist = infos[drone_id].get('fire_dist', 10.0)
                thermal = infos[drone_id].get('thermal', 0.0)
                wind_spd = infos[drone_id].get('wind_speed', 0.0)

                # Inline reward computation (exploration-dominant)
                if dones[drone_id] and not d['alive']:
                    reward = -15.0
                else:
                    reward = 0.05  # tiny survival
                    new_cells = len([c for c in d.get('visited', set()) if c not in prev_visited])
                    reward += 30.0 * new_cells
                    if new_cells == 0:
                        reward -= 0.5
                    if fire_dist < 3.0:
                        reward += 8.0 * (1.0 - fire_dist / 3.0)
                    if step >= 299:
                        reward += 5.0
                t = agent._trajectories[drone_id]
                t['reward'][-1] = reward
                t['done'][-1] = dones[drone_id]
                ep_reward += reward
                if dones[drone_id] and not d['alive']:
                    ep_crashes += 1

            obs = next_obs
            if all(not env.drones[i]['alive'] for i in range(10)):
                break

        for drone_id in range(10):
            if env.drones[drone_id]['alive'] and drone_id in agent._trajectories:
                t = agent._trajectories[drone_id]
                if len(t['done']) > 0 and not t['done'][-1]:
                    t['done'][-1] = True

        loss = agent.update()

        coverage = len(env.total_cells_explored) / (30 * 30) * 100
        safety = (1.0 - ep_crashes / 10) * 100

        episode_rewards.append(ep_reward)
        episode_coverages.append(coverage)
        episode_safety.append(safety)

        if (ep + 1) % 50 == 0:
            r100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards[-min(100,len(episode_rewards)):])
            c100 = np.mean(episode_coverages[-100:]) if len(episode_coverages) >= 100 else np.mean(episode_coverages[-min(100,len(episode_coverages)):])
            s100 = np.mean(episode_safety[-100:]) if len(episode_safety) >= 100 else np.mean(episode_safety[-min(100,len(episode_safety)):])
            elapsed = time.time() - t_start
            stage = curriculum.get_stage(ep)
            print(f"Ep {ep+1:5d}/{TOTAL_EPISODES} | R: {r100:7.1f} | "
                  f"Cov: {c100:5.1f}% | Safe: {s100:4.0f}% | "
                  f"Wind={wind_speed} Stg{stage} | Loss: {loss:.4f} | "
                  f"{elapsed:.0f}s")

            if r100 > best_reward:
                best_reward = r100
                torch.save(agent.net.state_dict(), BEST_MODEL)

        # Save checkpoint every 100 episodes
        if (ep + 1) % 100 == 0:
            save_checkpoint({
                "episode": ep + 1,
                "best_reward": best_reward,
                "episode_rewards": episode_rewards,
                "episode_coverages": episode_coverages,
                "episode_safety": episode_safety,
            })

    # Final save
    torch.save(agent.net.state_dict(), FINAL_MODEL)
    save_checkpoint({
        "episode": end_ep,
        "best_reward": best_reward,
        "episode_rewards": episode_rewards,
        "episode_coverages": episode_coverages,
        "episode_safety": episode_safety,
    })

    elapsed = time.time() - t_start
    print(f"\nBatch complete: {start_ep+1}→{end_ep} in {elapsed:.0f}s")

    # Print overall stats
    if len(episode_rewards) > 0:
        last_n = min(200, len(episode_rewards))
        print(f"Overall (last {last_n}): R={np.mean(episode_rewards[-last_n:]):.1f}, "
              f"Cov={np.mean(episode_coverages[-last_n:]):.1f}%, "
              f"Safe={np.mean(episode_safety[-last_n:]):.0f}%")

    return {
        "episode_rewards": episode_rewards,
        "episode_coverages": episode_coverages,
        "episode_safety": episode_safety,
    }


def run_benchmark():
    """Run benchmark with all baselines using trained model."""
    print("=" * 60)
    print("BENCHMARK: All baselines on 30×30 grid, wind=12, 20 eval eps")
    print("=" * 60)

    env = WildfireEnv(grid=30, n_drones=10, max_steps=300, wind_speed=12.0)

    # Load trained PPO
    ppo_agent = PPOAgent(obs_dim=env.obs_dim, act_dim=env.act_dim)
    if os.path.exists(BEST_MODEL):
        ppo_agent.net.load_state_dict(torch.load(BEST_MODEL, weights_only=True))
        print(f"Loaded {BEST_MODEL}")
    else:
        print(f"No {BEST_MODEL} found, using random PPO")

    results = {}
    n_eval = 20

    # 1. PPO (trained)
    print("\nEvaluating PPO (trained)...")
    ppo_safety, ppo_coverage, ppo_perimeter, ppo_alive = [], [], [], []
    for _ in range(n_eval):
        obs = env.reset()
        alive_set = set(range(10))
        total_explored = set()
        for step in range(300):
            alive_obs = [obs[i] for i in range(10) if env.drones[i]['alive']]
            alive_ids = [i for i in range(10) if env.drones[i]['alive']]
            if not alive_obs:
                break
            obs_tensor = torch.tensor(np.array(alive_obs), dtype=torch.float32)
            actions, _, _ = ppo_agent.net.get_action(obs_tensor, deterministic=True)
            action_array = np.zeros(10, dtype=np.int32)
            for j, did in enumerate(alive_ids):
                action_array[did] = actions[j].item()
            obs, _, dones, _ = env.step(action_array)
            for did in alive_ids:
                if dones[did] and not env.drones[did]['alive']:
                    alive_set.discard(did)
        coverage = len(env.total_cells_explored) / (30*30) * 100
        # Count perimeter cells (fire-adjacent explored cells)
        perimeter = 0
        if env._fire_dist_cache is not None:
            for (cx, cy) in env.total_cells_explored:
                if 0 <= cy < 30 and 0 <= cx < 30 and env._fire_dist_cache[cy, cx] < 2.0:
                    perimeter += 1
        ppo_safety.append(len(alive_set) / 10 * 100)
        ppo_coverage.append(coverage)
        ppo_perimeter.append(perimeter / max(1, len(env.total_cells_explored)) * 100)
        ppo_alive.append(len(alive_set))
    results['PPO (trained)'] = {
        'safety': np.mean(ppo_safety), 'coverage': np.mean(ppo_coverage),
        'perimeter': np.mean(ppo_perimeter), 'alive': np.mean(ppo_alive),
    }

    # 2. Greedy baseline
    print("Evaluating Greedy...")
    greedy_safety, greedy_coverage, greedy_perimeter, greedy_alive = [], [], [], []
    for _ in range(n_eval):
        obs = env.reset()
        alive_set = set(range(10))
        for step in range(300):
            alive_ids = [i for i in range(10) if env.drones[i]['alive']]
            if not alive_ids:
                break
            action_array = np.zeros(10, dtype=np.int32)
            for did in alive_ids:
                pos = env.drones[did]['pos']
                gx, gy = int(pos[0]), int(pos[1])
                # Move toward nearest unexplored cell
                best_a, best_d = 0, float('inf')
                for a in range(5):
                    dx, dy = [0, 0, 0, -1, 1][a], [0, -1, 1, 0, 0][a]
                    nx, ny = gx + dx, gy + dy
                    # Prefer unexplored
                    if (nx, ny) not in env.total_cells_explored and 0 <= nx < 30 and 0 <= ny < 30:
                        best_a, best_d = a, 0
                        break
                    if best_d > 0:
                        dist = abs(nx - 15) + abs(ny - 15)  # Move toward center
                        if dist < best_d:
                            best_d, best_a = dist, a
                action_array[did] = best_a
            obs, _, dones, _ = env.step(action_array)
            for did in alive_ids:
                if dones[did] and not env.drones[did]['alive']:
                    alive_set.discard(did)
        coverage = len(env.total_cells_explored) / (30*30) * 100
        perimeter = 0
        if env._fire_dist_cache is not None:
            for (cx, cy) in env.total_cells_explored:
                if 0 <= cy < 30 and 0 <= cx < 30 and env._fire_dist_cache[cy, cx] < 2.0:
                    perimeter += 1
        greedy_safety.append(len(alive_set) / 10 * 100)
        greedy_coverage.append(coverage)
        greedy_perimeter.append(perimeter / max(1, len(env.total_cells_explored)) * 100)
        greedy_alive.append(len(alive_set))
    results['Greedy'] = {
        'safety': np.mean(greedy_safety), 'coverage': np.mean(greedy_coverage),
        'perimeter': np.mean(greedy_perimeter), 'alive': np.mean(greedy_alive),
    }

    # 3. PID baseline
    print("Evaluating PID...")
    pid_safety, pid_coverage, pid_perimeter, pid_alive = [], [], [], []
    for _ in range(n_eval):
        obs = env.reset()
        alive_set = set(range(10))
        for step in range(300):
            alive_ids = [i for i in range(10) if env.drones[i]['alive']]
            if not alive_ids:
                break
            action_array = np.zeros(10, dtype=np.int32)
            for did in alive_ids:
                pos = env.drones[did]['pos']
                gx, gy = int(pos[0]), int(pos[1])
                # Simple PID: move toward perimeter
                if env._fire_dist_cache is not None:
                    fd = env._fire_dist_cache[min(gy,29), min(gx,29)]
                    if fd < 1.5:
                        action_array[did] = 0  # hover near fire
                    elif fd < 3.0:
                        # Move closer to fire
                        best_a, best_d = 0, fd
                        for a in range(1, 5):
                            dx, dy = [0, 0, -1, 1][a-1], [-1, 1, 0, 0][a-1]
                            nx, ny = min(max(gx+dx, 0), 29), min(max(gy+dy, 0), 29)
                            d = env._fire_dist_cache[ny, nx]
                            if abs(d - 2.0) < abs(best_d - 2.0):
                                best_d, best_a = d, a
                        action_array[did] = best_a
                    else:
                        # Move toward fire
                        best_a, best_d = 0, fd
                        for a in range(1, 5):
                            dx, dy = [0, 0, -1, 1][a-1], [-1, 1, 0, 0][a-1]
                            nx, ny = min(max(gx+dx, 0), 29), min(max(gy+dy, 0), 29)
                            d = env._fire_dist_cache[ny, nx]
                            if d < best_d:
                                best_d, best_a = d, a
                        action_array[did] = best_a
            obs, _, dones, _ = env.step(action_array)
            for did in alive_ids:
                if dones[did] and not env.drones[did]['alive']:
                    alive_set.discard(did)
        coverage = len(env.total_cells_explored) / (30*30) * 100
        perimeter = 0
        if env._fire_dist_cache is not None:
            for (cx, cy) in env.total_cells_explored:
                if 0 <= cy < 30 and 0 <= cx < 30 and env._fire_dist_cache[cy, cx] < 2.0:
                    perimeter += 1
        pid_safety.append(len(alive_set) / 10 * 100)
        pid_coverage.append(coverage)
        pid_perimeter.append(perimeter / max(1, len(env.total_cells_explored)) * 100)
        pid_alive.append(len(alive_set))
    results['PID'] = {
        'safety': np.mean(pid_safety), 'coverage': np.mean(pid_coverage),
        'perimeter': np.mean(pid_perimeter), 'alive': np.mean(pid_alive),
    }

    # 4. Random baseline
    print("Evaluating Random...")
    rand_safety, rand_coverage, rand_perimeter, rand_alive = [], [], [], []
    for _ in range(n_eval):
        obs = env.reset()
        alive_set = set(range(10))
        for step in range(300):
            alive_ids = [i for i in range(10) if env.drones[i]['alive']]
            if not alive_ids:
                break
            action_array = np.random.randint(0, 5, size=10, dtype=np.int32)
            # Only assign to alive drones
            for did in range(10):
                if did not in alive_ids:
                    action_array[did] = 0
            obs, _, dones, _ = env.step(action_array)
            for did in alive_ids:
                if dones[did] and not env.drones[did]['alive']:
                    alive_set.discard(did)
        coverage = len(env.total_cells_explored) / (30*30) * 100
        perimeter = 0
        if env._fire_dist_cache is not None:
            for (cx, cy) in env.total_cells_explored:
                if 0 <= cy < 30 and 0 <= cx < 30 and env._fire_dist_cache[cy, cx] < 2.0:
                    perimeter += 1
        rand_safety.append(len(alive_set) / 10 * 100)
        rand_coverage.append(coverage)
        rand_perimeter.append(perimeter / max(1, len(env.total_cells_explored)) * 100)
        rand_alive.append(len(alive_set))
    results['Random'] = {
        'safety': np.mean(rand_safety), 'coverage': np.mean(rand_coverage),
        'perimeter': np.mean(rand_perimeter), 'alive': np.mean(rand_alive),
    }

    # 5. Lawnmower baseline
    print("Evaluating Lawnmower...")
    lawnmower_safety, lawnmower_coverage, lawnmower_perimeter, lawnmower_alive = [], [], [], []
    for _ in range(n_eval):
        obs = env.reset()
        alive_set = set(range(10))
        # Assign each drone to a row band
        row_bands = [int(i * 30 / 10) for i in range(10)]
        directions = [1 if i % 2 == 0 else -1 for i in range(10)]
        for step in range(300):
            alive_ids = [i for i in range(10) if env.drones[i]['alive']]
            if not alive_ids:
                break
            action_array = np.zeros(10, dtype=np.int32)
            for did in alive_ids:
                pos = env.drones[did]['pos']
                gx, gy = int(pos[0]), int(pos[1])
                target_y = row_bands[did]
                if abs(gy - target_y) > 0:
                    action_array[did] = 3 if gy < target_y else 4  # North/South
                else:
                    if directions[did] > 0:
                        action_array[did] = 2 if gx < 28 else 3  # East, then turn
                    else:
                        action_array[did] = 1 if gx > 1 else 3  # West, then turn
            obs, _, dones, _ = env.step(action_array)
            for did in alive_ids:
                if dones[did] and not env.drones[did]['alive']:
                    alive_set.discard(did)
        coverage = len(env.total_cells_explored) / (30*30) * 100
        perimeter = 0
        if env._fire_dist_cache is not None:
            for (cx, cy) in env.total_cells_explored:
                if 0 <= cy < 30 and 0 <= cx < 30 and env._fire_dist_cache[cy, cx] < 2.0:
                    perimeter += 1
        lawnmower_safety.append(len(alive_set) / 10 * 100)
        lawnmower_coverage.append(coverage)
        lawnmower_perimeter.append(perimeter / max(1, len(env.total_cells_explored)) * 100)
        lawnmower_alive.append(len(alive_set))
    results['Lawnmower'] = {
        'safety': np.mean(lawnmower_safety), 'coverage': np.mean(lawnmower_coverage),
        'perimeter': np.mean(lawnmower_perimeter), 'alive': np.mean(lawnmower_alive),
    }

    # Print results table
    print("\n" + "=" * 80)
    print(f"{'Method':<20} {'Safety%':>8} {'Coverage%':>10} {'Perimeter%':>11} {'Alive':>6}")
    print("-" * 80)
    for name in ['Random', 'Lawnmower', 'PID', 'Greedy', 'PPO (trained)']:
        r = results[name]
        print(f"{name:<20} {r['safety']:>7.1f}% {r['coverage']:>9.1f}% {r['perimeter']:>10.1f}% {r['alive']:>5.1f}/10")
    print("=" * 80)

    # Save
    with open('benchmark_final.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to benchmark_final.json")

    return results


if __name__ == "__main__":
    if "--eval" in sys.argv:
        run_benchmark()
    else:
        train_batch(EPISODES_PER_RUN)
        state = load_checkpoint()
        if state["episode"] >= TOTAL_EPISODES:
            print("\n" + "=" * 60)
            print("ALL 3000 EPISODES COMPLETE!")
            print("=" * 60)
            last_n = min(200, len(state['episode_rewards']))
            print(f"Final reward (last {last_n}): {np.mean(state['episode_rewards'][-last_n:]):.1f}")
            print(f"Final coverage (last {last_n}): {np.mean(state['episode_coverages'][-last_n:]):.1f}%")
            print(f"Final safety (last {last_n}): {np.mean(state['episode_safety'][-last_n:]):.0f}%")
            print("\nNow run benchmark...")
            run_benchmark()
        else:
            remaining = TOTAL_EPISODES - state["episode"]
            print(f"\nDone. {state['episode']}/{TOTAL_EPISODES} episodes.")
            print(f"Run again to continue ({remaining} remaining).")
