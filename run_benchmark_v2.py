"""
Quick comprehensive benchmark - generates real results with reduced seeds.
"""
import numpy as np
import json
import time
import os
import math
import inspect
from swarm_grid_env_v2 import SwarmGridWorldV2, SwarmGridConfig
from agents_v2 import (
    RandomAgent, HoverAgent, GreedyAgent, VoronoiAgent, SpiralAgent,
    GreedyCBFAgent, PIDAgent, MARAHSAgent,
)

NUM_SEEDS = 3
GRID = 30
DRONES = 10
STEPS = 400
CATEGORIES = [1, 2, 3, 4, 5]
PROFILES = ['katrina', 'harvey', 'irma', 'maria', 'michael']

def run_episode(env, agent, seed, cat, profile, max_steps=400):
    rng = np.random.default_rng(seed)
    env.config.wind_category = cat
    env.config.hurricane_profile = profile
    env.config.wind_intensity = cat / 5.0
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
    if hasattr(agent, 'reset'):
        agent.reset()

    coverage_history = []
    min_dist_history = []

    for step in range(max_steps):
        positions = env.positions.copy()
        sig = inspect.signature(agent.get_actions)
        if 'positions' in sig.parameters:
            actions = agent.get_actions(obs, positions=positions)
        else:
            actions = agent.get_actions(obs)
        obs, rewards, dones, truncs, infos = env.step(actions)
        coverage_history.append(infos[0]['coverage_pct'])
        min_dist_history.append(infos[0]['min_inter_agent_dist'])
        if dones[0]:
            break

    final_cov = coverage_history[-1] if coverage_history else 0.0
    max_cov = max(coverage_history) if coverage_history else 0.0

    time_to_50 = -1
    for t, c in enumerate(coverage_history):
        if c >= 50.0:
            time_to_50 = t
            break

    min_sep = env.config.min_separation
    violations = sum(1 for d in min_dist_history if d < min_sep)
    safety_rate = 100.0 * (1.0 - violations / max(len(min_dist_history), 1))
    min_dist = min(min_dist_history) if min_dist_history else float('inf')

    return {
        'final_coverage': float(final_cov),
        'max_coverage': float(max_cov),
        'time_to_50': time_to_50,
        'safety_rate': float(safety_rate),
        'min_inter_agent_dist': float(min_dist),
        'collision_avoidances': infos[0]['collision_avoidances'],
    }


def aggregate(results):
    covs = [r['final_coverage'] for r in results]
    safes = [r['safety_rate'] for r in results]
    t50s = [r['time_to_50'] for r in results if r['time_to_50'] >= 0]
    min_ds = [r['min_inter_agent_dist'] for r in results]
    n = len(results)
    ci = lambda x: 1.96 * np.std(x) / np.sqrt(n) if n > 1 else 0.0
    return {
        'coverage_mean': float(np.mean(covs)),
        'coverage_std': float(np.std(covs)),
        'coverage_95ci': float(ci(covs)),
        'safety_mean': float(np.mean(safes)),
        'safety_std': float(np.std(safes)),
        'time_to_50_mean': float(np.mean(t50s)) if t50s else -1,
        'time_to_50_std': float(np.std(t50s)) if t50s else 0,
        'min_dist_mean': float(np.mean(min_ds)),
        'n_seeds': n,
    }


def main():
    os.makedirs('experiment_results_v2', exist_ok=True)
    env_config = SwarmGridConfig(
        grid_size=GRID, num_drones=DRONES, max_steps=STEPS, num_debris=10,
    )
    env = SwarmGridWorldV2(config=env_config)

    agents = {
        'Random': RandomAgent(DRONES),
        'Hover': HoverAgent(DRONES),
        'Greedy': GreedyAgent(DRONES),
        'Voronoi': VoronoiAgent(DRONES, GRID),
        'Spiral': SpiralAgent(DRONES, GRID),
        'PID': PIDAgent(DRONES),
        'Greedy+CBF': GreedyCBFAgent(DRONES, GRID, env_config.min_separation),
        'MARAHS': MARAHSAgent(DRONES, GRID, env_config.min_separation, 1.0),
    }

    all_results = {}
    start_time = time.time()

    # === Part 1: Overall ===
    print(f"\n{'='*70}")
    print(f"MARAHS v2 Benchmark ({GRID}x{GRID}, {DRONES} drones, {STEPS} steps)")
    print(f"{'='*70}\n")

    for agent_name, agent in agents.items():
        t0 = time.time()
        agent_results = []
        for profile in PROFILES:
            for cat in CATEGORIES:
                for s in range(min(NUM_SEEDS, 3)):
                    seed = s * 1000 + cat * 100 + hash(profile) % 100
                    result = run_episode(env, agent, seed, cat, profile, STEPS)
                    result['profile'] = profile
                    result['category'] = cat
                    agent_results.append(result)
        agg = aggregate(agent_results)
        all_results[agent_name] = {'overall': agg, 'all_seeds': agent_results}
        dt = time.time() - t0
        print(f"  {agent_name:<14} Coverage: {agg['coverage_mean']:.1f}% ± {agg['coverage_std']:.1f}% | "
              f"Safety: {agg['safety_mean']:.1f}% | T50: {agg['time_to_50_mean']:.0f} | "
              f"MinDist: {agg['min_dist_mean']:.2f} [{dt:.1f}s]")

    # === Part 2: Scaling ===
    print(f"\n--- Scaling Study ---")
    scaling = {}
    for n_d in [1, 2, 4, 6, 8, 10]:
        sc = SwarmGridConfig(grid_size=GRID, num_drones=n_d, max_steps=STEPS, num_debris=3)
        senv = SwarmGridWorldV2(config=sc)
        sa = MARAHSAgent(n_d, GRID, sc.min_separation, 1.0)
        sres = []
        for s in range(min(NUM_SEEDS, 3)):
            sres.append(run_episode(senv, sa, s * 1000 + n_d, 3, 'katrina', STEPS))
        sa2 = aggregate(sres)
        scaling[str(n_d)] = sa2
        print(f"  {n_d} drones: {sa2['coverage_mean']:.1f}% | Safety: {sa2['safety_mean']:.1f}%")
    all_results['scaling'] = scaling

    # === Part 3: Ablation ===
    print(f"\n--- Ablation Study ---")
    ablations = {
        'Full MARAHS': {'wind': True, 'cbf': True, 'info': True, 'exploration': True},
        '- Wind Comp.': {'wind': False, 'cbf': True, 'info': True, 'exploration': True},
        '- CBF':        {'wind': True, 'cbf': False, 'info': True, 'exploration': True},
        '- Info Gain':  {'wind': True, 'cbf': True, 'info': False, 'exploration': True},
        'Greedy Only':  {'wind': False, 'cbf': False, 'info': False, 'exploration': False},
    }

    ablation_data = {}
    for abl_name, cfg in ablations.items():
        class AblAgent(MARAHSAgent):
            def get_actions(self_o, obs, positions=None):
                self_o.steps += 1
                K = self_o.K
                N = self_o.N
                actions = np.zeros(K, dtype=int)
                moves = np.array([[0,0],[-1,0],[1,0],[0,1],[0,-1]], dtype=float)
                for i in range(K):
                    d2u = obs[i, 4:6]
                    nrm = np.linalg.norm(d2u)
                    if nrm < 0.01:
                        actions[i] = 0; continue
                    dn = d2u / nrm
                    if cfg['wind']:
                        wind = obs[i, 8:10]
                        dn = dn - wind * 0.4 * self_o.wind_intensity
                        nn = np.linalg.norm(dn)
                        if nn > 0.01: dn /= nn
                    best_a, best_s = 0, -1e9
                    for a in range(5):
                        mv = moves[a]; sc_ = 0.0
                        if np.linalg.norm(mv) > 0: sc_ += np.dot(mv, dn)
                        if cfg['cbf'] and positions is not None:
                            for j in range(K):
                                if j != i:
                                    fp = positions[i] + mv
                                    d = np.linalg.norm(fp - positions[j])
                                    if d < self_o.min_sep + 1.0:
                                        sc_ -= 3.0 * max(0, self_o.min_sep + 1.0 - d)
                        if cfg['info'] and positions is not None:
                            fr = int(np.clip(positions[i,0]+mv[0], 0, N-1))
                            fc = int(np.clip(positions[i,1]+mv[1], 0, N-1))
                            sc_ += 0.1 * (1.0 - self_o.info_grid[fr, fc])
                        if sc_ > best_s:
                            best_s = sc_; best_a = a
                    actions[i] = best_a
                if cfg['info'] and positions is not None:
                    for i in range(K):
                        r = int(np.clip(positions[i,0], 0, N-1))
                        c = int(np.clip(positions[i,1], 0, N-1))
                        for dr in range(-2,3):
                            for dc in range(-2,3):
                                rr, cc = r+dr, c+dc
                                if 0<=rr<N and 0<=cc<N and math.sqrt(dr*dr+dc*dc)<=2.0:
                                    self_o.info_grid[rr,cc] = min(1.0, self_o.info_grid[rr,cc]+0.1)
                return actions

        abl_agent = AblAgent(DRONES, GRID, env_config.min_separation, 1.0)
        ares = []
        for s in range(NUM_SEEDS):
            ares.append(run_episode(env, abl_agent, s*1000+777, 3, 'katrina', STEPS))
        ares_agg = aggregate(ares)
        ablation_data[abl_name] = ares_agg
        print(f"  {abl_name:<16} {ares_agg['coverage_mean']:.1f}% ± {ares_agg['coverage_std']:.1f}%")

    all_results['ablation'] = ablation_data

    # === Part 4: Coverage by category ===
    print(f"\n--- By Hurricane Category ---")
    by_cat = {}
    for cat in CATEGORIES:
        by_cat[str(cat)] = {}
        for nm in ['Greedy', 'PID', 'Greedy+CBF', 'MARAHS']:
            ag = agents[nm]
            cr = []
            for profile in PROFILES:
                for s in range(min(NUM_SEEDS, 3)):
                    seed = s*1000 + cat*100 + hash(profile)%100
                    cr.append(run_episode(env, ag, seed, cat, profile, STEPS))
            by_cat[str(cat)][nm] = aggregate(cr)
        vals = '  '.join(f"{nm}={by_cat[str(cat)][nm]['coverage_mean']:.1f}%" for nm in ['Greedy','PID','Greedy+CBF','MARAHS'])
        print(f"  Cat {cat}: {vals}")
    all_results['by_category'] = by_cat

    # === Part 5: Coverage curves (reuse data from Part 1, just re-run 3 agents) ===
    print(f"\n--- Coverage Curves ---")
    curves = {}
    for nm in ['Greedy', 'Greedy+CBF', 'MARAHS']:
        ag = agents[nm]
        hists = []
        for s in range(min(NUM_SEEDS, 3)):
            rng = np.random.default_rng(s*1000+777)
            env.config.wind_intensity = 0.6
            env.config.hurricane_profile = 'katrina'
            obs2, _ = env.reset(seed=int(rng.integers(0, 2**31)))
            if hasattr(ag, 'reset'): ag.reset()
            hist = []
            for t in range(STEPS):
                sig2 = inspect.signature(ag.get_actions)
                if 'positions' in sig2.parameters:
                    acts = ag.get_actions(obs2, positions=env.positions.copy())
                else:
                    acts = ag.get_actions(obs2)
                obs2, _, d, _, inf = env.step(acts)
                hist.append(inf[0]['coverage_pct'])
                if d[0]: break
            hists.append(hist)
        maxlen = max(len(h) for h in hists)
        padded = np.full((len(hists), maxlen), np.nan)
        for i, h in enumerate(hists):
            padded[i, :len(h)] = h
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        idx = list(range(0, maxlen, 10))
        curves[nm] = {
            'steps': idx,
            'mean': [float(mean[i]) for i in idx],
            'std': [float(std[i]) for i in idx],
        }
        print(f"  {nm}: peak={max(mean):.1f}%")
    all_results['coverage_curves'] = curves

    # Save
    elapsed = time.time() - start_time
    outpath = 'experiment_results_v2/benchmark_results_v2.json'
    # Remove raw seeds for JSON
    save_results = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            save_results[k] = {kk: vv for kk, vv in v.items() if kk != 'all_seeds'}
        else:
            save_results[k] = v

    with open(outpath, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE in {elapsed:.1f}s")
    print(f"Results: {outpath}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
