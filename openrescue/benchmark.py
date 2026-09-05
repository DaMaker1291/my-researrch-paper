"""
OpenRescue — Benchmark Runner (Phase 1 / Phase 2 protocol)
==========================================================

Sweeps policies across Failure Levels 1–5 over multiple seeds, aggregates with
IQM + 95% bootstrap CIs, and writes results.

    python -m openrescue.benchmark --seeds 10 --steps 200 --figure figures/benchmark.png

Defaults are tuned to run in a few seconds; use `--seeds 20 --steps 300` for
paper-grade numbers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from .environment import OpenRescueEnv
from .failures import level_config
from .metrics import METRICS, aggregate_runs, bootstrap_ci, results_table, summarize
from .policies import make_policy


def run_episode(policy_name: str, level: int, seed: int, steps: int,
                env_kwargs: dict) -> dict:
    """Run a single (policy, level, seed) episode; return its summary dict.

    Self-contained so it can be dispatched to worker processes by the Kaggle
    runner or the Phase-2 pool.
    """
    env = OpenRescueEnv(failure_level=level, max_steps=steps, seed=seed, **env_kwargs)
    policy = make_policy(policy_name)
    obs, info = env.reset(seed=seed)
    if hasattr(policy, 'reset'):
        policy.reset()
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        actions = policy.act(obs, info, rng)
        obs, _, terminated, truncated, info = env.step(actions)
        if bool(np.all(terminated)) or bool(np.all(truncated)):
            break
    return env.episode_summary()


def run_policy_at_level(policy_name: str, level: int, seeds: List[int],
                        steps: int, env_kwargs: dict) -> List[dict]:
    """Run one policy at one failure level; return per-episode summaries."""
    return [run_episode(policy_name, level, seed, steps, env_kwargs)
            for seed in seeds]


def run_benchmark(policies: List[str], levels: List[int], seeds: List[int],
                  steps: int, env_kwargs: dict, verbose: bool = True) -> Dict:
    # Failure levels are defined on {1..5}; clamp so labels always match the
    # environment's actual severity (the env clips internally).
    levels = [int(np.clip(l, 1, 5)) for l in levels]
    results: Dict[tuple, dict] = {}
    meta = {
        'framework': 'OpenRescue',
        'protocol': 'IQM + 95% bootstrap CI (Agarwal et al. 2021)',
        'policies': policies,
        'levels': levels,
        'seeds': seeds,
        'steps': steps,
        'env_kwargs': env_kwargs,
        'metrics': METRICS,
    }
    total = len(policies) * len(levels) * len(seeds)
    done = 0
    t0 = time.time()
    for lvl in levels:
        for pol in policies:
            runs = run_policy_at_level(pol, lvl, seeds, steps, env_kwargs)
            results[(pol, lvl)] = aggregate_runs(runs)
            done += len(seeds)
            if verbose:
                print(f"  [{done:>3}/{total}] {pol:<10} L{lvl}  "
                      f"coverage={runs[-1]['coverage']:.2f} "
                      f"pois={runs[-1]['pois_ratio']:.2f} "
                      f"mean_r={runs[-1]['mean_r']:.2f} "
                      f"eta_i={runs[-1]['eta_i']:.2f} bits/J", flush=True)
    meta['runtime_s'] = round(time.time() - t0, 2)
    return {'meta': meta, 'results': results}


# ---------------------------------------------------------------------------
# Time-varying failure regime (gust profile) — graceful recovery protocol
# ---------------------------------------------------------------------------

def gust_profile(t: int, steps: int, peak: int) -> int:
    """Failure level over time: calm -> ramp -> storm hold -> ramp -> calm.

    An episode spends the first 30% calm (L1), ramps linearly to `peak`
    between 30% and 42% of its length, holds the peak until 58%, ramps back
    down to L1 by 70%, and finishes calm — the recovery window in which
    re-expansion (P4) is measured.
    """
    x = t / steps
    if x < 0.30:
        return 1
    if x < 0.42:
        f = (x - 0.30) / 0.12
        return int(round(1 + (peak - 1) * f))
    if x < 0.58:
        return peak
    if x < 0.70:
        f = (x - 0.58) / 0.12
        return int(round(peak - (peak - 1) * f))
    return 1


def calm_profile(t: int, steps: int) -> int:
    """No-storm control: constant L1 over the whole episode."""
    return 1


def run_episode_timevarying(policy_name: str, level_fn, seed: int, steps: int,
                            env_kwargs: dict) -> dict:
    """Run one (policy, profile) episode with a time-varying failure level.

    `level_fn(t)` returns the failure level for step `t`. The level is applied
    by swapping ``env.failure`` mid-episode — the environment re-reads it every
    step, so the environment itself is untouched. The summary carries a
    per-step record (level, mean R, coverage, POIs found, survivors) used for
    the recovery figure and the P4 analysis.
    """
    env = OpenRescueEnv(failure_level=1, max_steps=steps, seed=seed, **env_kwargs)
    policy = make_policy(policy_name)
    obs, info = env.reset(seed=seed)
    if hasattr(policy, 'reset'):
        policy.reset()
    rng = np.random.default_rng(seed)
    tr = []
    for t in range(steps):
        lvl = int(level_fn(t, steps))
        if lvl != env.failure.level:
            env.failure = level_config(lvl, env._rng)
        actions = policy.act(obs, info, rng)
        obs, _, terminated, truncated, info = env.step(actions)
        tr.append({
            'step': t, 'level': lvl,
            'mean_r': float(info['mean_r']),
            'coverage': float(info['coverage']),
            'pois': int(info['pois_found']),
            'alive': int(sum(1 for a in info['agents'] if not a['grounded'])),
        })
        if bool(np.all(terminated)) or bool(np.all(truncated)):
            break
    summary = env.episode_summary()
    summary['trace'] = tr
    return summary


def run_timevarying_benchmark(policies: List[str], profiles: Dict[str, callable],
                              seeds: List[int], steps: int, env_kwargs: dict,
                              verbose: bool = True) -> Dict:
    """Sweep policies x time-varying level profiles over seeds."""
    runs: Dict[tuple, List[dict]] = {}
    meta = {
        'framework': 'OpenRescue',
        'protocol': 'time-varying failure (gust ramp + recovery window), '
                    'IQM + 95% bootstrap CI',
        'policies': policies,
        'profiles': list(profiles.keys()),
        'seeds': seeds,
        'steps': steps,
        'env_kwargs': env_kwargs,
    }
    t0 = time.time()
    for pname, level_fn in profiles.items():
        for pol in policies:
            runs[(pol, pname)] = [
                run_episode_timevarying(pol, level_fn, seed, steps, env_kwargs)
                for seed in seeds
            ]
            if verbose:
                r = runs[(pol, pname)][-1]
                print(f"  {pol:<10} {pname:<8} cov={r['coverage']:.2f} "
                      f"pois={r['pois_ratio']:.2f} lost={r['lost']} "
                      f"mean_r={r['mean_r']:.2f}", flush=True)
    meta['runtime_s'] = round(time.time() - t0, 2)
    return {'meta': meta, 'runs': runs}


def recovery_metrics(profile_runs: List[dict], calm_runs: List[dict],
                     steps: int) -> dict:
    """Per-seed recovery statistics for one (policy, gust profile) pair.

    Derived per-seed quantities, aggregated with IQM + 95% bootstrap CI:
      r_recovery      mean-R(post window) / mean-R(pre window)  (state recovery)
      r_pre / r_post  absolute mean-R in the pre / post windows
      r_min_storm     minimum mean-R during the storm hold
      cov_gained_post coverage gained after the storm passes
      cov_velocity_post linear slope of coverage over the recovery window
                        (post-storm search velocity, dC/dt)
      cov_final       end-of-episode coverage
      cov_deficit     cov_final minus the same seed's no-storm (calm) coverage
      pois_*          POI analogs of the coverage rows
      lost            agents lost (crash / battery)
    ``calm_runs`` must be in the same seed order as ``profile_runs``.
    """
    pre_w = max(10, int(0.15 * steps))        # calm baseline window
    post_w = max(10, int(0.10 * steps))       # recovery window
    storm_end = int(0.70 * steps)             # first calm step after ramp-down
    hold_lo, hold_hi = int(0.42 * steps), int(0.58 * steps)

    per_seed = {k: [] for k in (
        'r_recovery', 'r_pre', 'r_post', 'r_min_storm',
        'cov_final', 'cov_gained_post', 'cov_velocity_post',
        'pois_final', 'pois_gained_post',
        'cov_deficit', 'pois_deficit', 'lost')}
    calm_by_seed = {i: c for i, c in enumerate(calm_runs)}
    for i, run in enumerate(profile_runs):
        tr = run.get('trace')
        if not tr:
            raise ValueError('recovery_metrics requires traced episodes')
        pre = tr[:pre_w]
        post = tr[-post_w:]
        # episodes can terminate early (all POIs found), so clamp all windows
        # to the actual trace length
        hold = tr[hold_lo:min(hold_hi, len(tr))]
        r_pre = float(np.mean([s['mean_r'] for s in pre]))
        r_post = float(np.mean([s['mean_r'] for s in post]))
        per_seed['r_pre'].append(r_pre)
        per_seed['r_post'].append(r_post)
        per_seed['r_recovery'].append(r_post / max(r_pre, 1e-9))
        storm_r = [s['mean_r'] for s in hold] or [s['mean_r'] for s in tr]
        per_seed['r_min_storm'].append(float(np.min(storm_r)))
        storm_idx = min(storm_end - 1, len(tr) - 1)
        cov_storm = tr[storm_idx]['coverage'] if storm_idx >= 0 else 0.0
        pois_storm = tr[storm_idx]['pois'] if storm_idx >= 0 else 0
        cov_final, pois_final = run['coverage'], run['pois_found']
        per_seed['cov_final'].append(cov_final)
        per_seed['cov_gained_post'].append(cov_final - cov_storm)
        rec = tr[storm_end:] if storm_end < len(tr) else tr[-1:]
        if len(rec) >= 2:
            per_seed['cov_velocity_post'].append(float(
                np.polyfit(np.arange(len(rec)),
                           [s['coverage'] for s in rec], 1)[0]))
        else:
            per_seed['cov_velocity_post'].append(0.0)
        per_seed['pois_final'].append(pois_final)
        per_seed['pois_gained_post'].append(pois_final - pois_storm)
        per_seed['lost'].append(run['lost'])
        calm = calm_by_seed.get(i)
        if calm is not None:
            per_seed['cov_deficit'].append(cov_final - calm['coverage'])
            per_seed['pois_deficit'].append(pois_final - calm['pois_found'])
        else:
            per_seed['cov_deficit'].append(float('nan'))
            per_seed['pois_deficit'].append(float('nan'))
    out = {}
    for k, vals in per_seed.items():
        iqm, lo, hi = bootstrap_ci(vals)
        out[k] = {'iqm': iqm, 'ci_low': lo, 'ci_high': hi}
    return out


def summarize_timevarying(runs: Dict[tuple, List[dict]], calm_name: str,
                          steps: int) -> dict:
    """JSON-safe per (policy, profile) episode aggregates + recovery metrics."""
    out = {'recovery': {}, 'episodes': {}}
    for (pol, pname), pruns in runs.items():
        key = f"{pol}__{pname}"
        clean = [{k2: v2 for k2, v2 in r.items() if k2 != 'trace'} for r in pruns]
        out['episodes'][key] = {
            m: {k2: float(v2) for k2, v2 in aggregate_runs(clean)[m].items()}
            for m in METRICS
        }
    for pol in sorted({k[0] for k in runs}):
        calm_runs = runs.get((pol, calm_name), [])
        for (ppol, pname), pruns in runs.items():
            if ppol != pol or pname == calm_name:
                continue
            out['recovery'][f"{pol}__{pname}"] = recovery_metrics(
                pruns, calm_runs, steps)
    return out


def make_recovery_figure(runs: Dict[tuple, List[dict]], profiles: List[str],
                         steps: int, out_path: str):
    """Mean-R and coverage timelines over the storm, per policy."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'random': '#c0392b', 'frontier': '#e67e22', 'resilient': '#27ae60',
              'ig': '#2980b9'}
    gust = [p for p in profiles if p.startswith('gust')]
    if not gust:
        return
    fig, axes = plt.subplots(len(gust), 2, figsize=(11, 3.6 * len(gust)),
                             sharex=True)
    if len(gust) == 1:
        axes = axes[None, :]
    storm_lo, storm_hi = int(0.30 * steps), int(0.70 * steps)
    for r_i, pname in enumerate(gust):
        for c_i, metric in enumerate(['mean_r', 'coverage']):
            ax = axes[r_i, c_i]
            ax.axvspan(storm_lo, storm_hi, color='#bdc3c7', alpha=0.35,
                       label='storm window' if r_i == 0 and c_i == 0 else None)
            for pol in colors:
                pruns = runs.get((pol, pname), [])
                if not pruns:
                    continue
                n = max(len(r['trace']) for r in pruns)
                agg = np.zeros(n)
                cnt = np.zeros(n)
                for r in pruns:
                    vals = np.array([s[metric] for s in r['trace']])
                    agg[:len(vals)] += vals
                    cnt[:len(vals)] += 1
                # episodes may end early (all POIs found): average only over
                # the runs that actually reached each step
                with np.errstate(invalid='ignore', divide='ignore'):
                    agg = np.where(cnt > 0, agg / np.maximum(cnt, 1), np.nan)
                ax.plot(range(n), agg, label=pol, color=colors[pol])
            ax.set_xlabel('step')
            ax.set_ylabel('$\\bar{R}$' if metric == 'mean_r' else 'Coverage')
            ax.set_title(pname if c_i == 0 else '')
            ax.grid(alpha=0.3)
            if r_i == 0 and c_i == 0:
                ax.legend()
    fig.suptitle('OpenRescue: Degradation and Recovery Under a Passing Gust')
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved -> {out_path}")


def make_figure(results: Dict, out_path: str):
    """Coverage and mean-R vs failure level, per policy (IQM + CI)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    policies = sorted({k[0] for k in results})
    levels = sorted({k[1] for k in results})
    colors = {'random': '#c0392b', 'frontier': '#e67e22', 'resilient': '#27ae60',
              'ig': '#2980b9'}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, ylabel in [
        (axes[0], 'coverage', 'Coverage (IQM)'),
        (axes[1], 'mean_r', 'Mean Resilience Index R̄ (IQM)'),
    ]:
        for pol in policies:
            ys = [results[(pol, l)].get(metric, {}).get('iqm', np.nan) for l in levels]
            lo = [results[(pol, l)].get(metric, {}).get('ci_low', np.nan) for l in levels]
            hi = [results[(pol, l)].get(metric, {}).get('ci_high', np.nan) for l in levels]
            ax.plot(levels, ys, marker='o', label=pol,
                    color=colors.get(pol, None))
            ax.fill_between(levels, lo, hi, alpha=0.15, color=colors.get(pol, None))
        ax.set_xlabel('Failure Level')
        ax.set_ylabel(ylabel)
        ax.set_xticks(levels)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle('OpenRescue: Graceful Autonomy Under Infrastructure Failure')
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved -> {out_path}")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description='OpenRescue benchmark sweep')
    p.add_argument('--policies', nargs='+', default=['random', 'frontier', 'resilient'])
    p.add_argument('--levels', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                   choices=range(1, 6), metavar='{1..5}',
                   help='failure levels to sweep (valid: 1-5)')
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--seed-offset', type=int, default=0)
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--grid', type=int, default=24)
    p.add_argument('--drones', type=int, default=6)
    p.add_argument('--pois', type=int, default=10)
    p.add_argument('--out', type=str, default='openrescue_benchmark.json')
    p.add_argument('--figure', type=str, default='')
    p.add_argument('--timevarying', nargs='*', type=int, metavar='PEAK',
                   help='run the time-varying gust protocol at these peak '
                        'levels (e.g. 4 5) instead of the static level sweep; '
                        'a no-storm calm control is always included')
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args(argv)

    env_kwargs = dict(grid=args.grid, n_drones=args.drones, n_pois=args.pois)
    seeds = list(range(args.seed_offset, args.seed_offset + args.seeds))

    if args.timevarying is not None:
        peaks = args.timevarying if args.timevarying else [4, 5]
        bad = [pk for pk in peaks if pk not in range(1, 6)]
        if bad:
            p.error(f"--timevarying peaks must be in 1..5, got {bad}")
        profiles: Dict[str, callable] = {'calm': calm_profile}
        for pk in peaks:
            profiles[f'gust-L{pk}'] = (
                lambda t, steps, pk=pk: gust_profile(t, steps, pk))
        print(f"OpenRescue time-varying benchmark | policies={args.policies} "
              f"profiles={list(profiles)} seeds={seeds} steps={args.steps}")
        res = run_timevarying_benchmark(args.policies, profiles, seeds,
                                        args.steps, env_kwargs,
                                        verbose=not args.quiet)
        payload = {'meta': res['meta'],
                   'results': summarize_timevarying(res['runs'], 'calm',
                                                    args.steps)}
        with open(args.out, 'w') as f:
            json.dump(payload, f, indent=2, allow_nan=False)
        print(f"\nResults saved -> {args.out}")
        if args.figure:
            make_recovery_figure(res['runs'], list(profiles.keys()),
                                 args.steps, args.figure)
        return 0

    print(f"OpenRescue benchmark | policies={args.policies} levels={args.levels} "
          f"seeds={seeds} steps={args.steps}")
    results = run_benchmark(args.policies, args.levels, seeds, args.steps,
                            env_kwargs, verbose=not args.quiet)

    print("\n" + results_table(results['results']))

    out = summarize(results['results'])
    payload = {'meta': results['meta'], 'results': out}
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved -> {args.out}")

    if args.figure:
        make_figure(results['results'], args.figure)
    return 0


if __name__ == '__main__':
    sys.exit(main())