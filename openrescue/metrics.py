"""
OpenRescue — Metrics
====================

Statistical protocol for the benchmark (Phase 2 of the roadmap): every cell of
the results table reports the **Interquartile Mean (IQM)** over seeds with a
**95% bootstrap confidence interval** (percentile method), following the
RL reliability protocol of Agarwal et al. (2021, "Deep RL at the Edge of the
Statistical Precipice").

IQM discards the bottom and top 25% of runs, which makes it robust to the
occasional catastrophic episode while still using more data than the median.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

METRICS = [
    'coverage', 'pois_ratio', 'survival', 'lost', 'mean_r', 'info_gain_bits',
    'energy_joules', 'eta_i', 'eta_i_surv',
]


def interquartile_mean(values: Sequence[float]) -> float:
    """Trimmed mean over the interquartile range."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float('nan')
    if v.size < 4:
        return float(np.mean(v))
    q1, q3 = np.quantile(v, [0.25, 0.75])
    iqr = v[(v >= q1) & (v <= q3)]
    if iqr.size == 0:
        return float(np.mean(v))
    return float(np.mean(iqr))


def bootstrap_ci(values: Sequence[float], n_resamples: int = 1000,
                 alpha: float = 0.05, seed: int = 42) -> tuple:
    """
    Percentile bootstrap confidence interval of the IQM.

    Returns (iqm, ci_low, ci_high) with coverage 1 - alpha.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return (float('nan'), float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    n = v.size
    stats = np.empty(n_resamples)
    for b in range(n_resamples):
        sample = v[rng.integers(0, n, size=n)]
        stats[b] = interquartile_mean(sample)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return (float(np.mean(stats)), float(lo), float(hi))


def aggregate_runs(runs: Sequence[dict]) -> dict:
    """
    Aggregate per-episode summaries into IQM + 95% CI per metric.

    runs: list of `env.episode_summary()` dicts (one per seed/episode).
    """
    out = {}
    for m in METRICS:
        vals = [r[m] for r in runs if m in r]
        iqm, lo, hi = bootstrap_ci(vals)
        out[m] = {'iqm': iqm, 'ci_low': lo, 'ci_high': hi}
    return out


def format_metric(agg: dict, decimals: int = 3) -> str:
    """'0.612 [0.601, 0.623]' formatting for table cells."""
    v, lo, hi = agg['iqm'], agg['ci_low'], agg['ci_high']
    if np.isnan(v):
        return '—'
    return f"{v:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


def results_table(results: Dict[str, Dict[str, dict]]) -> str:
    """
    Pretty console table: rows = (policy, failure level), cols = metrics.
    results: {(policy, level): agg_dict}
    """
    keys = sorted(results.keys(), key=lambda k: (k[1], k[0]))
    header = f"{'policy':<12}{'level':<6}" + "".join(f"{m:<26}" for m in METRICS)
    lines = [header, '-' * len(header)]
    for pol, lvl in keys:
        agg = results[(pol, lvl)]
        cells = [pol[:11], f"L{lvl}"]
        for m in METRICS:
            cells.append(format_metric(agg[m])[:25].ljust(26))
        lines.append("".join(c[:12].ljust(12) if i < 2 else c for i, c in enumerate(cells)))
    return "\n".join(lines)


def summarize(results: Dict[str, Dict[str, dict]]) -> dict:
    """
    Condensed JSON-safe view: per (policy, level) -> {metric: {iqm, ci_low, ci_high}}.
    """
    return {
        f"{pol}__L{lvl}": {m: {k: float(v) for k, v in agg[m].items()}
                           for m in METRICS}
        for (pol, lvl), agg in results.items()
    }