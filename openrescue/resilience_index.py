"""
OpenRescue — Resilience Index R_{i,t}
======================================

The local agent's Resilience Index is a composite entropy function evaluated
onboard at time step t:

    R_{i,t} = w_1 * H_sensor(s_i) + w_2 * Q_comm(c_i) + w_3 * D_consensus(m_i)

with all three terms normalized to [0, 1] and w_1 + w_2 + w_3 = 1, so
R_{i,t} in [0, 1].

  * H_sensor  — variance-weighted inverse entropy of the local IMU/optical-flow
                reading. High variance  =>  high entropy  =>  low confidence.
  * Q_comm    — communication quality: packet reception rate, RSSI, and ping
                latency combined into a single score.
  * D_consensus — local agreement: mean exponential KL-divergence similarity
                between agent i's occupancy map and its neighbors' maps.
                1 means perfect agreement, -> 0 means divergent maps.

All functions are pure NumPy so they are trivially portable to the onboard
stack (ESP32/STM32) in Phase 3 of the roadmap.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

# --------------------------------------------------------------------------
# Component 1: H_sensor — variance-weighted inverse entropy
# --------------------------------------------------------------------------

# Reference variance used to normalize: the squared noise std at Failure
# Level 5 (catastrophic). A reading variance >= this maps H_sensor -> 0.
_REF_VARIANCE = (0.80 ** 2)


def sensor_confidence(readings: np.ndarray, ref_variance: float = _REF_VARIANCE) -> float:
    """
    Variance-weighted inverse entropy of a sliding window of IMU/optical-flow
    readings.

    readings: (W, D) array of the last W raw sensor vectors (e.g. accel x/y,
              optical-flow x/y). Returns H_sensor in [0, 1].

    H_sensor = 1 - min(1, log(1 + sigma^2 / sigma_ref^2) / log(1 + 1/sigma_ref_scale))

    A Gaussian reading with variance sigma^2 has differential entropy
    0.5*log(2*pi*e*sigma^2); normalizing by log(1 + sigma^2 / sigma_ref^2)
    gives a smooth, bounded, scale-invariant confidence score.
    """
    if readings.ndim != 2 or readings.shape[0] < 2:
        return 0.5  # not enough samples: assume half confidence
    var = float(np.mean(np.var(readings, axis=0)))
    ratio = var / max(ref_variance, 1e-12)
    entropy_ratio = np.log1p(ratio) / np.log1p(1.0)
    return float(np.clip(1.0 - min(1.0, entropy_ratio), 0.0, 1.0))


# --------------------------------------------------------------------------
# Component 2: Q_comm — communication quality
# --------------------------------------------------------------------------

def comm_quality(packet_reception_rate: float,
                 rssi_values: Sequence[float],
                 latencies_ms: Sequence[float],
                 max_latency_ms: float = 200.0,
                 w: Sequence[float] = (0.4, 0.3, 0.3)) -> float:
    """
    Combine packet reception rate, mean RSSI and mean ping latency.

    packet_reception_rate: fraction of expected neighbor messages actually
        received during the window (0 if the agent has no neighbors).
    rssi_values: per-neighbor received signal strength in [0, 1] (1 = strong).
    latencies_ms: per-neighbor ping latency in ms.
    """
    prr = float(np.clip(packet_reception_rate, 0.0, 1.0))

    if len(rssi_values) == 0:
        rssi_score = 0.0
        lat_score = 0.0
    else:
        rssi_score = float(np.clip(float(np.mean(rssi_values)), 0.0, 1.0))
        lat = float(np.mean(latencies_ms))
        lat_score = float(np.clip(1.0 - lat / max(max_latency_ms, 1e-9), 0.0, 1.0))

    w0, w1, w2 = w
    return float(np.clip(w0 * prr + w1 * rssi_score + w2 * lat_score, 0.0, 1.0))


# --------------------------------------------------------------------------
# Component 3: D_consensus — KL divergence between local occupancy maps
# --------------------------------------------------------------------------

# Soft-occupancy priors per map label: {0: unknown, 1: free, 2: occupied}
_OCCUPANCY_PRIOR = np.array([0.5, 0.05, 0.95], dtype=np.float64)
_EPS = 1e-9


def _soft_occupancy(maps: np.ndarray) -> np.ndarray:
    """maps: (K, N, N) int labels -> (K, N, N) float P(occupied)."""
    return _OCCUPANCY_PRIOR[np.clip(maps, 0, 2).astype(int)]


def map_kl_similarity(map_i: np.ndarray, map_j: np.ndarray) -> float:
    """
    Exponential KL-divergence similarity between two occupancy maps.

    KL is computed cell-wise between the Bernoulli P(occupied) distributions,
    averaged only over cells observed by BOTH agents (overlap set O_ij).
    Agreement on ignorance (both unknown) is intentionally NOT rewarded:
    consensus requires shared *observations*.

    similarity = exp(-mean_KL(O_ij)), with a floor of 0.1 when there is no
    common observation. Returns D in [0.05, 1].
    """
    pi = _soft_occupancy(map_i)
    pj = _soft_occupancy(map_j)

    known_i = map_i > 0
    known_j = map_j > 0
    overlap = known_i & known_j
    n_overlap = int(overlap.sum())

    if n_overlap == 0:
        return 0.3  # no shared observations: weak agreement by default

    a = np.clip(pi[overlap], _EPS, 1 - _EPS)
    b = np.clip(pj[overlap], _EPS, 1 - _EPS)
    # symmetric-ish smoothing guards log(0) and makes the metric robust to
    # single-cell flips
    kl = a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
    mean_kl = float(np.mean(np.clip(kl, 0.0, 10.0)))
    return float(np.clip(np.exp(-mean_kl), 0.05, 1.0))


def consensus_agreement(maps: np.ndarray, neighbors: Sequence[int],
                        self_idx: int) -> float:
    """
    D_consensus for agent ``self_idx``: mean map similarity against neighbors.

    maps: (K, N, N) int label arrays.
    neighbors: indices of agents whose map was successfully received this step.
    """
    if len(neighbors) == 0:
        return 0.3  # isolated: no agreement signal
    sims = [map_kl_similarity(maps[self_idx], maps[j]) for j in neighbors]
    return float(np.mean(sims))


# --------------------------------------------------------------------------
# Composite index
# --------------------------------------------------------------------------

DEFAULT_WEIGHTS = (0.35, 0.35, 0.30)


def resilience_index(h_sensor: float, q_comm: float, d_consensus: float,
                     weights: Sequence[float] = DEFAULT_WEIGHTS) -> float:
    """
    Composite Resilience Index:

        R_{i,t} = w_1 * H_sensor + w_2 * Q_comm + w_3 * D_consensus  in [0, 1]
    """
    w0, w1, w2 = weights
    return float(np.clip(w0 * h_sensor + w1 * q_comm + w2 * d_consensus, 0.0, 1.0))


# Behavior-mode thresholds (from the OpenRescue blueprint)
MODE_EXPLORE = "explore"          # R >= 0.75 : Explore-Aggressive
MODE_CLUSTER = "cluster"          # 0.35 <= R < 0.75 : Coordinated-Cluster
MODE_RELAY = "relay"              # R < 0.35 : Relay-Mesh / Return

EXPLORE_THRESHOLD = 0.75
RELAY_THRESHOLD = 0.35


def behavior_mode(r: float,
                  explore_threshold: float = EXPLORE_THRESHOLD,
                  relay_threshold: float = RELAY_THRESHOLD) -> str:
    """State-driven behavioral policy selector pi(s_i | R_{i,t})."""
    if r >= explore_threshold:
        return MODE_EXPLORE
    if r >= relay_threshold:
        return MODE_CLUSTER
    return MODE_RELAY


def estimate_packet_reception_rate(expected_per_step: List[int],
                                   received_per_step: List[int]) -> float:
    """PRR over a sliding window of comm slots."""
    if len(expected_per_step) == 0 or sum(expected_per_step) == 0:
        return 0.0
    return float(np.clip(sum(received_per_step) / sum(expected_per_step), 0.0, 1.0))


def rssi_from_distance(dist: float, range_m: float, ref: float = 1.0) -> float:
    """
    Simplified RSSI: exp decay with distance, normalized so that agents at the
    edge of comm range still register a weak (but nonzero) signal.
    """
    if range_m <= 0:
        return 0.0
    return float(np.clip(ref * np.exp(-3.0 * dist / max(range_m, 1e-9)), 0.0, 1.0))


def latency_from_distance(dist: float, base_ms: float = 20.0,
                          per_unit_ms: float = 15.0) -> float:
    """Simulated ping latency: base + distance-proportional term."""
    return float(base_ms + per_unit_ms * dist)