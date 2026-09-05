"""
OpenRescue — Failure Injection Model
====================================

Failure Levels 1–5 parameterize the severity of infrastructure collapse in the
OpenRescue environment. Each level drives five independent degradation axes:

  * ``sensor_noise_std``  : std of additive noise on the synthetic IMU/optical-flow
                           reading. Directly lowers the sensor-confidence term
                           H_sensor of the Resilience Index.
  * ``gps_denial_prob``   : probability per step that the agent's position belief
                           (GPS fix) is denied / not updated. Drives belief-vs-true
                           divergence and hence map divergence.
  * ``gps_drift_std``     : std of the random-walk drift applied to the position
                           belief when a fix IS available (degraded GPS).
  * ``packet_loss_prob``  : probability that a message from a given neighbor is
                           dropped. Directly lowers Q_comm.
  * ``comm_range_scale``  : multiplicative shrink of the communication radius.
                           Fragments the mesh and isolates agents.
  * ``map_corrupt_prob``  : probability per step that a sensed cell is written to
                           the local map with a wrong label. Diverges local maps
                           and lowers D_consensus.

Failure Level semantics (used in the benchmark sweep):

    L1  Nominal        — occasional sensor jitter, full GPS, no packet loss.
    L2  Mild           — elevated sensor noise, rare GPS dropouts, 5% packet loss.
    L3  Moderate       — GPS denial ~15%, 15% packet loss, comm range -20%,
                         occasional map corruption.
    L4  Severe         — GPS denial ~35%, 30% packet loss, comm range -40%,
                         sustained map corruption.
    L5  Catastrophic   — GPS denial 60%+, 50% packet loss, comm range -60%,
                         heavy sensor noise and map corruption.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

# --------------------------------------------------------------------------
# Level configuration
# --------------------------------------------------------------------------

# Central values for each level; per-run stochasticity is added by
# `level_config` (jitter around the nominal), so every seed sees a slightly
# different realization of the same "level".
_LEVEL_NOMINAL: Dict[int, dict] = {
    1: dict(sensor_noise_std=0.05, gps_denial_prob=0.00, gps_drift_std=0.05,
            packet_loss_prob=0.00, comm_range_scale=1.00, map_corrupt_prob=0.00),
    2: dict(sensor_noise_std=0.15, gps_denial_prob=0.05, gps_drift_std=0.15,
            packet_loss_prob=0.05, comm_range_scale=0.90, map_corrupt_prob=0.02),
    3: dict(sensor_noise_std=0.30, gps_denial_prob=0.15, gps_drift_std=0.30,
            packet_loss_prob=0.15, comm_range_scale=0.80, map_corrupt_prob=0.05),
    4: dict(sensor_noise_std=0.50, gps_denial_prob=0.35, gps_drift_std=0.50,
            packet_loss_prob=0.30, comm_range_scale=0.60, map_corrupt_prob=0.10),
    5: dict(sensor_noise_std=0.80, gps_denial_prob=0.60, gps_drift_std=0.80,
            packet_loss_prob=0.50, comm_range_scale=0.40, map_corrupt_prob=0.20),
}

# Relative jitter applied per realization (multiplicative for probs/scales).
_JITTER = 0.10


@dataclass(frozen=True)
class FailureConfig:
    """A concrete, seeded realization of one failure level for one episode."""

    level: int = 1
    sensor_noise_std: float = 0.05
    gps_denial_prob: float = 0.0
    gps_drift_std: float = 0.05
    packet_loss_prob: float = 0.0
    comm_range_scale: float = 1.0
    map_corrupt_prob: float = 0.0

    @property
    def name(self) -> str:
        return f"L{self.level}"

    @property
    def severity(self) -> float:
        """Continuous severity in [0, 1] for plotting / regression analysis."""
        return (self.level - 1) / 4.0


def level_config(level: int, rng: np.random.Generator) -> FailureConfig:
    """Sample a stochastic realization of the given failure level."""
    level = int(np.clip(level, 1, 5))
    nominal = _LEVEL_NOMINAL[level]

    def jitter_p(p: float) -> float:
        # keep probabilities in [0, 1]
        return float(np.clip(p * (1.0 + _JITTER * rng.standard_normal()), 0.0, 1.0))

    def jitter_s(s: float) -> float:
        # keep stds non-negative
        return float(max(0.0, s * (1.0 + _JITTER * rng.standard_normal())))

    return FailureConfig(
        level=level,
        sensor_noise_std=jitter_s(nominal["sensor_noise_std"]),
        gps_denial_prob=jitter_p(nominal["gps_denial_prob"]),
        gps_drift_std=jitter_s(nominal["gps_drift_std"]),
        packet_loss_prob=jitter_p(nominal["packet_loss_prob"]),
        comm_range_scale=jitter_s(nominal["comm_range_scale"]),
        map_corrupt_prob=jitter_p(nominal["map_corrupt_prob"]),
    )


