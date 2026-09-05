"""
OpenRescue — Resilient Autonomous Robot Swarms
==============================================

Benchmark for **Graceful Autonomy Under Infrastructure Failure**: a
multi-agent rescue-exploration benchmark in which GPS, communication and
sensors degrade from Failure Level 1 (nominal) to Level 5 (catastrophic),
and agents must keep operating via an onboard **Resilience Index R_{i,t}**
that gates between Explore-Aggressive, Coordinated-Cluster and
Relay-Mesh/Return behaviors.

    from openrescue import OpenRescueEnv, ResilientPolicy
    from openrescue.metrics import aggregate_runs
    from openrescue.failures import level_config

    env = OpenRescueEnv(grid=24, n_drones=6, failure_level=3, seed=0)
    obs, info = env.reset()
    policy = ResilientPolicy()
    actions = policy.act(obs, info, np.random.default_rng(0))

Reference: OPENRESCUE_SPEC.md (mathematical framework, failure-level table,
metric protocol, 12-month roadmap).
"""
__version__ = "0.1.0"
__author__ = "OpenRescue Project"

from .environment import OpenRescueEnv
from .failures import FailureConfig, level_config
from .metrics import aggregate_runs, bootstrap_ci, interquartile_mean, results_table
from .policies import (
    RandomPolicy, FrontierPolicy, ResilientPolicy, IgPolicy, POLICIES, make_policy,
)
from .resilience_index import (
    resilience_index, sensor_confidence, comm_quality, consensus_agreement,
    behavior_mode, DEFAULT_WEIGHTS,
    MODE_EXPLORE, MODE_CLUSTER, MODE_RELAY,
)

__all__ = [
    "OpenRescueEnv",
    "FailureConfig", "level_config",
    "aggregate_runs", "bootstrap_ci", "interquartile_mean", "results_table",
    "RandomPolicy", "FrontierPolicy", "ResilientPolicy", "IgPolicy",
    "POLICIES", "make_policy",
    "resilience_index", "sensor_confidence", "comm_quality", "consensus_agreement",
    "behavior_mode", "DEFAULT_WEIGHTS",
    "MODE_EXPLORE", "MODE_CLUSTER", "MODE_RELAY",
]