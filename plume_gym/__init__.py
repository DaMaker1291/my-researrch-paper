"""
PlumeGym-MARL: A Physics-Informed Multi-Agent RL Benchmark for
Autonomous Wildfire Perimeter Tracking in Convective Plume Winds.

Combines fire-spread dynamics, atmospheric plume physics, and
GNN-based multi-agent reinforcement learning with Neural-CBF safety.
"""

__version__ = "0.1.0"

from .wildfire_env import WildfirePlumeEnv, WildfireConfig
from .agents import GATMARAHS, PPOAgent, SACAgent, GreedyTracker, PIDTracker, RandomTracker
from .neural_cbf import NeuralCBF
from .information_gain import GPInformationGain

__all__ = [
    "WildfirePlumeEnv",
    "WildfireConfig",
    "GATMARAHS",
    "PPOAgent",
    "SACAgent",
    "GreedyTracker",
    "PIDTracker",
    "RandomTracker",
    "NeuralCBF",
    "GPInformationGain",
]
