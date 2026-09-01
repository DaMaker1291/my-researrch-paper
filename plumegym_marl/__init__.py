"""
PlumeGym-MARL: Physics-Informed Multi-Agent Reinforcement Learning
for Wildfire Perimeter Tracking in Convective Plume Winds

The first open-source benchmark for extreme-weather aerial robotics.

Usage:
    from plumegym_marl import WildfireEnv, GPFireFront, NeuralCBFSafetyFilter, GATCommunication
    
    env = WildfireEnv(grid=30, n_drones=10, max_steps=300, wind_speed=12.0)
    obs = env.reset()
    
    gp = GPFireFront(grid_size=30)
    cbf = NeuralCBFSafetyFilter()
    gat = GATCommunication(in_dim=env.obs_dim)
"""

__version__ = "0.1.0"
__author__ = "Shaurjesh Basu"

import os as _os
import sys as _sys

# Add parent directory to path so we can import the original modules
_parent = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _parent not in _sys.path:
    _sys.path.insert(0, _parent)

from paper_ready_train import WildfireEnv
from gp_firefront import GPFireFront, InformationTheoreticPlanner
from neural_cbf import NeuralCBFSafetyFilter
from gat_communication import GATCommunication

__all__ = [
    "WildfireEnv",
    "GPFireFront",
    "InformationTheoreticPlanner",
    "NeuralCBFSafetyFilter",
    "GATCommunication",
]
