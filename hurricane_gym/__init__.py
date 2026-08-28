"""
HurricaneGym: Extreme Atmospheric UAV Reinforcement Learning Environment
========================================================================

The first open-source benchmark for training drones to survive
hurricane-force winds using reinforcement learning.

Installation:
    pip install hurricane-gym

Quick Start:
    from hurricane_gym import CrazyflieEnv, QuadrotorEnv
    
    env = CrazyflieEnv(wind_speed=70.0)
    obs, info = env.reset()
    
    for step in range(3000):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)

Environments:
    - QuadrotorEnv: Generic quadrotor with direct motor control
    - CrazyflieEnv: Crazyflie 2.1 micro drone dynamics
    - SwarmGridWorld: Multi-drone cooperative coverage

Wind Models:
    - RankineVortex: Realistic hurricane wind field
    - DrydenTurbulence: Stochastic turbulence model

Safety:
    - ControlBarrierFunction: Provable safety guarantees
    - SafetyLayer: Wraps any controller with CBF safety

Adaptation:
    - NeuralFlyController: Online meta-adaptive control
    - RLSScheduler: Recursive Least Squares for real-time adaptation

References:
    [1] Ames et al., "Control Barrier Functions: Theory and Applications"
    [2] Li et al., "Neural Fly: Adaptive Neural Network for Drone Control"
    [3] Moncada et al., "Meta-Learning for Sim-to-Real Transfer"
"""

__version__ = "1.0.0"
__author__ = "MARAHS Research Team"

from hurricane_gym.envs import QuadrotorEnv, CrazyflieEnv, SwarmGridWorld
from hurricane_gym.wind import RankineVortex, DrydenTurbulence
from hurricane_gym.safety import ControlBarrierFunction, SafetyLayer
from hurricane_gym.meta import NeuralFlyController, RLSScheduler

__all__ = [
    'QuadrotorEnv',
    'CrazyflieEnv', 
    'SwarmGridWorld',
    'RankineVortex',
    'DrydenTurbulence',
    'ControlBarrierFunction',
    'SafetyLayer',
    'NeuralFlyController',
    'RLSScheduler',
]
