"""
HurricaneGym: Extreme Atmospheric UAV Reinforcement Learning Environment
========================================================================

The first open-source benchmark for training drones to survive
hurricane-force winds using reinforcement learning.

Installation:
    pip install hurricane-gym

Features:
    - Realistic quadrotor dynamics (PyBullet)
    - Crazyflie 2.1 micro drone model
    - Rankine vortex hurricane wind field
    - Dryden turbulence model
    - Control Barrier Functions for provable safety
    - Neural Fly meta-adaptive control
    - Domain randomization for sim-to-real transfer
    - Multi-drone cooperative coverage
"""

from setuptools import setup, find_packages

setup(
    name="hurricane-gym",
    version="1.0.0",
    description="Extreme Atmospheric UAV Reinforcement Learning Environment",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="MARAHS Research Team",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.12.0",
        "pybullet>=3.2.0",
        "gymnasium>=0.26.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords="reinforcement-learning drone hurricane robotics safety meta-learning",
    project_urls={
        "Documentation": "https://github.com/marahs/hurricane-gym",
        "Source": "https://github.com/marahs/hurricane-gym",
        "Tracker": "https://github.com/marahs/hurricane-gym/issues",
    },
)
