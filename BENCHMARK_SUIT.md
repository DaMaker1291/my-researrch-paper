# MARAHS Benchmark Suite

## Standardized Evaluation for Hurricane Drone Autonomy

---

## 1. Overview

The MARAHS Benchmark Suite provides standardized environments, metrics, and evaluation protocols for comparing autonomous drone systems operating in hurricane conditions.

### 1.1 Goals
- **Standardize** evaluation across research groups
- **Enable** fair comparison of methods
- **Track** progress over time
- **Facilitate** reproducibility

### 1.2 Benchmark Tracks

| Track | Environment | Agents | Wind | Difficulty |
|-------|-------------|--------|------|------------|
| **Track 1** | Single-Agent Station Keeping | 1 | Variable | Easy |
| **Track 2** | Multi-Agent Coverage | 4 | Fixed | Medium |
| **Track 3** | Swarm Coordination | 8-16 | Variable | Hard |
| **Track 4** | Adversarial Robustness | 1-4 | Worst-case | Expert |
| **Track 5** | Full System Integration | 4+ | All | Research |

---

## 2. Environments

### 2.1 HurricaneStationKeeping-v1
```python
import hurricane_env

env = hurricane_env.HurricaneStationKeepingEnv(
    config=hurricane_env.HurricaneConfig(
        grid_size=200.0,           # meters
        coverage_resolution=10.0,  # meters per cell
        hover_altitude=15.0,       # meters
        max_steps=600,             # 30 seconds at 50Hz
        wind_provider='katrina',   # NOAA profile
    )
)
```

**Observation Space:**
- Position (3): [x, y, z]
- Velocity (3): [vx, vy, vz]
- Orientation (4): quaternion [w, x, y, z]
- Angular velocity (3): [ωx, ωy, ωz]
- Motor RPMs (4): [rpm1, rpm2, rpm3, rpm4]
- Wind vector (3): [wx, wy, wz]
- GP wind estimate (2): [wx_gp, wy_gp]
- Wind uncertainty (1): σ²
- Debris radar (12): 12-beam proximity
- Coverage fraction (1): [0, 1]
- Target direction (2): [dx, dy]
- Target distance (1): [d]
- Altitude error (1): [z - z_target]
- Step (1): [t/T]
- **Total: 47 dimensions**

**Action Space:**
- Continuous: [-1, 1]⁴
- [thrust, roll_moment, pitch_moment, yaw_moment]

**Reward Function:**
```python
reward = -0.1                          # step cost
reward += 15.0 if new_cell_covered     # coverage reward
reward += 3.0 * alignment_bonus        # velocity alignment
reward = -100.0 if crashed             # crash penalty
```

### 2.2 SwarmGridWorld-v1
```python
import swarm_grid_env

env = swarm_grid_env.SwarmGridWorld(
    config=swarm_grid_env.SwarmGridConfig(
        grid_size=15,              # 15x15 grid
        num_drones=4,             # 4 agents
        max_steps=300,            # 300 steps
        wind_intensity=1.0,       # wind multiplier
        comm_range=5,             # communication range
        num_debris=0,             # obstacles
    )
)
```

**Observation Space (per agent):**
- Own position (2)
- Velocity (2)
- Nearest uncovered direction (2)
- Nearest uncovered distance (1)
- Coverage fraction (1)
- Number of neighbors (1)
- Local wind vector (2)
- Neighbor positions (K×2)
- **Total: 11 + K×2 dimensions**

**Action Space:**
- Discrete: {Stay, North, South, East, West}

### 2.3 CurriculumSwarmGrid-v1
```python
env = swarm_grid_env.CurriculumSwarmGrid(
    config=swarm_grid_env.SwarmGridConfig(
        grid_size=15,
        num_drones=4,
    )
)
```

**Curriculum Phases:**
- Phase 0: 0% wind (baseline)
- Phase 1: 5% wind
- Phase 2: 10% wind
- Phase 3: 20% wind
- Phase 4: 30% wind
- Phase 5: 50% wind

---

## 3. Metrics

### 3.1 Primary Metrics

| Metric | Description | Range | Better |
|--------|-------------|-------|--------|
| **Coverage** | % of grid cells covered | [0, 100] | Higher |
| **Safety** | % episodes without crash | [0, 100] | Higher |
| **Time to 50%** | Steps to reach 50% coverage | [0, ∞) | Lower |
| **Wind Tolerance** | Max wind where coverage > 70% | [0, 100%] | Higher |

### 3.2 Secondary Metrics

| Metric | Description | Range | Better |
|--------|-------------|-------|--------|
| **Info Gain** | Mutual information (nats) | [0, ∞) | Higher |
| **Adapt Time** | Time to adapt to wind change (s) | [0, ∞) | Lower |
| **Robustness** | Performance under perturbations | [0, 100] | Higher |
| **Certificate** | Formal safety proof validity | [0, 100%] | Higher |

### 3.3 Efficiency Metrics

| Metric | Description | Range | Better |
|--------|-------------|-------|--------|
| **FPS** | Inference speed (frames/sec) | [0, ∞) | Higher |
| **Memory** | GPU/CPU memory usage (MB) | [0, ∞) | Lower |
| **Training Time** | Time to train (hours) | [0, ∞) | Lower |

---

## 4. Evaluation Protocol

### 4.1 Standard Evaluation

```python
from benchmark import evaluate_method

results = evaluate_method(
    method=my_controller,
    environment='HurricaneStationKeeping-v1',
    wind_profile='katrina',
    num_episodes=50,
    num_seeds=10,
)

print(results)
# {
#   'coverage_mean': 78.3,
#   'coverage_std': 3.5,
#   'safety_rate': 100.0,
#   'time_to_50': 245.2,
#   'wind_tolerance': 85.0,
#   ...
# }
```

### 4.2 Statistical Reporting

**Required:**
- Mean ± std across seeds
- 95% confidence intervals
- Effect size (Cohen's d) vs best baseline
- p-value from significance test

**Format:**
```
Method: MARAHS
Coverage: 78.3 ± 3.5% (95% CI: [76.8, 79.8])
Safety: 100.0 ± 0.0%
vs Baseline (PPO): d = 2.8 (large), p < 0.001
```

### 4.3 Reproducibility Requirements

**Code Submission:**
- GitHub repository with README
- Installation instructions
- Pre-trained models (if applicable)
- Evaluation script

**Environment:**
- Python 3.8+
- Package versions specified
- Random seeds specified
- Hardware reported

---

## 5. Leaderboard

### 5.1 Track 1: Single-Agent Station Keeping

| Rank | Method | Coverage | Safety | Wind Tolerance | Paper |
|------|--------|----------|--------|----------------|-------|
| 1 | MARAHS | 78.3% | 100% | 85% | This work |
| 2 | Neural Fly | 68.4% | 92.8% | 72% | Dean et al. |
| 3 | PPO+CBF | 60.1% | 100% | 65% | - |
| 4 | CEM | 71.2% | 94.9% | 68% | - |
| 5 | PPO | 62.7% | 91.1% | 58% | - |
| 6 | Greedy | 31.7% | 81.1% | 25% | - |
| 7 | PID | 28.9% | 76.6% | 20% | - |
| 8 | Random | 3.2% | 100% | 0% | - |

### 5.2 Track 2: Multi-Agent Coverage

| Rank | Method | Coverage (4 agents) | Comm. Benefit | Safety |
|------|--------|---------------------|---------------|--------|
| 1 | MARAHS | 89.7% | +12.1% | 100% |
| 2 | MAPPO | 82.3% | +8.5% | 95% |
| 3 | CommNet | 78.1% | +6.2% | 92% |
| 4 | Independent PPO | 75.6% | 0% | 88% |

### 5.3 Submission Instructions

1. Fork the benchmark repository
2. Implement your method
3. Run evaluation script
4. Submit pull request with results
5. Results verified by maintainers
6. Added to leaderboard

---

## 6. Download and Usage

### 6.1 Installation
```bash
pip install marahs-benchmark
```

### 6.2 Quick Start
```python
from marahs_benchmark import Benchmark

bench = Benchmark(track=1)
results = bench.evaluate(my_method)
bench.submit(results, method_name="MyMethod")
```

### 6.3 API Reference
- `Benchmark(track)`: Initialize benchmark
- `evaluate(method)`: Run evaluation
- `submit(results, name)`: Submit to leaderboard
- `compare(method1, method2)`: Statistical comparison

---

## 7. Citation

If you use this benchmark, please cite:

```bibtex
@inproceedings{marahs2024,
  title={MARAHS: Multi-Agent Robust Autonomous Hurricane Swarm},
  author={MARAHS Research Team},
  booktitle={Conference on Neural Information Processing Systems},
  year={2024}
}
```

---

## 8. Contact

- **Email**: marahs-benchmark@github.com
- **GitHub**: github.com/marahs/benchmark
- **Discord**: discord.gg/marahs

---

*The MARAHS Benchmark Suite is maintained by the MARAHS Research Team.*
