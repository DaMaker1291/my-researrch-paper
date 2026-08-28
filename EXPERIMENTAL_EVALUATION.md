# Experimental Evaluation Framework

## MARAHS: Comprehensive Experimental Protocol

---

## 1. Experimental Setup

### 1.1 Environments

| Environment | Description | Agents | Wind Model | Grid |
|-------------|-------------|--------|------------|------|
| **HurricaneStationKeeping** | Continuous physics, 6-DOF | 1 | NOAA Holland | 200m×200m |
| **SwarmGridWorld** | Discrete grid world | 1-16 | Rankine vortex | 15×15 to 100×100 |
| **CurriculumSwarmGrid** | Wind curriculum training | 4 | Progressive | 15×15 |

### 1.2 Baselines

| Baseline | Description | Reference |
|----------|-------------|-----------|
| **Random** | Random actions | - |
| **PID** | Classical PID controller | Traditional |
| **Greedy** | Move to nearest uncovered cell | Heuristic |
| **PPO** | Standard PPO (shared critic) | Schulman et al., 2017 |
| **PPO+CBF** | PPO with CBF safety layer | Ames et al., 2019 |
| **Neural Fly** | Meta-adaptive without safety | Dean et al., 2022 |
| **CEM** | Cross-Entropy Method | Currently used |
| **MARAHS (Ours)** | Full system with all 7 components | This work |

### 1.3 Metrics

**Primary Metrics:**
- **Coverage %**: Percentage of grid cells covered
- **Safety Violations**: Number of crashes/collisions
- **Time to 50% Coverage**: Efficiency metric
- **Wind Tolerance**: Maximum wind intensity where coverage > 70%

**Secondary Metrics:**
- **Information Gain**: Mutual information about wind field (nats)
- **Adaptation Speed**: Time to converge after wind change (seconds)
- **Robustness Score**: Performance under adversarial perturbations
- **Formal Safety Certificate**: Whether mathematical proof exists

---

## 2. Experiment 1: Single-Agent Coverage Performance

### 2.1 Protocol
- Grid: 200m×200m, 10m resolution (400 cells)
- Wind: Progressive from 0% to 100% intensity
- Steps: 600 per episode
- Trials: 50 per method
- GPU: NVIDIA T4 or equivalent

### 2.2 Expected Results

| Method | No Wind | Cat 1 | Cat 3 | Cat 5 | Safety |
|--------|---------|-------|-------|-------|--------|
| Random | 12.3 ± 2.1% | 8.1 ± 1.8% | 3.2 ± 1.2% | 1.1 ± 0.8% | N/A |
| PID | 45.6 ± 4.3% | 38.2 ± 5.1% | 28.9 ± 6.2% | 15.3 ± 4.8% | 23.4% |
| Greedy | 52.1 ± 3.8% | 42.5 ± 4.9% | 31.7 ± 5.8% | 18.2 ± 5.1% | 18.9% |
| PPO | 89.2 ± 3.2% | 78.5 ± 4.1% | 62.7 ± 5.3% | 41.2 ± 6.8% | 8.9% |
| PPO+CBF | 87.8 ± 3.5% | 76.2 ± 4.4% | 60.1 ± 5.6% | 39.8 ± 7.1% | 0.0% |
| Neural Fly | 91.5 ± 2.8% | 82.3 ± 3.9% | 68.4 ± 4.8% | 48.7 ± 6.2% | 7.2% |
| CEM | 94.7 ± 2.1% | 85.1 ± 3.2% | 71.2 ± 4.1% | 52.3 ± 5.8% | 5.1% |
| **MARAHS** | **96.2 ± 1.8%** | **88.7 ± 2.9%** | **78.3 ± 3.5%** | **58.9 ± 5.1%** | **0.0%** |

### 2.3 Statistical Tests
- **ANOVA** across all methods (p < 0.001 expected)
- **Tukey HSD** post-hoc pairwise comparisons
- **Effect size** (Cohen's d) for MARAHS vs best baseline

---

## 3. Experiment 2: Ablation Study

### 3.1 Protocol
Remove each novel component one at a time:

| Configuration | Components | Expected Coverage (Cat 3) |
|---------------|------------|---------------------------|
| **Full MARAHS** | All 7 | 78.3% |
| **-Wind Mapper** | 6 (no GP) | 66.0% (-12.3%) |
| **-Adaptive CBF** | 6 (no safety verification) | 75.1% (-3.2%) |
| **-Inverse Dynamics** | 6 (no wind estimation) | 68.5% (-9.8%) |
| **-Adversarial** | 6 (no worst-case testing) | 74.2% (-4.1%) |
| **-Information** | 6 (random exploration) | 72.8% (-5.5%) |
| **-Multi-Scale** | 6 (single timescale) | 76.1% (-2.2%) |
| **-Formal** | 6 (no formal verification) | 77.9% (-0.4%) |

### 3.2 Expected Findings
1. **Wind Mapper** has largest impact (+12.3%)
2. **Inverse Dynamics** is critical for sensorless operation (+9.8%)
3. **Information Planning** improves efficiency (+5.5%)
4. **Adversarial** improves robustness (+4.1%)
5. **Adaptive CBF** maintains safety (+3.2%)
6. **Multi-Scale** improves adaptation speed (+2.2%)
7. **Formal** provides guarantees (+0.4% performance, but +100% trust)

---

## 4. Experiment 3: Multi-Agent Coordination

### 4.1 Protocol
- Grid: 15×15 (225 cells)
- Agents: 1, 2, 4, 8, 16
- Wind: 50% intensity (Cat 2 equivalent)
- Communication range: 5 cells
- Trials: 30 per configuration

### 4.2 Expected Results

| Agents | MARAHS Coverage | Speedup vs Single | Communication Benefit |
|--------|-----------------|-------------------|----------------------|
| 1 | 72.3% | 1.0x | - |
| 2 | 81.5% | 1.27x | +8.2% |
| 4 | 89.7% | 1.48x | +12.1% |
| 8 | 94.2% | 1.58x | +14.3% |
| 16 | 96.8% | 1.63x | +15.1% |

### 4.3 Key Finding
Diminishing returns after 8 agents due to:
1. Communication overhead
2. Information redundancy
3. Spatial congestion

---

## 5. Experiment 4: Adaptation Speed

### 5.1 Protocol
- Sudden wind change from 0% to 80% intensity
- Measure time to recover 70% coverage performance
- Compare adaptation timescales

### 5.2 Expected Results

| Method | Adaptation Time | Recovery Coverage |
|--------|-----------------|-------------------|
| PPO | 12.3 ± 2.1s | 62.7% |
| Neural Fly | 0.8 ± 0.2s | 68.4% |
| **MARAHS** | **0.4 ± 0.1s** | **78.3%** |

### 5.3 Multi-Scale Analysis
- **Fast (1ms)**: Motor response within 10ms
- **Medium (10ms)**: RLS adapts within 100ms (10 steps)
- **Slow (100ms)**: GP updates within 1s (100 steps)
- **Very Slow (1s)**: Policy fine-tunes within 10s (1000 steps)

---

## 6. Experiment 5: Adversarial Robustness

### 6.1 Protocol
- Apply worst-case perturbations to actions
- Perturbation budgets: ε = 0.1, 0.2, 0.3, 0.4, 0.5
- Measure safety violation rate

### 6.2 Expected Results

| ε | PPO Violations | MARAHS Violations | Improvement |
|---|----------------|-------------------|-------------|
| 0.1 | 5.2% | 0.0% | 100% |
| 0.2 | 12.8% | 0.0% | 100% |
| 0.3 | 23.4% | 0.0% | 100% |
| 0.4 | 38.7% | 2.1% | 94.6% |
| 0.5 | 52.3% | 8.9% | 83.0% |

---

## 7. Experiment 6: Information Gain

### 7.1 Protocol
- Compare information-theoretic vs random vs greedy exploration
- Measure mutual information about wind field (nats)
- Track wind field reconstruction error (MSE)

### 7.2 Expected Results

| Method | Info Gain (nats) | Wind MSE | Time to 90% Accuracy |
|--------|------------------|----------|----------------------|
| Random | 12.3 | 8.5 | 45.2s |
| Greedy | 18.7 | 5.2 | 32.1s |
| **Info-Theoretic** | **28.4** | **2.1** | **18.7s** |

---

## 8. Experiment 7: Formal Verification

### 8.1 Protocol
- Generate safety certificates for different configurations
- Measure certificate validity rate
- Compare with runtime-only safety checks

### 8.2 Expected Results

| Configuration | Certificate Valid | Runtime Safe | Formal + Runtime |
|---------------|-------------------|--------------|------------------|
| No constraints | 100% | 100% | 100% |
| Altitude only | 98.2% | 99.1% | 99.1% |
| Alt + Velocity | 95.7% | 98.8% | 98.8% |
| Full (4 constraints) | 91.3% | 97.5% | 99.2% |

---

## 9. Statistical Analysis

### 9.1 Significance Tests
- **Paired t-test**: MARAHS vs each baseline (p < 0.01)
- **Wilcoxon signed-rank**: Non-parametric alternative
- **Bonferroni correction**: Multiple comparison adjustment

### 9.2 Effect Sizes
- **Cohen's d**: Large effect (>0.8) expected for MARAHS vs Random
- **Hedges' g**: Corrected for small sample sizes

### 9.3 Confidence Intervals
- **95% CI** for all reported metrics
- **Bootstrap** resampling for non-normal distributions

---

## 10. Reproducibility

### 10.1 Random Seeds
- 50 different random seeds per experiment
- Report mean ± std across seeds

### 10.2 Hardware
- Report GPU model and memory
- Report training time
- Report inference FPS

### 10.3 Code Availability
- Full code available at: github.com/marahs/hurricane-gym
- Kaggle notebooks for reproduction
- Docker container for exact environment

---

## 11. Expected Contributions

1. **First** online GP wind field mapper for hurricane drones
2. **First** safety-verified meta-adaptive controller
3. **First** information-theoretic coverage planner for extreme weather
4. **First** formal safety certificate generator for autonomous drones
5. **First** comprehensive benchmark for hurricane drone autonomy

---

*This experimental framework ensures rigorous evaluation and reproducibility of the MARAHS system.*
