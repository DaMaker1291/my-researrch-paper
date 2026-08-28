"""
MARAHS v2: Multi-Agent Robust Autonomous Hurricane Swarm
==========================================================
Research-grade cooperative coverage environment for 10-drone swarm.

Key improvements over v1:
- 25x25 grid (625 cells) with proper wind physics
- 10 drones with inter-agent collision avoidance (CBF)
- Realistic Rankine vortex hurricane model (Cat 1-5)
- Communication range-limited observations
- Velocity persistence (momentum)
- Proper coverage reward shaping
- Multiple hurricane profiles (Katrina, Harvey, Irma, Maria, Michael)

Research metrics:
- Coverage % over time
- Coverage under different hurricane categories
- Multi-agent scaling (1→2→4→6→8→10 drones)
- Inter-agent collision avoidance statistics
- Wind compensation effectiveness
- Information-theoretic vs greedy coverage
"""

import numpy as np
from gymnasium import spaces, Env
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional
import math


# ─── Hurricane Profiles ──────────────────────────────────────────────────────

HURRICANE_PROFILES = {
    'katrina': {
        'R_max_km': 45.0, 'V_max_kts': 150, 'central_pressure_hPa': 902,
        'category': 5, 'asymmetry': 0.15,
    },
    'harvey': {
        'R_max_km': 30.0, 'V_max_kts': 130, 'central_pressure_hPa': 937,
        'category': 4, 'asymmetry': 0.20,
    },
    'irma': {
        'R_max_km': 65.0, 'V_max_kts': 160, 'central_pressure_hPa': 914,
        'category': 5, 'asymmetry': 0.10,
    },
    'maria': {
        'R_max_km': 25.0, 'V_max_kts': 175, 'central_pressure_hPa': 908,
        'category': 5, 'asymmetry': 0.25,
    },
    'michael': {
        'R_max_km': 20.0, 'V_max_kts': 160, 'central_pressure_hPa': 919,
        'category': 5, 'asymmetry': 0.18,
    },
}


# Category to m/s mapping
CAT_TO_MS = {
    0: 0.0,
    1: 33.0,   # 33-42 m/s
    2: 42.0,   # 43-49 m/s
    3: 50.0,   # 50-58 m/s
    4: 58.0,   # 58-70 m/s
    5: 70.0,   # >70 m/s
}

# m/s to grid units per step
# 1 m/s ≈ 0.02 cells/step at 10Hz with 10m cells


@dataclass
class SwarmGridConfig:
    """Configuration for the v2 multi-agent coverage environment."""
    grid_size: int = 25
    num_drones: int = 10
    max_steps: int = 400
    cell_size_m: float = 10.0  # meters per cell
    control_freq_hz: float = 10.0  # Hz

    # Wind
    wind_category: int = 3  # 1-5
    wind_intensity: float = 1.0  # multiplier [0, 1]
    hurricane_profile: str = 'katrina'
    wind_drift_speed: float = 0.05  # cells/step
    wind_gust_std: float = 0.3  # std of gust noise

    # Drone
    drone_speed: float = 1.5  # cells per step (max)
    drone_acceleration: float = 0.5  # cells per step²
    momentum_factor: float = 0.7  # velocity persistence [0,1]

    # Communication
    comm_range: float = 8.0  # cells
    min_separation: float = 2.5  # cells (collision avoidance)
    separation_buffer: float = 1.0  # extra buffer for CBF

    # Coverage
    coverage_radius: float = 0.5  # cells (cell is "covered" if drone within radius)
    num_debris: int = 5
    debris_radius: float = 0.5

    # Safety
    boundary_margin: float = 1.0  # cells from edge


class RankineVortexHurricane:
    """
    Realistic Rankine vortex hurricane wind model.

    Wind speed profile:
      Inner core (r < R_max):  V(r) = V_max * (r / R_max)
      Outer region (r >= R_max): V(r) = V_max * (R_max / r)^1.5

    Wind direction: tangential (clockwise in N. hemisphere)
    Includes radial inflow component and asymmetry.
    """

    def __init__(self, grid_size: int, profile_name: str = 'katrina',
                 rng: np.random.Generator = None):
        self.N = grid_size
        self.rng = rng or np.random.default_rng()

        profile = HURRICANE_PROFILES.get(profile_name, HURRICANE_PROFILES['katrina'])
        self.R_max = profile['R_max_km'] / 10.0  # convert km to grid units (10m cells)
        self.R_max = np.clip(self.R_max, 3.0, grid_size * 0.4)
        self.V_max = profile['V_max_kts'] * 0.5144 / 10.0  # kts → m/s → cells/step
        self.V_max = np.clip(self.V_max, 0.5, 3.0)
        self.asymmetry = profile['asymmetry']

        # Eye position and drift
        self.eye_pos = np.array([grid_size / 2, grid_size / 2])
        self.drift_dir = np.array([0.0, 0.0])
        self.drift_speed = 0.0

        # Gust memory
        self.gust_field = None

    def reset(self):
        """Randomize hurricane center and drift direction."""
        margin = self.N * 0.25
        self.eye_pos = np.array([
            self.rng.uniform(margin, self.N - margin),
            self.rng.uniform(margin, self.N - margin),
        ])
        angle = self.rng.uniform(0, 2 * np.pi)
        self.drift_dir = np.array([np.cos(angle), np.sin(angle)])
        self.drift_speed = self.rng.uniform(0.01, 0.08)

        # Pre-generate gust field (spatially correlated noise)
        self.gust_field = self.rng.normal(0, 1, (self.N, self.N, 2))

    def step(self):
        """Advance hurricane state."""
        self.eye_pos += self.drift_dir * self.drift_speed
        self.eye_pos = np.clip(self.eye_pos, -self.N * 0.3, self.N * 1.3)

    def get_wind(self, row: float, col: float, intensity: float = 1.0,
                 t: int = 0) -> Tuple[float, float]:
        """Get wind vector (wx, wy) at position (row, col)."""
        dr = row - self.eye_pos[0]
        dc = col - self.eye_pos[1]
        r = math.sqrt(dr * dr + dc * dc)

        if r < 0.1:
            return 0.0, 0.0

        # Rankine vortex wind speed
        if r < self.R_max:
            speed = self.V_max * (r / self.R_max)
        else:
            speed = self.V_max * (self.R_max / max(r, 0.5)) ** 1.5

        speed *= intensity

        # Tangential direction (clockwise)
        # Add asymmetry (wavenumber-1)
        theta = math.atan2(dr, dc)
        asym = 1.0 + self.asymmetry * math.cos(theta - math.pi / 4)
        speed *= asym

        # Tangential wind
        wx = speed * (-dc / r)
        wy = speed * (dr / r)

        # Add gust noise
        r_int, c_int = int(np.clip(row, 0, self.N - 1)), int(np.clip(col, 0, self.N - 1))
        gust_scale = 0.3 * intensity
        if self.gust_field is not None:
            wx += self.gust_field[r_int, c_int, 0] * gust_scale
            wy += self.gust_field[r_int, c_int, 1] * gust_scale

        return float(wx), float(wy)

    def get_wind_speed(self, row: float, col: float, intensity: float = 1.0) -> float:
        """Get scalar wind speed at position."""
        wx, wy = self.get_wind(row, col, intensity)
        return math.sqrt(wx * wx + wy * wy)


class SwarmGridWorldV2(Env):
    """
    Multi-drone cooperative coverage with 10 drones and hurricane physics.

    Each drone observes:
      - own position (2)
      - velocity (2)
      - direction to nearest uncovered cell (2)
      - distance to nearest uncovered cell (1)
      - global coverage fraction (1)
      - neighbor count (1)
      - local wind vector (2)
      - safety margin to nearest drone (1)
      - nearest drone direction (2)
      - max 9 neighbor positions (18)

    Total obs: 32 features per drone

    Action: 5 discrete (Stay, N, S, E, W)
    """

    MOVES = np.array([[0, 0], [-1, 0], [1, 0], [0, 1], [0, -1]], dtype=np.float64)
    MOVE_NAMES = ['Stay', 'N', 'S', 'E', 'W']

    def __init__(self, config: SwarmGridConfig = None, render_mode=None):
        super().__init__()
        self.config = config or SwarmGridConfig()
        self.render_mode = render_mode

        self.N = self.config.grid_size
        self.K = self.config.num_drones
        self.total_cells = self.N * self.N

        self.obs_size = 32  # fixed, padding for variable neighbors
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(self.K, self.obs_size), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        # State
        self.positions = np.zeros((self.K, 2), dtype=np.float64)
        self.velocities = np.zeros((self.K, 2), dtype=np.float64)
        self.coverage = np.zeros((self.N, self.N), dtype=bool)
        self.debris = np.zeros((self.N, self.N), dtype=bool)
        self.step_count = 0

        # Wind
        self.wind = RankineVortexHurricane(
            self.N, self.config.hurricane_profile
        )

        # Collision avoidance stats
        self.collision_avoidances = 0
        self.total_separations = 0
        self.min_dists_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.K = self.config.num_drones
        self.step_count = 0
        self.coverage = np.zeros((self.N, self.N), dtype=bool)
        self.debris = np.zeros((self.N, self.N), dtype=bool)
        self.velocities = np.zeros((self.K, 2), dtype=np.float64)
        self.collision_avoidances = 0
        self.total_separations = 0
        self.min_dists_history = []

        # Initialize drone positions in a grid pattern across the map
        n_rows = int(math.ceil(math.sqrt(self.K)))
        n_cols = int(math.ceil(self.K / n_rows))
        margin = 3  # cells from edge

        for i in range(self.K):
            r = i // n_cols
            c = i % n_cols
            row = margin + r * ((self.N - 2 * margin) // max(n_rows - 1, 1))
            col = margin + c * ((self.N - 2 * margin) // max(n_cols - 1, 1))
            row = np.clip(row + self.np_random.integers(-1, 2), margin, self.N - margin - 1)
            col = np.clip(col + self.np_random.integers(-1, 2), margin, self.N - margin - 1)
            self.positions[i] = [float(row), float(col)]

        # Mark starting positions as covered
        for i in range(self.K):
            r, c = int(self.positions[i, 0]), int(self.positions[i, 1])
            self.coverage[r, c] = True

        # Place debris
        if self.config.num_debris > 0:
            placed = 0
            attempts = 0
            while placed < self.config.num_debris and attempts < 1000:
                r = self.np_random.integers(2, self.N - 2)
                c = self.np_random.integers(2, self.N - 2)
                if not self.debris[r, c] and not self.coverage[r, c]:
                    self.debris[r, c] = True
                    placed += 1
                attempts += 1

        # Reset hurricane
        self.wind.reset()

        return self._get_obs(), {}

    def _get_drone_obs(self, drone_idx: int) -> np.ndarray:
        obs = np.zeros(self.obs_size, dtype=np.float32)
        offset = 0

        # Own position (normalized to [-1, 1])
        obs[offset] = self.positions[drone_idx, 0] / self.N * 2 - 1
        obs[offset + 1] = self.positions[drone_idx, 1] / self.N * 2 - 1
        offset += 2

        # Velocity (normalized by max speed)
        max_s = max(self.config.drone_speed, 0.1)
        obs[offset] = self.velocities[drone_idx, 0] / max_s
        obs[offset + 1] = self.velocities[drone_idx, 1] / max_s
        offset += 2

        # Direction and distance to nearest uncovered cell
        uncovered = np.argwhere(~self.coverage)
        if len(uncovered) > 0:
            pos = self.positions[drone_idx]
            dists = np.linalg.norm(uncovered - pos, axis=1)
            nearest_idx = np.argmin(dists)
            nearest = uncovered[nearest_idx]
            dr = nearest[0] - pos[0]
            dc = nearest[1] - pos[1]
            dist = dists[nearest_idx]
            norm = max(dist, 0.01)
            obs[offset] = dr / norm
            obs[offset + 1] = dc / norm
            obs[offset + 2] = dist / (self.N * 1.414)  # normalize by diagonal
        offset += 3

        # Global coverage fraction
        obs[offset] = self.coverage.sum() / self.total_cells * 2 - 1
        offset += 1

        # Number of neighbors in communication range
        n_neighbors = 0
        min_dist = float('inf')
        nearest_neighbor_dir = np.array([0.0, 0.0])

        for j in range(self.K):
            if j != drone_idx:
                d = np.linalg.norm(self.positions[drone_idx] - self.positions[j])
                if d <= self.config.comm_range:
                    n_neighbors += 1
                if d < min_dist:
                    min_dist = d
                    diff = self.positions[j] - self.positions[drone_idx]
                    norm_d = max(d, 0.01)
                    nearest_neighbor_dir = diff / norm_d

        obs[offset] = n_neighbors / max(self.K - 1, 1) * 2 - 1
        offset += 1

        # Local wind vector
        wx, wy = self.wind.get_wind(
            self.positions[drone_idx, 0],
            self.positions[drone_idx, 1],
            self.config.wind_intensity
        )
        obs[offset] = np.clip(wx / max(self.wind.V_max, 0.1), -1, 1)
        obs[offset + 1] = np.clip(wy / max(self.wind.V_max, 0.1), -1, 1)
        offset += 2

        # Safety margin (distance to nearest drone / min separation)
        sep_ratio = min_dist / max(self.config.min_separation, 0.1)
        obs[offset] = np.clip(sep_ratio - 1.0, -1, 1)
        offset += 1

        # Direction to nearest drone
        obs[offset] = nearest_neighbor_dir[0]
        obs[offset + 1] = nearest_neighbor_dir[1]
        offset += 2

        # Neighbor positions (up to 9 neighbors, normalized)
        neighbor_count = 0
        for j in range(self.K):
            if j != drone_idx and neighbor_count < 9 and offset + 1 < self.obs_size:
                d = np.linalg.norm(self.positions[drone_idx] - self.positions[j])
                if d <= self.config.comm_range:
                    obs[offset] = self.positions[j, 0] / self.N * 2 - 1
                    obs[offset + 1] = self.positions[j, 1] / self.N * 2 - 1
                    neighbor_count += 1
                offset += 2
            else:
                break

        return obs

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [self._get_drone_obs(i) for i in range(self.K)],
            dtype=np.float32,
        )

    def _apply_collision_avoidance(self, new_positions: np.ndarray,
                                    velocities: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Apply inter-agent collision avoidance (CBF-inspired).

        For each pair of drones closer than min_separation + buffer,
        push them apart along the line connecting them.

        Returns:
            corrected_positions: (K, 2) positions after avoidance
            n_avoidances: number of collision avoidances applied
        """
        corrected = new_positions.copy()
        n_avoid = 0
        min_sep = self.config.min_separation

        for _ in range(5):  # multiple iterations for dense swarms
            any_changed = False
            for i in range(self.K):
                for j in range(i + 1, self.K):
                    diff = corrected[i] - corrected[j]
                    dist = np.linalg.norm(diff)
                    threshold = min_sep + self.config.separation_buffer

                    if dist < threshold and dist > 0.01:
                        # Push apart
                        overlap = threshold - dist
                        direction = diff / dist
                        push = direction * overlap * 0.6  # CBF gain

                        corrected[i] += push
                        corrected[j] -= push

                        n_avoid += 1
                        any_changed = True

            if not any_changed:
                break

        # Clip to grid
        corrected = np.clip(corrected, 0, self.N - 1)

        return corrected, n_avoid

    def step(self, actions):
        self.step_count += 1
        self.wind.step()

        # ── Phase 1: Compute desired new positions from actions ──
        desired_new = np.zeros((self.K, 2), dtype=np.float64)

        for i in range(self.K):
            action = int(actions[i])
            delta = self.MOVES[action]

            if self.config.momentum_factor > 0:
                self.velocities[i] = (
                    self.config.momentum_factor * self.velocities[i]
                    + (1 - self.config.momentum_factor) * delta
                )
                # Clip speed
                speed = np.linalg.norm(self.velocities[i])
                if speed > self.config.drone_speed:
                    self.velocities[i] *= self.config.drone_speed / speed
                desired_new[i] = self.positions[i] + self.velocities[i]
            else:
                desired_new[i] = self.positions[i] + delta

        # ── Phase 2: Collision avoidance ──
        corrected, n_avoid = self._apply_collision_avoidance(
            desired_new, self.velocities
        )
        self.collision_avoidances += n_avoid

        # ── Phase 3: Debris avoidance and wind ──
        for i in range(self.K):
            new_pos = corrected[i]

            # Check debris
            r, c = int(np.clip(new_pos[0], 0, self.N - 1)), int(np.clip(new_pos[1], 0, self.N - 1))
            if self.debris[r, c]:
                # Revert to old position
                new_pos = self.positions[i].copy()
                self.velocities[i] *= -0.5

            # Apply wind
            wx, wy = self.wind.get_wind(
                new_pos[0], new_pos[1], self.config.wind_intensity
            )
            wind_force = np.array([wx, wy]) * 0.5
            wind_pos = new_pos + wind_force
            wind_pos = np.clip(wind_pos, 0, self.N - 1)

            wr, wc = int(wind_pos[0]), int(wind_pos[1])
            if not self.debris[wr, wc]:
                self.positions[i] = wind_pos
            else:
                self.positions[i] = new_pos

        # ── Phase 4: Update coverage ──
        new_per_drone = np.zeros(self.K, dtype=bool)
        total_new = 0

        for i in range(self.K):
            # Mark the nearest cell as covered
            r = int(round(self.positions[i, 0]))
            c = int(round(self.positions[i, 1]))
            r = int(np.clip(r, 0, self.N - 1))
            c = int(np.clip(c, 0, self.N - 1))
            if not self.coverage[r, c]:
                self.coverage[r, c] = True
                total_new += 1
                new_per_drone[i] = True

        # Track minimum inter-agent distance
        min_d = float('inf')
        for i in range(self.K):
            for j in range(i + 1, self.K):
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                min_d = min(min_d, d)
        self.min_dists_history.append(min_d)

        coverage_pct = self.coverage.sum() / self.total_cells * 100.0

        # ── Phase 5: Reward ──
        uncovered = np.argwhere(~self.coverage)
        reward = np.full(self.K, -0.005)  # tiny step penalty

        for i in range(self.K):
            # Large reward for covering new cells
            if new_per_drone[i]:
                reward[i] += 3.0

            # Shaped reward: encourage movement toward uncovered areas
            if len(uncovered) > 0:
                dists = np.linalg.norm(uncovered - self.positions[i], axis=1)
                min_dist = np.min(dists)
                # Reward inversely proportional to distance to nearest uncovered
                reward[i] += max(0, 0.5 * (1.0 - min_dist / (self.N * 1.414)))

            # Small penalty for being too close to other drones (soft constraint)
            for j in range(self.K):
                if j != i:
                    d = np.linalg.norm(self.positions[i] - self.positions[j])
                    if d < self.config.min_separation:
                        reward[i] -= 0.5 * (1.0 - d / self.config.min_separation)

        # Bonus for full coverage
        if self.coverage.all():
            reward += 50.0

        # Bonus for high coverage
        if coverage_pct > 90:
            reward += 1.0 * (coverage_pct / 100.0)

        done = self.step_count >= self.config.max_steps

        # Information about nearest uncovered
        nearest_uncov_dist = np.full(self.K, self.N * 1.414)
        for i in range(self.K):
            if len(uncovered) > 0:
                dists = np.linalg.norm(uncovered - self.positions[i], axis=1)
                nearest_uncov_dist[i] = np.min(dists)

        infos = []
        for i in range(self.K):
            infos.append({
                'coverage_pct': coverage_pct,
                'cells_covered': int(self.coverage.sum()),
                'new_cells_total': total_new,
                'positions': self.positions.copy(),
                'wind_eye': self.wind.eye_pos.copy(),
                'collision_avoidances': self.collision_avoidances,
                'min_inter_agent_dist': float(min_d),
                'nearest_uncov_dist': float(nearest_uncov_dist[i]),
                'wind_speed_at_drone': self.wind.get_wind_speed(
                    self.positions[i, 0], self.positions[i, 1],
                    self.config.wind_intensity
                ),
                'step': self.step_count,
            })

        return (
            self._get_obs(),
            reward,
            np.full(self.K, done),
            np.full(self.K, False),
            infos,
        )

    def get_metrics(self) -> Dict:
        """Get current environment metrics."""
        min_d = float('inf')
        for i in range(self.K):
            for j in range(i + 1, self.K):
                d = np.linalg.norm(self.positions[i] - self.positions[j])
                min_d = min(min_d, d)

        return {
            'coverage_pct': float(self.coverage.sum() / self.total_cells * 100),
            'cells_covered': int(self.coverage.sum()),
            'collision_avoidances': self.collision_avoidances,
            'min_inter_agent_dist': float(min_d),
            'steps': self.step_count,
        }
