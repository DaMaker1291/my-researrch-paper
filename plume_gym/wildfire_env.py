"""
WildfirePlumeEnv: A physics-informed wildfire simulation environment.

Fire spread uses a simplified Farsite-like model with:
  - Rothermel fire spread rate equations
  - Wind-driven asymmetric spread
  - Fuel moisture content effects
  - Spotting (ember transport)
  - Convective plume dynamics

Drone dynamics model quadrotor UAVs with:
  - 6-DOF state (position, velocity)
  - Discrete or continuous actions
  - Wind coupling (atmospheric + fire-induced thermal plumes)
  - Communication range constraints
  - Battery life modeling
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional
import time


@dataclass
class WildfireConfig:
    """Configuration for the wildfire simulation environment."""
    # Grid
    grid_size: int = 40              # 40x40 grid (1600 cells)
    cell_size: float = 10.0          # 10m per cell

    # Fire
    base_spread_rate: float = 0.05   # cells per step
    wind_spread_multiplier: float = 5.0  # wind amplification
    max_fire_intensity: float = 1.0  # fully burned
    fuel_moisture: float = 0.15      # 0-1, higher = harder to burn
    spotting_probability: float = 0.02  # ember jump probability
    spotting_distance: int = 3       # max ember jump cells
    fire_start_cells: int = 6        # initial fire perimeter cells

    # Atmosphere
    ambient_wind_speed: float = 10.0    # m/s base wind
    ambient_wind_dir: float = 0.0       # radians, 0 = east
    gust_amplitude: float = 5.0         # m/s gust variation
    gust_frequency: float = 0.08        # spatial frequency
    thermal_plume_strength: float = 10.0  # m/s updraft at fire center
    thermal_plume_radius: float = 3.0    # cells radius of influence
    turbulence_intensity: float = 0.8    # noise scale

    # Drones
    num_drones: int = 6
    drone_speed: float = 1.5           # cells per step max
    drone_battery_steps: int = 500     # steps before return-to-base
    drone_comm_range: float = 8.0      # cells
    drone_wind_resistance: float = 0.5  # how well drone resists wind (0-1)
    thermal_damage_threshold: float = 12.0  # m/s updraft = damage

    # Safety
    min_separation: float = 1.5       # cells between drones
    boundary_margin: float = 1.5      # cells from edge
    fire_damage_radius: float = 1.0   # cells from fire = damage

    # Episode
    max_steps: int = 600
    wind_change_interval: int = 100   # steps between wind shifts

    # Observations
    local_obs_radius: int = 5         # cells visible to each drone
    obs_channels: int = 7             # fire, wind_x, wind_y, fuel, drone_id, comm_mask, dist_to_fire

    # Action space
    discrete_actions: bool = True     # True = 5 actions, False = continuous

    def __post_init__(self):
        self.grid_area = self.grid_size ** 2
        obs_size = 2 * self.local_obs_radius + 1
        self.observation_space_size = self.obs_channels * obs_size * obs_size


@dataclass
class DroneState:
    """State of a single drone."""
    position: np.ndarray    # (2,) float, grid coordinates
    velocity: np.ndarray    # (2,) float
    battery: int = 500
    is_active: bool = True
    cells_visited: int = 0
    total_distance: float = 0.0
    crash_count: int = 0


class WildfirePlumeEnv:
    """
    Multi-agent wildfire perimeter tracking environment.

    The fire spreads according to simplified Rothermel dynamics with
    wind-driven asymmetry. Drones must track the fire perimeter while
    maintaining safety in convective plume winds.

    Observation space per drone (local):
        [fire_intensity, wind_x, wind_y, fuel_moisture,
         drone_id, comm_mask, dist_to_nearest_fire]
        Shape: (obs_channels, 2*local_obs_radius+1, 2*local_obs_radius+1)

    Action space:
        Discrete: {Stay, North, South, East, West}
        Continuous: (vx, vy) in [-1, 1]
    """

    def __init__(self, config: WildfireConfig = None):
        self.cfg = config or WildfireConfig()
        self.rng = np.random.default_rng()

        # Grids
        self.fire_grid = np.zeros((self.cfg.grid_size, self.cfg.grid_size), dtype=np.float32)
        self.fuel_grid = np.ones((self.cfg.grid_size, self.cfg.grid_size), dtype=np.float32)
        self.wind_x = np.zeros((self.cfg.grid_size, self.cfg.grid_size), dtype=np.float32)
        self.wind_y = np.zeros((self.cfg.grid_size, self.cfg.grid_size), dtype=np.float32)
        self.thermal_plume = np.zeros((self.cfg.grid_size, self.cfg.grid_size), dtype=np.float32)
        self.burned_history = np.zeros((self.cfg.grid_size, self.cfg.grid_size), dtype=np.float32)

        # Drones
        self.drones: List[DroneState] = []
        self.drone_positions_history: List[List[np.ndarray]] = []

        # Episode tracking
        self.step_count = 0
        self.total_fire_cells = 0
        self.total_burned_cells = 0
        self.perimeter_cells = 0
        self.perimeter_visited = 0
        self.safety_violations = 0
        self.total_actions = 0
        self.wind_phase = 0.0

        # Coordinate grids (precomputed)
        yy, xx = np.meshgrid(
            np.arange(self.cfg.grid_size),
            np.arange(self.cfg.grid_size),
            indexing='ij'
        )
        self.xx = xx.astype(np.float32)
        self.yy = yy.astype(np.float32)

    def reset(self, seed: int = None) -> np.ndarray:
        """Reset the environment and return initial observations."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Reset grids
        self.fire_grid[:] = 0
        self.fuel_grid[:] = 1.0
        self.wind_x[:] = 0
        self.wind_y[:] = 0
        self.thermal_plume[:] = 0
        self.burned_history[:] = 0

        # Random fuel moisture patches
        n_patches = self.rng.integers(3, 8)
        for _ in range(n_patches):
            cx, cy = self.rng.integers(0, self.cfg.grid_size, size=2)
            radius = self.rng.integers(3, 8)
            mask = (self.xx - cx)**2 + (self.yy - cy)**2 < radius**2
            self.fuel_grid[mask] = self.rng.uniform(0.05, 0.4)

        # Initialize fire at random location (not at edge)
        margin = 5
        fire_cx = self.rng.integers(margin, self.cfg.grid_size - margin)
        fire_cy = self.rng.integers(margin, self.cfg.grid_size - margin)
        fire_radius = self.cfg.fire_start_cells
        mask = (self.xx - fire_cx)**2 + (self.yy - fire_cy)**2 < fire_radius**2
        self.fire_grid[mask] = 0.8

        # Initialize drones in a formation away from fire
        self.drones = []
        self.drone_positions_history = []
        spawn_angle_offset = self.rng.uniform(0, 2 * np.pi)

        for i in range(self.cfg.num_drones):
            # Spawn outside fire zone (need to be > fire_radius + safety margin)
            angle = spawn_angle_offset + (2 * np.pi * i / self.cfg.num_drones)
            spawn_dist = self.cfg.grid_size * 0.3  # far from fire center
            sx = np.clip(
                fire_cx + spawn_dist * np.cos(angle),
                self.cfg.boundary_margin + 2,
                self.cfg.grid_size - self.cfg.boundary_margin - 2
            )
            sy = np.clip(
                fire_cy + spawn_dist * np.sin(angle),
                self.cfg.boundary_margin + 2,
                self.cfg.grid_size - self.cfg.boundary_margin - 2
            )
            drone = DroneState(
                position=np.array([sx, sy], dtype=np.float32),
                velocity=np.array([0.0, 0.0], dtype=np.float32),
                battery=self.cfg.drone_battery_steps,
            )
            self.drones.append(drone)
            self.drone_positions_history.append([drone.position.copy()])

        # Initialize wind field
        self._update_wind_field()

        # Reset counters
        self.step_count = 0
        self.total_fire_cells = int(np.sum(self.fire_grid > 0.1))
        self.total_burned_cells = 0
        self.perimeter_cells = 0
        self.perimeter_visited = 0
        self.safety_violations = 0
        self.total_actions = 0
        self.wind_phase = 0.0

        return self._get_all_observations()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Execute one step.

        Args:
            actions: (num_drones,) array of discrete actions {0=Stay, 1=N, 2=S, 3=E, 4=W}
                     or (num_drones, 2) for continuous.

        Returns:
            observations: (num_drones, obs_dim) array
            rewards: (num_drones,) array
            dones: (num_drones,) boolean array
            infos: list of dicts with per-drone metrics
        """
        self.step_count += 1
        dones = np.zeros(self.cfg.num_drones, dtype=bool)
        infos = [{} for _ in range(self.cfg.num_drones)]

        # ── 1. Update fire spread ──
        self._spread_fire()
        self._update_wind_field()
        self._update_thermal_plume()

        # ── 2. Execute drone actions ──
        rewards = np.zeros(self.cfg.num_drones, dtype=np.float32)
        prev_positions = [d.position.copy() for d in self.drones]

        for i in range(self.cfg.num_drones):
            if not self.drones[i].is_active:
                continue

            # Apply action
            dx, dy = self._action_to_displacement(actions[i] if self.cfg.discrete_actions else actions[i])
            new_pos = self.drones[i].position + np.array([dx, dy], dtype=np.float32)

            # Add wind coupling
            wx = float(self.wind_x[int(np.clip(new_pos[0], 0, self.cfg.grid_size-1)),
                                     int(np.clip(new_pos[1], 0, self.cfg.grid_size-1))])
            wy = float(self.wind_y[int(np.clip(new_pos[0], 0, self.cfg.grid_size-1)),
                                     int(np.clip(new_pos[1], 0, self.cfg.grid_size-1))])
            wind_push = np.array([wx, wy]) * (1.0 - self.cfg.drone_wind_resistance) * 0.08
            new_pos += wind_push

            # Thermal plume push (updraft + horizontal)
            thermal = float(self.thermal_plume[int(np.clip(new_pos[0], 0, self.cfg.grid_size-1)),
                                                int(np.clip(new_pos[1], 0, self.cfg.grid_size-1))])
            thermal_push_x = self.rng.normal(0, thermal * 0.08)
            thermal_push_y = self.rng.normal(0, thermal * 0.08)
            new_pos[0] += thermal_push_x
            new_pos[1] += thermal_push_y

            # Boundary constraint
            new_pos = np.clip(new_pos, self.cfg.boundary_margin,
                              self.cfg.grid_size - self.cfg.boundary_margin - 1)

            # Velocity update
            displacement = new_pos - self.drones[i].position
            self.drones[i].velocity = displacement
            self.drones[i].position = new_pos
            self.drones[i].battery -= 1

            # ── Reward computation ──
            reward = 0.0

            # (a) Survival bonus
            reward += 0.5

            # (b) Fire perimeter tracking reward (dense potential-based)
            nearest_fire_dist = self._distance_to_nearest_fire(new_pos)
            # Reward for being near fire perimeter (3-6 cells away = sweet spot)
            if 2.0 < nearest_fire_dist < 6.0:
                reward += (6.0 - abs(nearest_fire_dist - 4.0)) * 1.5  # peak at 4 cells
            elif nearest_fire_dist <= 2.0:
                # Too close = danger penalty
                reward -= (2.0 - nearest_fire_dist) * 8.0
            else:
                # Too far = small penalty to encourage approach
                reward -= 0.3

            # (c) Information gain reward (novelty)
            ix, iy = int(new_pos[0]), int(new_pos[1])
            if 0 <= ix < self.cfg.grid_size and 0 <= iy < self.cfg.grid_size:
                if self.burned_history[ix, iy] < 0.1:
                    reward += 3.0  # first visit bonus
                    self.burned_history[ix, iy] = 1.0
                    self.drones[i].cells_visited += 1

            # (d) Perimeter coverage reward
            if self._is_on_perimeter(new_pos):
                reward += 5.0
                self.perimeter_visited += 1

            # (e) Smoothness penalty
            if len(self.drone_positions_history[i]) > 1:
                prev = self.drone_positions_history[i][-1]
                jerk = np.sum((new_pos - 2 * prev + (prev - self.drone_positions_history[i][-2] if len(self.drone_positions_history[i]) > 1 else prev))**2)
                reward -= 0.1 * jerk

            # (f) Communication reward (staying near teammates)
            for j in range(self.cfg.num_drones):
                if j != i and self.drones[j].is_active:
                    dist = np.linalg.norm(new_pos - self.drones[j].position)
                    if dist < self.cfg.drone_comm_range:
                        reward += 0.1  # bonus for maintaining comms

            # (g) Crash penalty - probabilistic based on thermal/wind severity
            crash = False
            if thermal > self.cfg.thermal_damage_threshold:
                # Probability of crash scales with thermal excess
                crash_prob = min(0.8, (thermal - self.cfg.thermal_damage_threshold) / 10.0)
                if self.rng.random() < crash_prob:
                    crash = True
                    reward -= 20.0
            if self._is_in_fire(new_pos):
                crash = True
                reward -= 25.0
            # Wind-induced crash at high wind speeds
            wind_speed = float(np.sqrt(wx**2 + wy**2))
            if wind_speed > 25.0 and self.rng.random() < 0.1:
                crash = True
                reward -= 15.0

            # Separation violation
            for j in range(self.cfg.num_drones):
                if j != i and self.drones[j].is_active:
                    dist = np.linalg.norm(new_pos - self.drones[j].position)
                    if dist < self.cfg.min_separation:
                        reward -= 5.0
                        self.safety_violations += 1

            # Battery depletion (return to base, not crash)
            if self.drones[i].battery <= 0:
                self.drones[i].is_active = False
                dones[i] = True
                reward -= 5.0

            if crash:
                self.drones[i].crash_count += 1
                self.drones[i].is_active = False
                dones[i] = True

            self.total_actions += 1
            self.drones[i].total_distance += np.linalg.norm(displacement)
            self.drone_positions_history[i].append(new_pos.copy())
            rewards[i] = reward

        # ── 3. Check if all drones crashed ──
        all_crashed = all(not d.is_active for d in self.drones)
        if all_crashed:
            dones[:] = True

        # ── 4. Episode termination ──
        if self.step_count >= self.cfg.max_steps:
            dones[:] = True

        # ── 5. Compute perimeter tracking metrics ──
        self._update_perimeter()
        total_perimeter = max(1, self.perimeter_cells)
        tracking_fraction = self.perimeter_visited / (total_perimeter * self.step_count) if self.step_count > 0 else 0

        for i in range(self.cfg.num_drones):
            infos[i] = {
                'coverage': self.drones[i].cells_visited / self.cfg.grid_area * 100,
                'battery': self.drones[i].battery,
                'is_active': self.drones[i].is_active,
                'crashes': self.drones[i].crash_count,
                'distance_to_fire': self._distance_to_nearest_fire(self.drones[i].position) if self.drones[i].is_active else -1,
                'thermal': float(self.thermal_plume[int(np.clip(self.drones[i].position[0], 0, self.cfg.grid_size-1)),
                                                     int(np.clip(self.drones[i].position[1], 0, self.cfg.grid_size-1))]) if self.drones[i].is_active else 0,
                'perimeter_tracking': tracking_fraction * 100,
                'wind_speed': float(np.sqrt(self.wind_x**2 + self.wind_y**2).mean()),
            }

        return self._get_all_observations(), rewards, dones, infos

    # ── Fire Spread Model ──

    def _spread_fire(self):
        """Rothermel-inspired fire spread with wind asymmetry."""
        new_fire = self.fire_grid.copy()
        fire_mask = self.fire_grid > 0.1

        # Kernel for spread (8-connected + diagonal)
        kernel = np.array([[0.05, 0.1, 0.05],
                           [0.1,  0.0, 0.1],
                           [0.05, 0.1, 0.05]])

        # Convolve fire grid with kernel
        from scipy.signal import convolve2d
        fire_neighbors = convolve2d(fire_mask.astype(float), kernel, mode='same', boundary='fill')

        # Wind-driven spread rate
        wind_speed = np.sqrt(self.wind_x**2 + self.wind_y**2)
        spread_modifier = 1.0 + self.cfg.wind_spread_multiplier * wind_speed / 10.0

        # Fuel effect
        fuel_modifier = np.maximum(0, 1.0 - self.cfg.fuel_moisture * 2)

        # Spread probability
        spread_prob = self.cfg.base_spread_rate * spread_modifier * fuel_modifier * fire_neighbors

        # Random spread
        spread_noise = self.rng.random((self.cfg.grid_size, self.cfg.grid_size))
        spreading = (spread_noise < spread_prob) & (~fire_mask) & (self.fuel_grid > 0.05)

        # Increase fire intensity where already burning
        burn_increase = 0.05 * fire_mask * fuel_modifier
        new_fire = np.clip(new_fire + burn_increase + spreading.astype(float) * 0.3, 0, self.cfg.max_fire_intensity)

        # Fuel consumption
        consumed = 0.01 * fire_mask
        self.fuel_grid = np.maximum(0, self.fuel_grid - consumed)

        # Spotting (ember transport)
        fire_cells = np.argwhere(fire_mask)
        for cell in fire_cells:
            if self.rng.random() < self.cfg.spotting_probability:
                angle = self.rng.uniform(0, 2 * np.pi)
                dist = self.rng.integers(1, self.cfg.spotting_distance + 1)
                sx = int(cell[0] + dist * np.cos(angle))
                sy = int(cell[1] + dist * np.sin(angle))
                if 0 <= sx < self.cfg.grid_size and 0 <= sy < self.cfg.grid_size:
                    if self.fuel_grid[sx, sy] > 0.05 and new_fire[sx, sy] < 0.1:
                        new_fire[sx, sy] = 0.5

        self.fire_grid = new_fire
        self.total_fire_cells = int(np.sum(self.fire_grid > 0.1))

    def _update_wind_field(self):
        """Update wind field with time-varying gusts and fire-induced circulation."""
        self.wind_phase += 0.05

        # Ambient wind with sinusoidal variation
        wind_dir = self.cfg.ambient_wind_dir + 0.3 * np.sin(self.wind_phase)
        wind_speed = self.cfg.ambient_wind_speed + self.cfg.gust_amplitude * np.sin(self.wind_phase * 1.7)

        self.wind_x[:] = wind_speed * np.cos(wind_dir)
        self.wind_y[:] = wind_speed * np.sin(wind_dir)

        # Add spatial gusts (Perlin-like noise via superposition)
        for k in range(3):
            freq = self.cfg.gust_frequency * (2 ** k)
            amp = self.cfg.gust_amplitude / (2 ** k)
            self.wind_x += amp * np.sin(self.xx * freq + self.wind_phase * (k+1))
            self.wind_y += amp * np.cos(self.yy * freq + self.wind_phase * (k+1) * 0.7)

        # Fire-induced wind circulation (convergence toward fire center)
        fire_cells = np.argwhere(self.fire_grid > 0.3)
        if len(fire_cells) > 0:
            fire_cx = np.mean(fire_cells[:, 0])
            fire_cy = np.mean(fire_cells[:, 1])
            dx = fire_cx - self.xx
            dy = fire_cy - self.yy
            dist = np.sqrt(dx**2 + dy**2) + 0.1
            fire_wind_strength = 3.0 * self.fire_grid.mean()
            self.wind_x += fire_wind_strength * dx / dist
            self.wind_y += fire_wind_strength * dy / dist

        # Turbulence
        self.wind_x += self.rng.normal(0, self.cfg.turbulence_intensity, self.wind_x.shape)
        self.wind_y += self.rng.normal(0, self.cfg.turbulence_intensity, self.wind_y.shape)

    def _update_thermal_plume(self):
        """Compute thermal updraft from fire."""
        fire_cells = np.argwhere(self.fire_grid > 0.2)
        self.thermal_plume[:] = 0

        for cell in fire_cells:
            intensity = self.fire_grid[cell[0], cell[1]]
            r = np.sqrt((self.xx - cell[0])**2 + (self.yy - cell[1])**2)
            # Gaussian plume: peak at fire cell, decays with radius
            # Cap per-cell contribution to avoid runaway accumulation
            plume = self.cfg.thermal_plume_strength * intensity * np.exp(-r**2 / (2 * self.cfg.thermal_plume_radius**2))
            plume = np.minimum(plume, self.cfg.thermal_plume_strength * 0.5)  # cap per cell
            self.thermal_plume += plume

    # ── Observation Functions ──

    def _get_all_observations(self) -> np.ndarray:
        """Get observations for all drones. Returns (num_drones, obs_channels, obs_size, obs_size)."""
        r = self.cfg.local_obs_radius
        obs_size = 2 * r + 1
        all_obs = np.zeros((self.cfg.num_drones, self.cfg.obs_channels, obs_size, obs_size), dtype=np.float32)

        for i in range(self.cfg.num_drones):
            if not self.drones[i].is_active:
                continue

            cx, cy = int(self.drones[i].position[0]), int(self.drones[i].position[1])

            # Extract local patch
            x_min = max(0, cx - r)
            x_max = min(self.cfg.grid_size, cx + r + 1)
            y_min = max(0, cy - r)
            y_max = min(self.cfg.grid_size, cy + r + 1)

            # Channel 0: Fire intensity
            all_obs[i, 0, :x_max-x_min, :y_max-y_min] = self.fire_grid[x_min:x_max, y_min:y_max]

            # Channel 1-2: Wind field
            all_obs[i, 1, :x_max-x_min, :y_max-y_min] = self.wind_x[x_min:x_max, y_min:y_max]
            all_obs[i, 2, :x_max-x_min, :y_max-y_min] = self.wind_y[x_min:x_max, y_min:y_max]

            # Channel 3: Fuel moisture
            all_obs[i, 3, :x_max-x_min, :y_max-y_min] = self.fuel_grid[x_min:x_max, y_min:y_max]

            # Channel 4: Thermal plume
            all_obs[i, 4, :x_max-x_min, :y_max-y_min] = self.thermal_plume[x_min:x_max, y_min:y_max]

            # Channel 5: Other drone positions (relative)
            for j in range(self.cfg.num_drones):
                if j != i and self.drones[j].is_active:
                    jx = int(self.drones[j].position[0]) - cx + r
                    jy = int(self.drones[j].position[1]) - cy + r
                    if 0 <= jx < obs_size and 0 <= jy < obs_size:
                        all_obs[i, 5, jx, jy] = 1.0

            # Channel 6: Distance to nearest fire edge
            fire_mask = self.fire_grid > 0.1
            if np.any(fire_mask):
                from scipy.ndimage import distance_transform_edt
                dist_to_fire = distance_transform_edt(~fire_mask)
                all_obs[i, 6, :x_max-x_min, :y_max-y_min] = dist_to_fire[x_min:x_max, y_min:y_max]

        return all_obs

    # ── Action Helpers ──

    def _action_to_displacement(self, action) -> Tuple[float, float]:
        """Convert action to (dx, dy) displacement."""
        if self.cfg.discrete_actions:
            action = int(action)
            if action == 0: return (0.0, 0.0)        # Stay
            elif action == 1: return (0.0, self.cfg.drone_speed)   # North (+y)
            elif action == 2: return (0.0, -self.cfg.drone_speed)  # South (-y)
            elif action == 3: return (self.cfg.drone_speed, 0.0)   # East (+x)
            elif action == 4: return (-self.cfg.drone_speed, 0.0)  # West (-x)
            else: return (0.0, 0.0)
        else:
            # Continuous: action is (vx, vy) in [-1, 1]
            return (float(action[0]) * self.cfg.drone_speed,
                    float(action[1]) * self.cfg.drone_speed)

    # ── Safety Query Functions ──

    def _distance_to_nearest_fire(self, pos: np.ndarray) -> float:
        """Distance from position to nearest fire cell."""
        fire_cells = np.argwhere(self.fire_grid > 0.1)
        if len(fire_cells) == 0:
            return float(self.cfg.grid_size)
        dists = np.sqrt(np.sum((fire_cells - pos)**2, axis=1))
        return float(np.min(dists))

    def _is_in_fire(self, pos: np.ndarray) -> bool:
        """Check if position is inside active fire."""
        ix, iy = int(pos[0]), int(pos[1])
        if 0 <= ix < self.cfg.grid_size and 0 <= iy < self.cfg.grid_size:
            return self.fire_grid[ix, iy] > 0.5
        return False

    def _is_on_perimeter(self, pos: np.ndarray) -> bool:
        """Check if position is on the fire perimeter (edge of burning area)."""
        ix, iy = int(pos[0]), int(pos[1])
        if 0 <= ix < self.cfg.grid_size and 0 <= iy < self.cfg.grid_size:
            if self.fire_grid[ix, iy] > 0.1:
                # Check if any neighbor is not burning
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = ix + dx, iy + dy
                        if 0 <= nx < self.cfg.grid_size and 0 <= ny < self.cfg.grid_size:
                            if self.fire_grid[nx, ny] < 0.1:
                                return True
        return False

    def _update_perimeter(self):
        """Count perimeter cells."""
        fire_mask = self.fire_grid > 0.1
        # Perimeter = fire cells adjacent to non-fire cells
        from scipy.ndimage import convolve
        kernel = np.ones((3, 3))
        neighbors = convolve(fire_mask.astype(float), kernel, mode='constant')
        self.perimeter_cells = int(np.sum((fire_mask) & (neighbors < 9)))

    # ── Cost Functions for Safety Layer ──

    def get_safety_constraints(self, drone_idx: int, position: np.ndarray) -> Dict:
        """
        Compute all safety constraint values h(x) for a drone at given position.
        h(x) >= 0 means safe.
        """
        h = {}

        # Fire distance
        fire_dist = self._distance_to_nearest_fire(position)
        h['fire_distance'] = fire_dist - self.cfg.fire_damage_radius

        # Thermal plume
        ix, iy = int(np.clip(position[0], 0, self.cfg.grid_size-1)), int(np.clip(position[1], 0, self.cfg.grid_size-1))
        thermal = float(self.thermal_plume[ix, iy])
        h['thermal'] = self.cfg.thermal_damage_threshold - thermal

        # Boundary
        h['boundary_x_min'] = position[0] - self.cfg.boundary_margin
        h['boundary_x_max'] = self.cfg.grid_size - self.cfg.boundary_margin - position[0]
        h['boundary_y_min'] = position[1] - self.cfg.boundary_margin
        h['boundary_y_max'] = self.cfg.grid_size - self.cfg.boundary_margin - position[1]

        # Inter-agent separation
        for j in range(self.cfg.num_drones):
            if j != drone_idx and self.drones[j].is_active:
                dist = np.linalg.norm(position - self.drones[j].position)
                h[f'separation_{j}'] = dist - self.cfg.min_separation

        # Battery
        h['battery'] = float(self.drones[drone_idx].battery) / self.cfg.drone_battery_steps

        return h

    def get_obs_flat(self, drone_idx: int) -> np.ndarray:
        """Get flattened observation for a single drone."""
        obs = self._get_all_observations()
        return obs[drone_idx].flatten()

    @property
    def observation_space_size(self) -> int:
        """Total observation size per drone."""
        r = self.cfg.local_obs_radius
        obs_size = 2 * r + 1
        return self.cfg.obs_channels * obs_size * obs_size

    @property
    def action_space_size(self) -> int:
        """Number of discrete actions."""
        return 5 if self.cfg.discrete_actions else 2

    def render_ascii(self) -> str:
        """Render the grid as ASCII art."""
        symbols = {
            'empty': '.',
            'fire': '#',
            'burned': 'x',
            'drone': 'D',
            'fuel_low': ',',
        }
        lines = []
        for y in range(self.cfg.grid_size - 1, -1, -1):
            row = ''
            for x in range(self.cfg.grid_size):
                if self.fire_grid[x, y] > 0.5:
                    row += symbols['fire']
                elif self.fire_grid[x, y] > 0.1:
                    row += symbols['fire']
                elif self.fuel_grid[x, y] < 0.3:
                    row += symbols['fuel_low']
                else:
                    row += symbols['empty']
            # Mark drones
            for i, d in enumerate(self.drones):
                if d.is_active:
                    dx, dy = int(d.position[0]), int(d.position[1])
                    if dy == y:
                        row_list = list(row)
                        row_list[dx] = str(i) if i < 10 else '*'
                        row = ''.join(row_list)
            lines.append(row)
        return '\n'.join(lines)
