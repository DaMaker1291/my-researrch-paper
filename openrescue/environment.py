"""
OpenRescue — Simulation Environment
===================================

A Gymnasium-style multi-agent grid environment for **Graceful Autonomy Under
Infrastructure Failure**: a swarm of drones must explore an area, find points
of interest (POIs), and report them while GPS, communication and sensors are
injected with failures (Failure Levels 1–5, see ``failures.py``).

Design principles
-----------------
* **Decentralized by construction** — every quantity the policies need
  (local map, R-index, neighbors) is available per agent in ``info``; there is
  no central oracle exposed to policies.
* **Failure injection is physical** — GPS denial separates *belief* from
  *true* position; packet loss drops directed communication edges; sensor
  noise inflates the variance of the IMU reading that feeds H_sensor.
* **Energy is conserved** — every action drains battery; Information-per-Joule
  (eta_I) is a headline metric of the benchmark.

API (Gymnasium-flavored)
------------------------
    env = OpenRescueEnv(grid=24, n_drones=6, failure_level=3, seed=0)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(actions)   # actions: (K,) ints
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .failures import FailureConfig, level_config
from .resilience_index import (
    MODE_CLUSTER,
    behavior_mode, comm_quality, consensus_agreement,
    estimate_packet_reception_rate, latency_from_distance,
    resilience_index, rssi_from_distance, sensor_confidence,
)

# Map labels
UNKNOWN = 0
FREE = 1
OCCUPIED = 2

# Action deltas: stay, N, S, E, W
_ACTION_DELTAS = np.array([[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int32)


class OpenRescueEnv:
    """Multi-agent rescue exploration environment with failure injection."""

    def __init__(
        self,
        grid: int = 24,
        n_drones: int = 6,
        n_pois: int = 10,
        max_steps: int = 200,
        obstacle_density: float = 0.12,
        sense_range: int = 4,
        comm_range: float = 10.0,
        failure_level: int = 1,
        failure: Optional[FailureConfig] = None,
        battery_capacity: float = 100.0,
        energy_move: float = 0.5,
        energy_stay: float = 0.1,
        energy_to_joules: float = 5.0,
        imu_window: int = 20,
        seed: Optional[int] = None,
    ):
        self.grid = grid
        self.n_drones = n_drones
        self.n_pois = int(max(0, int(n_pois)))
        self.max_steps = max_steps
        self.obstacle_density = obstacle_density
        self.sense_range = int(sense_range)
        self.comm_range = float(comm_range)
        self.failure_level = int(np.clip(failure_level, 1, 5))
        self.failure = failure or level_config(self.failure_level, np.random.default_rng(0))
        self.battery_capacity = float(battery_capacity)
        self.energy_move = float(energy_move)
        self.energy_stay = float(energy_stay)
        self.energy_to_joules = float(energy_to_joules)
        self.imu_window = int(imu_window)
        self.seed = seed

        # Observation space: (K, obs_dim) stacked array
        self.obs_r = 2
        self.patch = 2 * self.obs_r + 1
        self.n_patch_channels = 3
        self.n_scalars = 5
        self.obs_dim = self.n_patch_channels * self.patch * self.patch + self.n_scalars
        self.act_dim = 5

        from gymnasium.spaces import Box, Discrete
        self.action_space = Discrete(self.act_dim)
        self.observation_space = Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

        self._rng = np.random.default_rng(self.seed)
        self.reset(seed=self.seed)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed = int(seed)
            self._rng = np.random.default_rng(self.seed)

        # Fresh stochastic realization of the failure level per episode
        self.failure = level_config(self.failure_level, self._rng)

        self.step_count = 0
        self._visited = np.zeros((self.grid, self.grid), dtype=bool)
        self._entropy_init = 0.0

        # Obstacles
        self.obstacles = (self._rng.random((self.grid, self.grid)) < self.obstacle_density)
        # Keep center clear for the base station
        self.obstacles[self.grid // 2, self.grid // 2] = False
        # Keep spawn corners clear
        self.obstacles[0:2, 0:2] = False
        self.obstacles[-2:, 0:2] = False
        self.obstacles[0:2, -2:] = False
        self.obstacles[-2:, -2:] = False

        # Points of interest (free cells, away from base)
        self.pois: List[np.ndarray] = []
        self.poi_found = np.zeros(self.n_pois, dtype=bool)
        attempts = 0
        while len(self.pois) < self.n_pois and attempts < 10_000:
            attempts += 1
            p = self._rng.integers(0, self.grid, size=2)
            if self.obstacles[p[1], p[0]]:
                continue
            if np.linalg.norm(p - np.array([self.grid // 2, self.grid // 2])) < 5:
                continue
            self.pois.append(p.astype(np.float32))

        # Base station at center
        self.base = np.array([self.grid // 2, self.grid // 2], dtype=np.float32)

        # Agents
        self.drones: List[Dict] = []
        spawns = self._sample_spawns()
        for i in range(self.n_drones):
            self.drones.append(self._make_agent(spawns[i]))

        # Initial map entropy (all-unknown maps), in bits
        h_cell = 2 * (0.5 * np.log(2.0))  # H(p=0.5) = ln 2 per cell per agent
        self._entropy_init = self.n_drones * (self.grid * self.grid) * h_cell / np.log(2.0)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _sample_spawns(self) -> List[np.ndarray]:
        """Distribute agents over the four corners + center as a seed swarm."""
        corners = [
            np.array([1.5, 1.5]), np.array([self.grid - 1.5, 1.5]),
            np.array([1.5, self.grid - 1.5]), np.array([self.grid - 1.5, self.grid - 1.5]),
        ]
        spawns = []
        for i in range(self.n_drones):
            if i < len(corners):
                p = corners[i].copy()
                p += self._rng.uniform(-0.5, 0.5, size=2)
            else:
                # pack remaining agents near the base
                ang = self._rng.uniform(0, 2 * np.pi)
                rad = float(self._rng.uniform(2.0, 4.0))
                p = self.base + rad * np.array([np.cos(ang), np.sin(ang)])
            spawns.append(np.clip(p, 0.5, self.grid - 0.5).astype(np.float32))
        return spawns

    def _make_agent(self, pos: np.ndarray) -> Dict:
        return {
            'pos': pos.copy(),                       # true position
            'belief': pos.copy(),                    # GPS-based position belief
            'battery': self.battery_capacity,
            'grounded': False,
            'cause': None,
            'map': np.zeros((self.grid, self.grid), dtype=np.int32),      # local map
            'map_conf': np.zeros((self.grid, self.grid), dtype=np.int32), # observation count
            'visited': set(),
            'imu_buffer': np.zeros((0, 2), dtype=np.float32),
            'energy_spent': 0.0,
            'last_base_bearing': np.zeros(2, dtype=np.float32),           # cached when GPS trusted
            'blocked': False,                        # last move attempt rejected (known obstacle)
            'comm_expected': [],                     # per-step expected messages
            'comm_received': [],                     # per-step received messages
            'neighbors_received': [],                # per-step list of neighbors whose msgs arrived
            'rssi': [],                              # per-step mean rssi
            'latency': [],                           # per-step mean latency
            'h_sensor': 0.5, 'q_comm': 0.5, 'd_consensus': 0.5,
            'r_index': 0.5, 'mode': MODE_CLUSTER,
        }

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, actions: np.ndarray):
        rng = self._rng
        self.step_count += 1
        f = self.failure

        pois_found_before = int(self.poi_found.sum())

        for i, d in enumerate(self.drones):
            if d['grounded']:
                continue

            d['blocked'] = False  # reset per-step collision flag

            # ---- 1. Sensor reading (feeds H_sensor) -------------------
            true_vel = np.array([0.0, 0.0], dtype=np.float32)
            reading = true_vel + rng.normal(0.0, f.sensor_noise_std, size=2).astype(np.float32)
            d['imu_buffer'] = np.vstack([d['imu_buffer'], reading])[-self.imu_window:]

            # ---- 2. GPS fix / denial (feeds belief divergence) --------
            if rng.random() < f.gps_denial_prob:
                pass  # denied: belief stays stale while the drone moves
            else:
                d['belief'] = d['pos'] + rng.normal(0.0, f.gps_drift_std, size=2).astype(np.float32)
                # cache base bearing while navigation is trustworthy
                to_base = self.base - d['belief']
                nrm = float(np.linalg.norm(to_base))
                if nrm > 1e-6:
                    d['last_base_bearing'] = to_base / nrm

            # ---- 3. Communication graph (physical, on true positions) --
            comm_range = self.comm_range * f.comm_range_scale
            neighbors = [j for j in range(self.n_drones)
                         if j != i and np.linalg.norm(self.drones[j]['pos'] - d['pos']) <= comm_range]

            expected = len(neighbors)
            received = 0
            rssi_vals, lat_vals, received_idx = [], [], []
            for j in neighbors:
                dist = float(np.linalg.norm(self.drones[j]['pos'] - d['pos']))
                rssi_vals.append(rssi_from_distance(dist, comm_range))
                lat_vals.append(latency_from_distance(dist))
                if rng.random() >= f.packet_loss_prob:
                    received += 1
                    received_idx.append(j)
            d['comm_expected'].append(expected)
            d['comm_received'].append(received)
            d['neighbors_received'].append(received_idx)
            d['rssi'].append(float(np.mean(rssi_vals)) if rssi_vals else 0.0)
            d['latency'].append(float(np.mean(lat_vals)) if lat_vals else 0.0)
            # keep sliding windows
            for key in ('comm_expected', 'comm_received', 'rssi', 'latency'):
                d[key] = d[key][-self.imu_window:]
            d['neighbors_received'] = d['neighbors_received'][-self.imu_window:]

            # ---- 4. Sensing: write cells into the local map -----------
            self._sense(d, rng)

            # ---- 5. Map merging from received neighbor messages -------
            for j in received_idx:
                self._merge_map(d, self.drones[j])

            # ---- 6. Compute R-index components -------------------------
            self._update_resilience(i, d, received_idx)

            # ---- 7. Execute action (navigation on BELIEF) --------------
            a = int(actions[i])
            delta = _ACTION_DELTAS[a]
            energy = self.energy_move if a != 0 else self.energy_stay
            d['battery'] -= energy
            d['energy_spent'] += energy

            if d['battery'] <= 0:
                d['battery'] = 0.0
                d['grounded'] = True
                d['cause'] = 'battery'
                continue

            target = d['belief'] + delta.astype(np.float32)
            new_pos = np.clip(target, 0.5, self.grid - 0.5)
            gx, gy = int(round(new_pos[0])), int(round(new_pos[1]))
            gx, gy = int(np.clip(gx, 0, self.grid - 1)), int(np.clip(gy, 0, self.grid - 1))

            if self.obstacles[gy, gx]:
                if d['map'][gy, gx] == OCCUPIED:
                    # Obstacle is known to the local map: the drone sees it and
                    # is safely blocked (no crash). The policy reacts via the
                    # 'blocked' flag in info.
                    d['blocked'] = True
                else:
                    # BLIND contact: the cell was never sensed or was corrupted
                    # (free label on a true obstacle) — only possible under
                    # sensor/comm/GPS failure. Collision probability scales
                    # with sensing noise and belief-vs-true divergence.
                    belief_err = min(float(np.linalg.norm(d['belief'] - d['pos'])), 3.0)
                    p_blind = min(0.9, 0.05 + 1.0 * f.sensor_noise_std + 0.35 * belief_err)
                    if rng.random() < p_blind:
                        d['grounded'] = True
                        d['cause'] = 'crash'
                    else:
                        d['blocked'] = True
                continue
            d['pos'] = new_pos.astype(np.float32)
            if not self._visited[gy, gx]:
                self._visited[gy, gx] = True
                d['visited'].add((gx, gy))

            # ---- 8. POI discovery (physical sensing) ------------------
            for k, p in enumerate(self.pois):
                if not self.poi_found[k] and np.linalg.norm(d['pos'] - p) <= 1.5:
                    self.poi_found[k] = True

        pois_found_now = int(self.poi_found.sum())
        pois_delta = pois_found_now - pois_found_before

        obs = self._get_obs()
        rewards = self._rewards(pois_delta)
        terminated = np.zeros(self.n_drones, dtype=bool)
        if self.n_pois > 0 and pois_found_now == self.n_pois:
            terminated[:] = True
        truncated = np.full(self.n_drones, self.step_count >= self.max_steps, dtype=bool)
        info = self._get_info()

        return obs, rewards, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _sense(self, d: Dict, rng: np.random.Generator):
        """Write sensed cells into the agent's local map (with corruption)."""
        r = self.sense_range
        cx, cy = int(round(d['pos'][0])), int(round(d['pos'][1]))
        x0, x1 = max(0, cx - r), min(self.grid, cx + r + 1)
        y0, y1 = max(0, cy - r), min(self.grid, cy + r + 1)
        corrupt = rng.random() < self.failure.map_corrupt_prob

        for gy in range(y0, y1):
            for gx in range(x0, x1):
                if np.hypot(gx - cx, gy - cy) > r:
                    continue
                label = OCCUPIED if self.obstacles[gy, gx] else FREE
                if corrupt:
                    label = FREE if label == OCCUPIED else OCCUPIED
                d['map_conf'][gy, gx] += 1
                d['map'][gy, gx] = label

    @staticmethod
    def _merge_map(dst: Dict, src: Dict):
        """Merge src's map into dst's where src has higher confidence."""
        m = dst['map_conf'] < src['map_conf']
        dst['map_conf'][m] = src['map_conf'][m]
        dst['map'][m] = src['map'][m]

    def _update_resilience(self, i: int, d: Dict, received_idx: List[int]):
        d['h_sensor'] = sensor_confidence(d['imu_buffer'])

        prr = estimate_packet_reception_rate(d['comm_expected'][-self.imu_window:],
                                             d['comm_received'][-self.imu_window:])
        d['q_comm'] = comm_quality(prr, d['rssi'][-self.imu_window:], d['latency'][-self.imu_window:])

        maps = np.stack([a['map'] for a in self.drones])
        d['d_consensus'] = consensus_agreement(maps, received_idx, i)

        d['r_index'] = resilience_index(d['h_sensor'], d['q_comm'], d['d_consensus'])
        d['mode'] = behavior_mode(d['r_index'])

    def _rewards(self, pois_delta: int) -> np.ndarray:
        r = np.zeros(self.n_drones, dtype=np.float32)
        if pois_delta > 0:
            r += 10.0 * pois_delta / self.n_drones
        # shared exploration reward scaled by swarm progress
        coverage = self.coverage
        r += 0.02 * (1.0 - coverage)
        for i, d in enumerate(self.drones):
            r[i] -= 0.01 * (self.energy_move if d['battery'] < self.battery_capacity else 0.0)
        return r

    # ------------------------------------------------------------------
    # Observations / info
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.n_drones, self.obs_dim), dtype=np.float32)
        r = self.obs_r
        p = self.patch
        ch = p * p
        for i, d in enumerate(self.drones):
            cx, cy = int(round(d['belief'][0])), int(round(d['belief'][1]))
            x0, x1 = max(0, cx - r), min(self.grid, cx + r + 1)
            y0, y1 = max(0, cy - r), min(self.grid, cy + r + 1)
            h, w = y1 - y0, x1 - x0

            def pad(arr):
                out = np.zeros((p, p), dtype=np.float32)
                out[:h, :w] = arr
                return out

            map_patch = pad(d['map'][y0:y1, x0:x1].astype(np.float32) / 2.0)
            vis_patch = pad(self._visited[y0:y1, x0:x1].astype(np.float32))
            neigh_patch = np.zeros((p, p), dtype=np.float32)
            for j, dj in enumerate(self.drones):
                if j != i:
                    jx, jy = int(round(dj['pos'][0])) - cx + r, int(round(dj['pos'][1])) - cy + r
                    if 0 <= jx < p and 0 <= jy < p:
                        neigh_patch[jy, jx] = 1.0

            to_base = self.base - d['belief']
            nrm = float(np.linalg.norm(to_base)) or 1.0
            bearing = to_base / nrm
            scalars = np.array([
                d['r_index'], d['battery'] / self.battery_capacity,
                (bearing[0] + 1.0) / 2.0, (bearing[1] + 1.0) / 2.0,
                self.coverage,
            ], dtype=np.float32)

            obs[i] = np.concatenate([
                map_patch.ravel(), vis_patch.ravel(), neigh_patch.ravel(), scalars,
            ])
        return obs

    def _get_info(self) -> dict:
        return {
            'agents': [
                {
                    'id': i,
                    'pos': d['pos'].copy(),
                    'belief': d['belief'].copy(),
                    'battery': d['battery'],
                    'grounded': d['grounded'],
                    'cause': d.get('cause'),
                    'map': d['map'].copy(),
                    'map_conf': d['map_conf'].copy(),
                    'r_index': d['r_index'],
                    'mode': d['mode'],
                    'h_sensor': d['h_sensor'],
                    'q_comm': d['q_comm'],
                    'd_consensus': d['d_consensus'],
                    'neighbors_received': list(d['neighbors_received'][-1]) if d['neighbors_received'] else [],
                    'blocked': d['blocked'],
                    'last_base_bearing': d['last_base_bearing'].copy(),
                    'energy_spent': d['energy_spent'],
                }
                for i, d in enumerate(self.drones)
            ],
            'failure': {
                'level': self.failure.level,
                'name': self.failure.name,
                'config': self.failure,
                'description': self.failure.__repr__(),
            },
            'step': self.step_count,
            'coverage': self.coverage,
            'pois_found': int(self.poi_found.sum()),
            'n_pois': self.n_pois,
            'mean_r': self.mean_r,
            'energy_joules': self.energy_joules,
            'info_gain_bits': self.info_gain_bits,
            'eta_i': self.eta_i,
            'all_pois_found': bool(self.n_pois > 0 and self.poi_found.all()),
        }

    # ------------------------------------------------------------------
    # Metrics (episode-level)
    # ------------------------------------------------------------------
    @property
    def free_cells(self) -> int:
        return int((~self.obstacles).sum())

    @property
    def coverage(self) -> float:
        return float(self._visited.sum() / max(self.free_cells, 1))

    @property
    def mean_r(self) -> float:
        return float(np.mean([d['r_index'] for d in self.drones]))

    @property
    def agents_alive(self) -> float:
        return float(np.mean([not d['grounded'] for d in self.drones]))

    @property
    def agents_lost(self) -> int:
        return int(sum(1 for d in self.drones if d.get('cause') == 'crash'))

    @property
    def energy_joules(self) -> float:
        return float(sum(d['energy_spent'] for d in self.drones) * self.energy_to_joules)

    def _map_entropy_bits(self, drones: List[Dict]) -> float:
        """Total Bernoulli occupancy entropy of the given agents' maps, in bits."""
        h_total = 0.0
        for d in drones:
            p = np.where(d['map'] == OCCUPIED, 0.95,
                         np.where(d['map'] == FREE, 0.05, 0.5))
            p = np.clip(p, 1e-9, 1 - 1e-9)
            h_total += float(np.sum(-p * np.log(p) - (1 - p) * np.log(1 - p))) / np.log(2.0)
        return h_total

    @property
    def info_gain_bits(self) -> float:
        """Map entropy reduction + POI discovery bonus, in bits."""
        h_total = self._map_entropy_bits(self.drones)
        reduction = self._entropy_init - h_total
        poi_bonus = np.log2(1.0 + int(self.poi_found.sum()))
        return float(max(0.0, reduction) + poi_bonus)

    @property
    def eta_i(self) -> float:
        """Information-per-Joule: bits of map/POI information per Joule spent."""
        e = self.energy_joules
        if e <= 1e-9:
            return 0.0
        return float(self.info_gain_bits / e)

    @property
    def eta_i_survivor(self) -> float:
        """Survivor-normalized Information-per-Joule.

        Recomputes information from the maps of surviving agents only, against
        an initial-entropy baseline scaled to that survivor count, and divides
        by the energy those survivors spent (grounded agents stop spending
        energy, so the plain ``eta_i`` is inflated by dead drones that banked
        early exploration but no longer drain battery). This is the
        survivor-selection-corrected variant recommended in the paper.
        """
        alive = [d for d in self.drones if not d['grounded']]
        if not alive:
            return 0.0
        h_total = self._map_entropy_bits(alive)
        # per-agent all-unknown baseline is grid^2 bits (see reset())
        init = len(alive) * (self.grid * self.grid)
        reduction = max(0.0, init - h_total)
        info = reduction + np.log2(1.0 + int(self.poi_found.sum()))
        e = float(sum(d['energy_spent'] for d in alive)) * self.energy_to_joules
        if e <= 1e-9:
            return 0.0
        return float(info / e)

    def episode_summary(self) -> dict:
        """Compact per-episode metric dict (used by the benchmark)."""
        return {
            'coverage': self.coverage,
            'pois_found': int(self.poi_found.sum()),
            'pois_ratio': float(self.poi_found.sum() / max(self.n_pois, 1)),
            'survival': self.agents_alive,
            'lost': self.agents_lost,
            'mean_r': self.mean_r,
            'info_gain_bits': self.info_gain_bits,
            'energy_joules': self.energy_joules,
            'eta_i': self.eta_i,
            'eta_i_surv': self.eta_i_survivor,
            'steps': self.step_count,
        }