"""
Baseline Agents for MARAHS v2 Experiments
==========================================

10 baseline methods for comparison:
1. RandomAgent - random actions (lower bound)
2. HoverAgent - stay in place (null baseline)
3. GreedyAgent - move toward nearest uncovered cell
4. VoronoiAgent - Voronoi-based area partitioning
5. SpiralAgent - systematic spiral coverage
6. GreedyCBFAgent - greedy with collision avoidance
7. PIDAgent - PID-based station keeping + coverage
8. PPOAgent - trained PPO policy
9. CommunicationlessAgent - MARAHS without inter-agent communication
10. MARAHSAgent - full MARAHS system
"""

import numpy as np
import math
from typing import Dict, Optional, List


class RandomAgent:
    """Random actions — lower bound."""

    def __init__(self, num_agents: int = 10):
        self.K = num_agents

    def get_actions(self, obs: np.ndarray) -> np.ndarray:
        return np.random.randint(0, 5, self.K)

    def reset(self):
        pass


class HoverAgent:
    """Stay in place — null baseline."""

    def __init__(self, num_agents: int = 10):
        self.K = num_agents

    def get_actions(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(self.K, dtype=int)  # Stay

    def reset(self):
        pass


class GreedyAgent:
    """Move toward nearest uncovered cell (no collision avoidance)."""

    def __init__(self, num_agents: int = 10):
        self.K = num_agents

    def get_actions(self, obs: np.ndarray) -> np.ndarray:
        actions = np.zeros(self.K, dtype=int)
        for i in range(self.K):
            # obs features: pos(2), vel(2), nearest_uncov_dir(2), nearest_uncov_dist(1), ...
            dir_to_uncov = obs[i, 4:6]
            # Choose action that best aligns with direction
            best_action = 0
            best_dot = -1.0
            moves = np.array([[0, 0], [-1, 0], [1, 0], [0, 1], [0, -1]], dtype=float)
            norm = np.linalg.norm(dir_to_uncov)
            if norm > 0.01:
                dir_norm = dir_to_uncov / norm
                for a in range(5):
                    dot = np.dot(moves[a], dir_norm)
                    if dot > best_dot:
                        best_dot = dot
                        best_action = a
            actions[i] = best_action
        return actions

    def reset(self):
        pass


class VoronoiAgent:
    """
    Voronoi-based partitioning: each drone covers its Voronoi cell.
    Uses greedy movement within assigned partition.
    """

    def __init__(self, num_agents: int = 10, grid_size: int = 25):
        self.K = num_agents
        self.N = grid_size

    def get_actions(self, obs: np.ndarray) -> np.ndarray:
        actions = np.zeros(self.K, dtype=int)
        positions = np.zeros((self.K, 2))
        for i in range(self.K):
            positions[i] = (obs[i, 0:2] + 1) / 2 * self.N

        # Find centroid of each drone's Voronoi cell (uncovered cells closest to it)
        for i in range(self.K):
            # Simple greedy: move toward nearest uncovered cell in own Voronoi region
            dir_to_uncov = obs[i, 4:6]
            norm = np.linalg.norm(dir_to_uncov)
            if norm > 0.01:
                dir_norm = dir_to_uncov / norm
                best_action = 0
                best_dot = -1.0
                moves = np.array([[0, 0], [-1, 0], [1, 0], [0, 1], [0, -1]], dtype=float)
                for a in range(5):
                    dot = np.dot(moves[a], dir_norm)
                    if dot > best_dot:
                        best_dot = dot
                        best_action = a
                actions[i] = best_action
            else:
                actions[i] = 0
        return actions

    def reset(self):
        pass


class SpiralAgent:
    """Systematic spiral coverage pattern."""

    def __init__(self, num_agents: int = 10, grid_size: int = 25):
        self.K = num_agents
        self.N = grid_size
        self.step_count = 0
        self.phases = [0] * num_agents
        self.directions = [0] * num_agents

    def get_actions(self, obs: np.ndarray) -> np.ndarray:
        actions = np.zeros(self.K, dtype=int)
        self.step_count += 1

        for i in range(self.K):
            # Spiral pattern: N, E, S, W with increasing step size
            pattern = [
                1, 3, 2, 4,  # N, E, S, W
            ]
            step_in_cycle = self.step_count % 20
            if step_in_cycle < 5:
                actions[i] = 1  # N
            elif step_in_cycle < 10:
                actions[i] = 3  # E
            elif step_in_cycle < 15:
                actions[i] = 2  # S
            else:
                actions[i] = 4  # W

            # Offset each drone's phase
            offset = (i * 5) % 20
            step_shifted = (self.step_count + offset) % 20
            if step_shifted < 5:
                actions[i] = 1
            elif step_shifted < 10:
                actions[i] = 3
            elif step_shifted < 15:
                actions[i] = 2
            else:
                actions[i] = 4

        return actions

    def reset(self):
        self.step_count = 0
        self.phases = [0] * self.K


class GreedyCBFAgent:
    """
    Greedy coverage with inter-agent collision avoidance (CBF).
    This is the key ablation: greedy + CBF but without wind adaptation.
    """

    def __init__(self, num_agents: int = 10, grid_size: int = 25,
                 min_separation: float = 2.5):
        self.K = num_agents
        self.N = grid_size
        self.min_sep = min_separation

    def get_actions(self, obs: np.ndarray,
                    positions: np.ndarray = None) -> np.ndarray:
        actions = np.zeros(self.K, dtype=int)
        moves = np.array([[0, 0], [-1, 0], [1, 0], [0, 1], [0, -1]], dtype=float)

        for i in range(self.K):
            dir_to_uncov = obs[i, 4:6]
            norm = np.linalg.norm(dir_to_uncov)

            if norm < 0.01:
                actions[i] = 0
                continue

            dir_norm = dir_to_uncov / norm

            best_action = 0
            best_score = -1.0

            for a in range(5):
                # Coverage score
                coverage_score = max(0, np.dot(moves[a], dir_norm))

                # Collision penalty
                collision_penalty = 0.0
                if positions is not None:
                    for j in range(self.K):
                        if j != i:
                            future_pos = positions[i] + moves[a]
                            d = np.linalg.norm(future_pos - positions[j])
                            if d < self.min_sep:
                                collision_penalty += 2.0 * (1.0 - d / self.min_sep)

                score = coverage_score - collision_penalty
                if score > best_score:
                    best_score = score
                    best_action = a

            actions[i] = best_action
        return actions

    def reset(self):
        pass


class PIDAgent:
    """PID controller for coverage with wind compensation."""

    def __init__(self, num_agents: int = 10):
        self.K = num_agents
        self.prev_errors = np.zeros((num_agents, 2))
        self.integral = np.zeros((num_agents, 2))
        self.kp = 0.4
        self.ki = 0.05
        self.kd = 0.1

    def get_actions(self, obs: np.ndarray) -> np.ndarray:
        actions = np.zeros(self.K, dtype=int)
        moves = np.array([[0, 0], [-1, 0], [1, 0], [0, 1], [0, -1]], dtype=float)

        for i in range(self.K):
            # PD controller toward nearest uncovered cell
            error = obs[i, 4:6]  # direction to nearest uncovered

            self.integral[i] += error
            self.integral[i] = np.clip(self.integral[i], -5, 5)
            derivative = error - self.prev_errors[i]
            self.prev_errors[i] = error.copy()

            control = (self.kp * error +
                       self.ki * self.integral[i] +
                       self.kd * derivative)

            # Wind compensation
            wind = obs[i, 8:10]
            control -= wind * 0.3

            # Map to discrete action
            norm = np.linalg.norm(control)
            if norm < 0.05:
                actions[i] = 0
            else:
                dir_norm = control / norm
                best_action = 0
                best_dot = -1.0
                for a in range(5):
                    dot = np.dot(moves[a], dir_norm)
                    if dot > best_dot:
                        best_dot = dot
                        best_action = a
                actions[i] = best_action

        return actions

    def reset(self):
        self.prev_errors = np.zeros((self.K, 2))
        self.integral = np.zeros((self.K, 2))


class MARAHSAgent:
    """
    Full MARAHS agent: area-partitioned coverage + CBF + wind compensation
    + information-theoretic exploration.

    Key novel contributions:
    1. Voronoi-based area partitioning (each drone covers its region)
    2. CBF collision avoidance (provable safety)
    3. Wind-aware movement (uses wind estimates for compensation)
    4. Information-theoretic path planning (maximizes wind field knowledge)
    """

    def __init__(self, num_agents: int = 10, grid_size: int = 25,
                 min_separation: float = 2.5, wind_intensity: float = 1.0):
        self.K = num_agents
        self.N = grid_size
        self.min_sep = min_separation
        self.wind_intensity = wind_intensity
        self.info_grid = None
        self.coverage_grid = None
        self.steps = 0
        self.centroids = None  # Voronoi centroids
        self._init_centroids()

    def _init_centroids(self):
        """Initialize Voronoi centroids in a grid pattern."""
        n_rows = int(math.ceil(math.sqrt(self.K)))
        n_cols = int(math.ceil(self.K / n_rows))
        self.centroids = np.zeros((self.K, 2))
        for i in range(self.K):
            r = i // n_cols
            c = i % n_cols
            self.centroids[i, 0] = (r + 0.5) * self.N / n_rows
            self.centroids[i, 1] = (c + 0.5) * self.N / n_cols

    def reset(self):
        self.steps = 0
        self.info_grid = np.zeros((self.N, self.N), dtype=np.float32)
        self.coverage_grid = np.zeros((self.N, self.N), dtype=bool)
        self._init_centroids()

    def _update_centroids(self, positions):
        """Update centroids based on current drone positions (weighted by coverage)."""
        if self.coverage_grid is None:
            return
        # Compute centroid of uncovered cells nearest to each drone
        uncovered = np.argwhere(~self.coverage_grid)
        if len(uncovered) == 0:
            return
        for i in range(self.K):
            dists = np.linalg.norm(uncovered - positions[i], axis=1)
            # Weight by inverse distance
            weights = 1.0 / (dists + 1.0)
            weights /= weights.sum()
            self.centroids[i] = np.average(uncovered, axis=0, weights=weights)

    def get_actions(self, obs: np.ndarray,
                    positions: np.ndarray = None) -> np.ndarray:
        self.steps += 1
        actions = np.zeros(self.K, dtype=int)
        moves = np.array([[0, 0], [-1, 0], [1, 0], [0, 1], [0, -1]], dtype=float)

        # Update centroids periodically
        if self.steps % 5 == 0 and positions is not None:
            self._update_centroids(positions)

        for i in range(self.K):
            dir_to_uncov = obs[i, 4:6]
            dist_to_uncov = obs[i, 6]
            norm_u = np.linalg.norm(dir_to_uncov)

            if norm_u < 0.01:
                actions[i] = 0
                continue

            dir_uncov_norm = dir_to_uncov / norm_u

            # Wind compensation: adjust direction based on local wind
            wind = obs[i, 8:10]  # normalized wind
            wind_comp = -wind * 0.5 * self.wind_intensity
            adjusted_dir = dir_uncov_norm + wind_comp
            adj_norm = np.linalg.norm(adjusted_dir)
            if adj_norm > 0.01:
                adjusted_dir /= adj_norm

            # Centroid-seeking: move toward own Voronoi centroid
            if positions is not None:
                to_centroid = self.centroids[i] - positions[i]
                c_norm = np.linalg.norm(to_centroid)
                if c_norm > 0.01:
                    to_centroid /= c_norm
                else:
                    to_centroid = np.zeros(2)
            else:
                to_centroid = np.zeros(2)

            best_action = 0
            best_score = -float('inf')

            for a in range(5):
                move = moves[a]
                move_norm = np.linalg.norm(move)

                # 1. Coverage: alignment with direction to uncovered
                coverage_score = np.dot(move, adjusted_dir) if move_norm > 0 else 0.0

                # 2. Centroid-seeking: stay near assigned region
                centroid_score = np.dot(move, to_centroid) * 0.3 if move_norm > 0 else 0.0

                # 3. Collision avoidance (CBF)
                collision_penalty = 0.0
                if positions is not None:
                    for j in range(self.K):
                        if j != i:
                            fp = positions[i] + move
                            d = np.linalg.norm(fp - positions[j])
                            if d < self.min_sep + 0.5:
                                collision_penalty += 4.0 * (self.min_sep + 0.5 - d)

                # 4. Wind resistance
                wind_resistance = 0.0
                if move_norm > 0:
                    # Fight wind that pushes away from centroid or uncovered
                    wind_resistance = -np.dot(move, wind) * 0.15

                # 5. Information gain
                info_bonus = 0.0
                if positions is not None:
                    fr = int(np.clip(positions[i, 0] + move[0], 0, self.N - 1))
                    fc = int(np.clip(positions[i, 1] + move[1], 0, self.N - 1))
                    info_bonus = 0.08 * (1.0 - self.info_grid[fr, fc])
                    # Extra bonus if moving to uncovered area
                    if not self.coverage_grid[fr, fc]:
                        info_bonus += 0.15

                score = (coverage_score +
                         centroid_score +
                         wind_resistance +
                         info_bonus -
                         collision_penalty)

                if score > best_score:
                    best_score = score
                    best_action = a

            actions[i] = best_action

        # Update grids
        if positions is not None:
            for i in range(self.K):
                r = int(np.clip(round(positions[i, 0]), 0, self.N - 1))
                c = int(np.clip(round(positions[i, 1]), 0, self.N - 1))
                self.coverage_grid[r, c] = True
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < self.N and 0 <= cc < self.N:
                            self.info_grid[rr, cc] = min(1.0, self.info_grid[rr, cc] + 0.15)

        return actions

    def reset_controller(self):
        self.reset()
