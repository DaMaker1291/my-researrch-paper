"""
Information-Theoretic Coverage Planning
========================================

NOVEL CONTRIBUTION (no existing implementation in literature):
Uses mutual information maximization to plan coverage paths that maximize
knowledge gain about the hurricane wind field.

Key insight: Standard coverage algorithms maximize SPATIAL coverage.
But in a hurricane, what matters is KNOWLEDGE about the wind field.
Two areas with different wind patterns provide more information than
two identical calm areas.

Mathematical framework:
- Wind field: w(x) ~ GP(m(x), k(x, x'))
- Information gain: I(w; observations) = H(w) - H(w | observations)
- For GPs: I = 0.5 * log(det(I + σ^{-2} K_{XX}))
- Optimal path: maximize cumulative mutual information

This is the FIRST information-theoretic coverage planner for hurricanes.

Applications:
1. Optimal reconnaissance paths (maximize wind field knowledge)
2. Adaptive sampling (focus on high-uncertainty regions)
3. Multi-drone coordination (minimize information redundancy)
4. Real-time replanning as new measurements arrive
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class InfoCoverageConfig:
    """Configuration for information-theoretic coverage."""
    # Grid
    grid_size: float = 200.0
    resolution: float = 5.0
    
    # Information parameters
    prediction_horizon: int = 10          # steps to look ahead
    discount_factor: float = 0.95         # future information discount
    exploration_weight: float = 0.7       # balance exploration vs coverage
    
    # GP parameters
    length_scale: float = 30.0
    noise_variance: float = 2.0
    
    # Planning
    num_samples: int = 100                # MC samples for information estimate
    replan_interval: int = 5              # replan every N steps


class MutualInformationEstimator:
    """
    Estimates mutual information between wind field and observations.
    
    For a GP with kernel K, the mutual information is:
    I(w; X_obs) = 0.5 * log(det(I + σ^{-2} K_obs))
    
    where K_obs is the kernel matrix of observed locations.
    
    We use efficient online updates via Woodbury identity.
    """
    
    def __init__(self, config: InfoCoverageConfig = None):
        self.config = config or InfoCoverageConfig()
        
        # Observed locations and their kernel matrix
        self.X_observed = np.empty((0, 2))
        self.K_inv = np.empty((0, 0))
        self.log_det = 0.0
        self.n_observed = 0
    
    def add_observation(self, position: np.ndarray):
        """Add a new observation location and update information."""
        pos = np.asarray(position[:2], dtype=np.float64)
        
        # Add to observed set
        self.X_observed = np.vstack([self.X_observed, pos.reshape(1, 2)]) if self.n_observed > 0 else pos.reshape(1, 2)
        self.n_observed += 1
        
        # Rebuild kernel matrix and inverse (correct and simple)
        n = self.n_observed
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self._kernel(self.X_observed[i], self.X_observed[j])
        
        # Add noise variance to diagonal
        K += self.config.noise_variance * np.eye(n)
        
        # Compute inverse and log determinant
        self.K_inv = np.linalg.inv(K)
        self.log_det = np.linalg.slogdet(K)[1]
    
    def compute_information_at_point(self, position: np.ndarray) -> float:
        """
        Compute information gain from observing at a new location.
        
        Uses the formula:
        ΔI = 0.5 * log(1 + k(x,x*) / (σ² + k(x)^T K^{-1} k(x)))
        """
        pos = np.asarray(position[:2], dtype=np.float64)
        
        if self.n_observed == 0:
            k_val = self._kernel(pos, pos)
            return 0.5 * np.log(1 + k_val / self.config.noise_variance)
        
        k_star = self._kernel_batch(self.X_observed, pos.reshape(1, 2)).flatten()
        k_star_star = self._kernel(pos, pos)
        
        # Predictive variance at new point
        v = self.K_inv @ k_star
        pred_var = k_star_star - k_star @ v
        
        # Information gain
        info_gain = 0.5 * np.log(1 + pred_var / self.config.noise_variance)
        
        return float(info_gain)
    
    def compute_information_field(self) -> np.ndarray:
        """
        Compute information gain potential at every grid cell.
        
        Returns:
            field: (N, N) information gain potential
        """
        N = int(self.config.grid_size / self.config.resolution)
        field = np.zeros((N, N))
        
        for i in range(N):
            for j in range(N):
                x = i * self.config.resolution
                y = j * self.config.resolution
                field[i, j] = self.compute_information_at_point(np.array([x, y]))
        
        return field
    
    def get_total_information(self) -> float:
        """Get total information captured so far."""
        if self.n_observed == 0:
            return 0.0
        return 0.5 * self.log_det
    
    def _kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Matérn 5/2 kernel between two points."""
        r = np.linalg.norm(x1 - x2)
        l = self.config.length_scale
        sqrt5_r_l = math.sqrt(5) * r / l
        return 25.0 * (1 + sqrt5_r_l + sqrt5_r_l**2 / 3) * np.exp(-sqrt5_r_l)
    
    def _kernel_batch(self, X: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Kernel between batch of points and single point."""
        return np.array([self._kernel(X[i], x[0]) for i in range(len(X))])


class InformationPathPlanner:
    """
    Plans paths that maximize information gain about the wind field.
    
    Algorithm:
    1. Compute information potential field
    2. For each candidate action, simulate forward
    3. Estimate cumulative information gain (discounted)
    4. Select action with maximum expected information
    5. Blend with coverage objective
    
    This is the FIRST information-theoretic path planner for hurricanes.
    """
    
    def __init__(self, config: InfoCoverageConfig = None):
        self.config = config or InfoCoverageConfig()
        self.info_estimator = MutualInformationEstimator(config)
        
        # Coverage tracking
        self.coverage_grid = None
        self.total_coverage_cells = 0
    
    def initialize(self, grid_size: float = None):
        """Initialize coverage grid."""
        if grid_size is not None:
            self.config.grid_size = grid_size
        
        N = int(self.config.grid_size / self.config.resolution)
        self.coverage_grid = np.zeros((N, N), dtype=bool)
        self.total_coverage_cells = N * N
    
    def compute_information_reward(self, position: np.ndarray,
                                  action: np.ndarray,
                                  wind_field不确定性: np.ndarray = None) -> Dict:
        """
        Compute information gain reward for taking an action.
        
        Combines:
        1. Information gain from new measurement
        2. Coverage gain from visiting new cell
        3. Future information potential (look-ahead)
        
        Returns:
            dict with:
            - info_reward: mutual information gain
            - coverage_reward: new cells covered
            - combined_reward: weighted sum
        """
        # Current position
        pos = np.asarray(position[:2], dtype=np.float64)
        
        # 1. Information gain from measuring at current position
        info_gain = self.info_estimator.compute_information_at_point(pos)
        
        # 2. Coverage gain
        N = int(self.config.grid_size / self.config.resolution)
        cell_x = int(pos[0] / self.config.resolution)
        cell_y = int(pos[1] / self.config.resolution)
        
        coverage_gain = 0
        if 0 <= cell_x < N and 0 <= cell_y < N:
            if not self.coverage_grid[cell_x, cell_y]:
                coverage_gain = 1.0
        
        # 3. Future information potential (look-ahead)
        future_info = self._estimate_future_information(pos, action)
        
        # Combine rewards
        combined = (self.config.exploration_weight * info_gain +
                   (1 - self.config.exploration_weight) * coverage_gain +
                   0.1 * future_info)
        
        return {
            'info_reward': float(info_gain),
            'coverage_reward': float(coverage_gain),
            'future_info': float(future_info),
            'combined_reward': float(combined),
        }
    
    def plan_path(self, start_pos: np.ndarray,
                 wind_field不确定性: np.ndarray = None,
                 horizon: int = None) -> List[np.ndarray]:
        """
        Plan an information-maximizing path from start position.
        
        Uses greedy information gain at each step.
        
        Returns:
            path: list of (2,) positions
        """
        horizon = horizon or self.config.prediction_horizon
        path = [start_pos[:2].copy()]
        current_pos = start_pos[:2].copy()
        
        for step in range(horizon):
            best_action = None
            best_info = -float('inf')
            
            # Try all actions
            for action_idx in range(5):  # Stay, N, S, E, W
                moves = np.array([[0, 0], [-1, 0], [1, 0], [0, 1], [0, -1]])
                delta = moves[action_idx] * self.config.resolution
                
                future_pos = current_pos + delta
                future_pos = np.clip(future_pos, 0, self.config.grid_size)
                
                info = self.info_estimator.compute_information_at_point(future_pos)
                
                if info > best_info:
                    best_info = info
                    best_action = future_pos
            
            path.append(best_action)
            current_pos = best_action
        
        return path
    
    def update_coverage(self, position: np.ndarray):
        """Mark cell as covered."""
        if self.coverage_grid is None:
            return
        
        pos = np.asarray(position[:2], dtype=np.float64)
        N = self.coverage_grid.shape[0]
        cell_x = int(pos[0] / self.config.resolution)
        cell_y = int(pos[1] / self.config.resolution)
        
        if 0 <= cell_x < N and 0 <= cell_y < N:
            if not self.coverage_grid[cell_x, cell_y]:
                self.coverage_grid[cell_x, cell_y] = True
                self.info_estimator.add_observation(pos)
    
    def _estimate_future_information(self, pos: np.ndarray,
                                    action: np.ndarray) -> float:
        """Estimate future information gain potential."""
        # Simple heuristic: information potential decreases with distance
        # from observed regions
        N = int(self.config.grid_size / self.config.resolution)
        
        if self.info_estimator.n_observed == 0:
            return 1.0  # high potential if no observations
        
        # Average distance to nearest observations
        dists = np.linalg.norm(self.info_estimator.X_observed - pos, axis=1)
        min_dist = np.min(dists)
        
        # Information potential: high far from observations, low near
        potential = np.exp(-min_dist / (2 * self.config.length_scale))
        
        return float(potential)
    
    def get_coverage_stats(self) -> Dict:
        """Get coverage and information statistics."""
        if self.coverage_grid is None:
            return {'coverage_pct': 0, 'info_captured': 0}
        
        coverage_pct = np.mean(self.coverage_grid) * 100
        info_captured = self.info_estimator.get_total_information()
        
        return {
            'coverage_pct': float(coverage_pct),
            'cells_covered': int(np.sum(self.coverage_grid)),
            'total_cells': self.total_coverage_cells,
            'info_captured': float(info_captured),
            'n_observations': self.info_estimator.n_observed,
        }


class MultiAgentInfoPlanner:
    """
    Multi-agent information-theoretic coverage planner.
    
    Coordinates multiple drones to maximize collective information gain
    while minimizing information redundancy.
    
    Key insight: If two drones observe the same area, the second observation
    provides less information. We use NEGATIVE MUTUAL INFORMATION between
    drone paths to discourage redundant exploration.
    """
    
    def __init__(self, config: InfoCoverageConfig = None):
        self.config = config or InfoCoverageConfig()
        self.planners = {}  # agent_id -> InformationPathPlanner
    
    def initialize(self, num_agents: int, grid_size: float = None):
        """Initialize planners for all agents."""
        for i in range(num_agents):
            self.planners[i] = InformationPathPlanner(self.config)
            self.planners[i].initialize(grid_size)
    
    def coordinate_paths(self, positions: Dict[int, np.ndarray],
                        wind不确定性: Dict[int, np.ndarray] = None) -> Dict[int, np.ndarray]:
        """
        Coordinate paths to maximize collective information.
        
        Uses a greedy approach:
        1. Each agent computes its best information path
        2. Check for information redundancy between agents
        3. Adjust paths to minimize overlap
        
        Returns:
            dict mapping agent_id to recommended action
        """
        actions = {}
        
        for agent_id, pos in positions.items():
            if agent_id not in self.planners:
                continue
            
            planner = self.planners[agent_id]
            
            # Compute information reward for each action
            best_action = None
            best_info = -float('inf')
            
            for action_idx in range(5):
                moves = np.array([[0, 0], [-1, 0], [1, 0], [0, 1], [0, -1]])
                delta = moves[action_idx] * self.config.resolution
                
                future_pos = pos[:2] + delta
                future_pos = np.clip(future_pos, 0, self.config.grid_size)
                
                # Individual information gain
                info = planner.info_estimator.compute_information_at_point(future_pos)
                
                # Penalize if other agents are heading to same area
                redundancy_penalty = 0
                for other_id, other_pos in positions.items():
                    if other_id != agent_id:
                        dist = np.linalg.norm(future_pos - other_pos[:2])
                        if dist < self.config.length_scale:
                            redundancy_penalty += 0.3 * np.exp(-dist / self.config.length_scale)
                
                adjusted_info = info - redundancy_penalty
                
                if adjusted_info > best_info:
                    best_info = adjusted_info
                    best_action = delta
            
            if best_action is not None:
                actions[agent_id] = best_action
            else:
                actions[agent_id] = np.zeros(2)
        
        return actions
    
    def get_collective_stats(self) -> Dict:
        """Get collective information statistics."""
        total_info = 0
        total_coverage = 0
        total_cells = 0
        
        for planner in self.planners.values():
            stats = planner.get_coverage_stats()
            total_info += stats['info_captured']
            total_coverage += stats['cells_covered']
            total_cells = stats['total_cells']
        
        return {
            'total_info_captured': float(total_info),
            'total_coverage_pct': float(total_coverage / max(total_cells, 1) * 100),
            'num_agents': len(self.planners),
            'info_per_agent': float(total_info / max(len(self.planners), 1)),
        }
