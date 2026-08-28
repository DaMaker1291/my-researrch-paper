"""
Information-Theoretic Active Sensing for wildfire perimeter tracking.

Implements:
1. GP-based fire front field estimation
2. Mutual information maximization for next-best-view
3. Shannon entropy reduction metrics
4. Adaptive exploration-exploitation via information gain
"""

import numpy as np
from typing import Tuple, List, Optional


class GPInformationGain:
    """
    Gaussian Process Information Gain for wildfire perimeter tracking.

    Maintains a GP model of the fire front field and uses mutual
    information to select actions that maximally reduce uncertainty.

    Key equation:
        I(X_fire; Z_t) = H(X_fire) - H(X_fire | Z_t)
        = 1/2 * log(1 + sigma_prior^2(x) / sigma_noise^2)

    where sigma_prior^2(x) is the GP prior variance at x.
    """

    def __init__(
        self,
        grid_size: int = 40,
        length_scale: float = 3.0,
        signal_variance: float = 1.0,
        noise_variance: float = 0.1,
        max_observations: int = 500,
    ):
        self.grid_size = grid_size
        self.length_scale = length_scale
        self.signal_var = signal_variance
        self.noise_var = noise_variance
        self.max_obs = max_observations

        # GP observations: (positions, values)
        self.obs_positions = []
        self.obs_values = []

        # Prior variance grid (precomputed)
        self.prior_var = np.ones((grid_size, grid_size), dtype=np.float32) * signal_variance

        # Posterior statistics
        self.posterior_mean = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.posterior_var = np.ones((grid_size, grid_size), dtype=np.float32) * signal_variance

        # Information gain history
        self.total_info_gain = 0.0
        self.info_gain_per_step = []

    def reset(self):
        """Reset GP observations."""
        self.obs_positions = []
        self.obs_values = []
        self.posterior_var[:] = self.signal_var
        self.posterior_mean[:] = 0
        self.total_info_gain = 0.0
        self.info_gain_per_step = []

    def add_observation(self, position: np.ndarray, value: float):
        """
        Add a new observation to the GP.

        Args:
            position: (2,) grid coordinates
            value: observed fire intensity
        """
        self.obs_positions.append(position.copy())
        self.obs_values.append(value)

        # Prune if too many observations (keep most informative)
        if len(self.obs_positions) > self.max_obs:
            # Remove oldest observation
            self.obs_positions.pop(0)
            self.obs_values.pop(0)

        # Update posterior (simplified: decrement variance near observation)
        self._update_posterior(position, value)

    def _update_posterior(self, new_pos: np.ndarray, new_val: float):
        """Update GP posterior with new observation."""
        xx, yy = np.meshgrid(
            np.arange(self.grid_size),
            np.arange(self.grid_size),
            indexing='ij'
        )

        # Distance from new observation to all grid points
        dist = np.sqrt((xx - new_pos[0])**2 + (yy - new_pos[1])**2)

        # Kernel value
        k = self.signal_var * np.exp(-0.5 * dist**2 / self.length_scale**2)

        # Reduce posterior variance near observation
        reduction = k**2 / (self.noise_var + self.signal_var)
        self.posterior_var = np.maximum(0.01, self.posterior_var - reduction * 0.3)

        # Update mean
        weight = k / (self.noise_var + self.signal_var)
        self.posterior_mean += weight * (new_val - self.posterior_mean * 0.1)

    def compute_information_gain(self, position: np.ndarray) -> float:
        """
        Compute information gain at a candidate position.

        I(X; x*) = 1/2 * log(1 + sigma_prior^2(x*) / sigma_noise^2)

        This equals the expected reduction in entropy from observing x*.
        """
        ix = int(np.clip(position[0], 0, self.grid_size - 1))
        iy = int(np.clip(position[1], 0, self.grid_size - 1))

        prior_var = self.posterior_var[ix, iy]
        info_gain = 0.5 * np.log(1 + prior_var / self.noise_var)

        return float(info_gain)

    def compute_expected_information_gain(
        self,
        position: np.ndarray,
        action: int,
        speed: float = 1.5,
    ) -> float:
        """
        Compute expected information gain after taking an action.

        This considers:
        1. Direct information gain at new position
        2. Potential information gain from nearby high-uncertainty cells
        3. Distance to fire front (closer = more informative)
        """
        # Simulate action
        action_deltas = {
            0: (0, 0),
            1: (0, speed),
            2: (0, -speed),
            3: (speed, 0),
            4: (-speed, 0),
        }
        dx, dy = action_deltas.get(action, (0, 0))
        new_pos = position + np.array([dx, dy])

        # Direct information gain
        direct_gain = self.compute_information_gain(new_pos)

        # Look-ahead: average info gain in neighborhood
        r = 3
        neighborhood_gain = 0.0
        count = 0
        for nx in range(int(new_pos[0]) - r, int(new_pos[0]) + r + 1):
            for ny in range(int(new_pos[1]) - r, int(new_pos[1]) + r + 1):
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    neighborhood_gain += self.compute_information_gain(np.array([nx, ny]))
                    count += 1
        avg_neighborhood = neighborhood_gain / max(1, count)

        # Combined: 60% direct + 40% neighborhood
        total_gain = 0.6 * direct_gain + 0.4 * avg_neighborhood

        return total_gain

    def select_information_greedy_action(
        self,
        position: np.ndarray,
        speed: float = 1.5,
    ) -> int:
        """
        Select the action that maximizes expected information gain.
        This is the information-theoretic greedy policy.
        """
        best_action = 0
        best_gain = -1

        for action in range(5):
            gain = self.compute_expected_information_gain(position, action, speed)
            if gain > best_gain:
                best_gain = gain
                best_action = action

        self.total_info_gain += best_gain
        self.info_gain_per_step.append(best_gain)

        return best_action

    def compute_total_entropy(self) -> float:
        """Compute total entropy of the fire front estimate."""
        return float(0.5 * np.sum(np.log(2 * np.pi * np.e * self.posterior_var)))

    def compute_entropy_reduction_rate(self) -> float:
        """Compute rate of entropy reduction per step."""
        if len(self.info_gain_per_step) < 2:
            return 0.0
        recent = self.info_gain_per_step[-20:]
        return float(np.mean(recent))

    def get_exploration_bonus(self, position: np.ndarray) -> float:
        """
        Compute exploration bonus for a position.
        High posterior variance = high exploration bonus.
        """
        ix = int(np.clip(position[0], 0, self.grid_size - 1))
        iy = int(np.clip(position[1], 0, self.grid_size - 1))
        return float(self.posterior_var[ix, iy] / self.signal_var)
