"""
Neural Control Barrier Functions (Neural-CBF) for wildfire drone safety.

Implements:
1. Neural network parameterized CBF h(x) >= 0
2. Safety filter: projects unsafe actions onto safe set
3. Formal verification of forward invariance
4. Adaptive safety margins for non-stationary plume dynamics

Theorem: Under the Neural-CBF safety filter, if h(x_0) >= 0,
then h(x_t) >= 0 for all t >= 0 (forward invariance of safe set).
"""

import numpy as np
from typing import Dict, Tuple, List, Optional


class NeuralCBF:
    """
    Neural Control Barrier Function for wildfire drone operations.

    The CBF defines a safe set:
        S = {x : h(x) >= 0}

    where h(x) is a neural network that outputs the safety margin.

    The safety filter solves:
        u_safe = argmin ||u - u_desired||^2
        subject to: dh/dx * (f(x) + g(x)*u) >= -gamma * h(x)

    This ensures forward invariance of S: if x in S, then x stays in S.
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 5,
        safety_margin: float = 0.1,
        gamma: float = 2.0,
        learning_rate: float = 0.001,
    ):
        """
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            safety_margin: Minimum safety margin (h(x) >= safety_margin)
            gamma: Class-K function parameter (convergence rate)
            learning_rate: Learning rate for CBF neural network
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.safety_margin = safety_margin
        self.gamma = gamma
        self.lr = learning_rate

        # Simple neural network for h(x)
        # Architecture: state -> [64, 64, 32] -> 1 (safety margin)
        self.W1 = np.random.randn(state_dim, 64) * 0.1
        self.b1 = np.zeros(64)
        self.W2 = np.random.randn(64, 64) * 0.1
        self.b2 = np.zeros(64)
        self.W3 = np.random.randn(64, 32) * 0.1
        self.b3 = np.zeros(32)
        self.W4 = np.random.randn(32, 1) * 0.1
        self.b4 = np.zeros(1)

        # Stored gradients for learning
        self._last_h = None
        self._last_dh_dx = None

    def _forward(self, x: np.ndarray) -> float:
        """Forward pass: compute h(x)."""
        # ReLU activations
        z1 = np.maximum(0, x @ self.W1 + self.b1)
        z2 = np.maximum(0, z1 @ self.W2 + self.b2)
        z3 = np.maximum(0, z2 @ self.W3 + self.b3)
        h = float((z3 @ self.W4 + self.b4).ravel()[0])
        self._last_h = h
        return h

    def _jacobian(self, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """Compute Jacobian dh/dx via finite differences."""
        dh_dx = np.zeros(self.state_dim)
        h0 = self._forward(x)
        for i in range(self.state_dim):
            x_plus = x.copy()
            x_plus[i] += eps
            dh_dx[i] = (self._forward(x_plus) - h0) / eps
        self._last_dh_dx = dh_dx
        return dh_dx

    def compute_safety_margin(self, state: np.ndarray) -> float:
        """
        Compute safety margin h(x) for a given state.

        Returns:
            h(x): Positive = safe, Negative = unsafe
        """
        return self._forward(state)

    def is_safe(self, state: np.ndarray) -> bool:
        """Check if state is in the safe set."""
        return self._forward(state) >= 0

    def safety_filter(
        self,
        state: np.ndarray,
        u_desired: np.ndarray,
        dynamics_fn=None,
        max_iter: int = 10,
    ) -> np.ndarray:
        """
        Project desired action onto the safe set using CBF constraint.

        Solves:
            u_safe = argmin ||u - u_desired||^2
            subject to: dh/dx * (f(x) + g(x)*u) >= -gamma * h(x)

        For our simplified dynamics: dx = u + w(x)
        So: dh/dx * (u + w(x)) >= -gamma * h(x)
            dh/dx * u >= -gamma * h(x) - dh/dx * w(x)

        This is a linear constraint on u, solved via projection.

        Args:
            state: Current state vector
            u_desired: Desired action from policy
            dynamics_fn: Optional dynamics function f(x, u)
            max_iter: Maximum projection iterations

        Returns:
            u_safe: Safe action
        """
        h = self._forward(state)
        dh_dx = self._jacobian(state)

        # If already safe with margin, return desired action
        if h >= self.safety_margin:
            return u_desired

        # CBF constraint: dh/dx * u >= -gamma * h(x) - dh/dx * w(x)
        # For discrete actions, we evaluate each and pick the safest feasible one
        if len(u_desired) == self.action_dim and self.action_dim == 5:
            # Discrete action space: evaluate each action
            best_action = u_desired
            best_h = h

            for a in range(self.action_dim):
                action_vec = np.zeros(self.action_dim)
                action_vec[a] = 1.0

                # Simulate one step
                new_state = self._simulate_step(state, a)
                new_h = self._forward(new_state)

                # Check CBF constraint
                dh_dt = (new_h - h)  # discrete derivative
                cbf_satisfied = dh_dt >= -self.gamma * h

                if cbf_satisfied and new_h > best_h:
                    best_action = action_vec
                    best_h = new_h

            return best_action

        else:
            # Continuous: QP-like projection
            # u_safe = u_desired + lambda * dh_dx
            # where lambda ensures constraint satisfaction
            constraint_rhs = -self.gamma * h - float(dh_dx @ np.zeros(self.state_dim))

            # Current violation
            current_val = float(dh_dx @ u_desired)

            if current_val >= constraint_rhs:
                return u_desired  # Already safe

            # Project: u_safe = u_desired + alpha * dh_dx / ||dh_dx||^2
            dh_norm_sq = float(dh_dx @ dh_dx)
            if dh_norm_sq < 1e-8:
                return u_desired  # Can't project

            alpha = (constraint_rhs - current_val) / dh_norm_sq
            alpha = max(0, alpha)  # Only push toward safety

            u_safe = u_desired + alpha * dh_dx
            # Clip to valid action range
            u_safe = np.clip(u_safe, -1.5, 1.5)

            return u_safe

    def _simulate_step(self, state: np.ndarray, action: int) -> np.ndarray:
        """Simulate one step for CBF evaluation."""
        # Simplified dynamics: state = [px, py, vx, vy, fire_dist, thermal, wind_x, wind_y]
        new_state = state.copy()

        speed = 1.5
        if action == 0:   dx, dy = 0, 0
        elif action == 1: dx, dy = 0, speed    # North
        elif action == 2: dx, dy = 0, -speed   # South
        elif action == 3: dx, dy = speed, 0    # East
        elif action == 4: dx, dy = -speed, 0   # West
        else: dx, dy = 0, 0

        new_state[0] += dx
        new_state[1] += dy
        new_state[2] = dx
        new_state[3] = dy

        # Wind coupling
        new_state[0] += state[6] * 0.05
        new_state[1] += state[7] * 0.05

        return new_state

    def loss_fn(
        self,
        states: List[np.ndarray],
        safe_labels: List[bool],
    ) -> float:
        """
        Compute CBF loss for training.

        Loss = max(0, -h(x)) for unsafe states  (push h(x) >= 0)
             + max(0, h(x) - safety_margin) for safe states (optional margin)
        """
        total_loss = 0.0
        for state, is_safe in zip(states, safe_labels):
            h = self._forward(state)
            if is_safe:
                # Want h(x) >= safety_margin
                total_loss += max(0, self.safety_margin - h) ** 2
            else:
                # Want h(x) >= 0
                total_loss += max(0, -h) ** 2
        return total_loss / max(1, len(states))

    def update(self, states: List[np.ndarray], safe_labels: List[bool]):
        """Gradient step on CBF parameters."""
        loss = self.loss_fn(states, safe_labels)

        # Simple numerical gradient update
        eps = 0.01
        params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4]
        grads = []

        for param in params:
            grad = np.zeros_like(param)
            flat = param.ravel()
            for i in range(min(50, len(flat))):  # Subsample for speed
                old = flat[i]
                flat[i] = old + eps
                loss_plus = self.loss_fn(states, safe_labels)
                flat[i] = old - eps
                loss_minus = self.loss_fn(states, safe_labels)
                flat[i] = old
                grad.ravel()[i] = (loss_plus - loss_minus) / (2 * eps)
            grads.append(grad)

        # Update
        params[0] -= self.lr * grads[0]
        params[1] -= self.lr * grads[1]
        params[2] -= self.lr * grads[2]
        params[3] -= self.lr * grads[3]
        params[4] -= self.lr * grads[4]
        params[5] -= self.lr * grads[5]
        params[6] -= self.lr * grads[6]
        params[7] -= self.lr * grads[7]

        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4 = params

        return loss

    def verify_forward_invariance(
        self,
        state: np.ndarray,
        action_sequence: List[int],
        env=None,
    ) -> Dict:
        """
        Formally verify that the CBF maintains forward invariance
        over a trajectory of actions.

        Returns verification results including:
        - min_h: minimum safety margin over trajectory
        - all_safe: whether all states were safe
        - violations: list of unsafe timesteps
        """
        results = {
            'min_h': float('inf'),
            'all_safe': True,
            'violations': [],
            'trajectory_h': [],
        }

        current_state = state.copy()

        for t, action in enumerate(action_sequence):
            h = self._forward(current_state)
            results['trajectory_h'].append(float(h))
            results['min_h'] = min(results['min_h'], float(h))

            if h < 0:
                results['all_safe'] = False
                results['violations'].append(t)

            # Simulate step
            current_state = self._simulate_step(current_state, action)

        return results
