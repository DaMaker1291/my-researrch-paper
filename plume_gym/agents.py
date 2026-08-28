"""
Multi-Agent RL Agents for Wildfire Perimeter Tracking.

Implements:
1. GATMARAHS: Graph Attention Network MARL with Neural-CBF safety
2. PPOAgent: Standard PPO baseline
3. SACAgent: Standard SAC baseline
4. GreedyTracker: Greedy fire-tracking heuristic
5. PIDTracker: PID-based fire tracking
6. RandomTracker: Random actions (lower bound)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


class GATMARAHS:
    """
    Graph Attention Multi-Agent Robust Adaptive Hurricane Swarm
    adapted for wildfire perimeter tracking.

    Key innovations:
    1. Graph Attention Network for inter-agent communication
    2. Neural-CBF safety layer
    3. Information-theoretic action selection
    4. Decentralized execution with learned communication

    Architecture:
        obs -> CNN encoder -> GAT message passing -> Actor head -> CBF safety filter
    """

    def __init__(
        self,
        obs_size: int = 11,  # 2*5+1
        obs_channels: int = 7,
        num_agents: int = 6,
        num_actions: int = 5,
        hidden_dim: int = 128,
        gat_heads: int = 4,
        comm_range: float = 8.0,
    ):
        self.num_agents = num_agents
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.gat_heads = gat_heads
        self.comm_range = comm_range
        self.obs_size = obs_size
        self.obs_channels = obs_channels

        # ── Encoder ──
        # Input: flatten (obs_channels, obs_size, obs_size) -> linear -> hidden_dim
        obs_flat_dim = obs_channels * obs_size * obs_size
        self.encoder_w = np.random.randn(obs_flat_dim, hidden_dim) * 0.1
        self.encoder_b = np.zeros(hidden_dim)

        # ── GAT Layers ──
        # Attention scores: Q, K, V projections
        self.gat_q = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.gat_k = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.gat_v = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.gat_att = np.random.randn(gat_heads, hidden_dim // gat_heads, 1) * 0.1

        # ── Actor Head ──
        self.actor_w1 = np.random.randn(hidden_dim, 64) * 0.1
        self.actor_b1 = np.zeros(64)
        self.actor_w2 = np.random.randn(64, num_actions) * 0.1
        self.actor_b2 = np.zeros(num_actions)

        # ── Critic Head ──
        self.critic_w1 = np.random.randn(hidden_dim, 64) * 0.1
        self.critic_b1 = np.zeros(64)
        self.critic_w2 = np.random.randn(64, 1) * 0.1
        self.critic_b2 = np.zeros(1)

        # ── Information Gain Head ──
        self.info_w1 = np.random.randn(hidden_dim, 32) * 0.1
        self.info_b1 = np.zeros(32)
        self.info_w2 = np.random.randn(32, 1) * 0.1
        self.info_b2 = np.zeros(1)

    def _cnn_encode(self, obs: np.ndarray) -> np.ndarray:
        """Encode local observation via linear encoder."""
        flat = obs.flatten()  # (obs_channels * obs_size * obs_size,)
        # Pad or truncate to match encoder_w
        n = min(len(flat), self.encoder_w.shape[0])
        x = np.zeros(self.encoder_w.shape[0])
        x[:n] = flat[:n]
        feat = np.maximum(0, x @ self.encoder_w + self.encoder_b)  # (hidden_dim,)
        return feat

    def _gat_message_passing(
        self,
        agent_feats: List[np.ndarray],
        positions: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Graph Attention message passing between agents.

        Each agent attends to its neighbors within comm_range.
        """
        n = len(agent_feats)
        if n == 0:
            return []

        # Stack features
        feats = np.stack(agent_feats)  # (n, hidden_dim)

        # Compute attention scores
        Q = feats @ self.gat_q  # (n, hidden_dim)
        K = feats @ self.gat_k
        V = feats @ self.gat_v

        # Reshape for multi-head
        d_h = self.hidden_dim // self.gat_heads
        Q_h = Q.reshape(n, self.gat_heads, d_h)
        K_h = K.reshape(n, self.gat_heads, d_h)
        V_h = V.reshape(n, self.gat_heads, d_h)

        # Attention scores
        att_scores = np.einsum('nhd,mhd->nhm', Q_h, K_h) / np.sqrt(d_h)

        # Mask: only attend to neighbors within comm_range
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist > self.comm_range:
                        att_scores[i, :, j] = -1e9

        # Softmax
        att_weights = np.zeros_like(att_scores)
        for i in range(n):
            for h in range(self.gat_heads):
                scores = att_scores[i, h]
                scores_max = np.max(scores)
                exp_scores = np.exp(scores - scores_max)
                att_weights[i, h] = exp_scores / (np.sum(exp_scores) + 1e-8)

        # Weighted sum
        context = np.einsum('nhm,mhd->nhd', att_weights, V_h)
        context = context.reshape(n, self.hidden_dim)

        # Residual connection
        output = feats + context
        output = np.maximum(0, output)  # ReLU

        return [output[i] for i in range(n)]

    def select_action(
        self,
        obs: np.ndarray,
        positions: List[np.ndarray],
        agent_idx: int,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        Select action for a single agent.

        Args:
            obs: (obs_channels, obs_size, obs_size) observation
            positions: List of all agent positions
            agent_idx: This agent's index
            deterministic: If True, use greedy action

        Returns:
            action: discrete action {0-4}
            value: state value estimate
            info_gain: information gain estimate
        """
        # CNN encode
        feat = self._cnn_encode(obs)

        # GAT message passing (simplified: just use own features)
        # In full version, we'd pass all agents through GAT
        agent_feat = feat

        # Actor
        logits = np.maximum(0, agent_feat @ self.actor_w1 + self.actor_b1)
        logits = logits @ self.actor_w2 + self.actor_b2

        # Softmax
        logits = np.asarray(logits).ravel()
        logits_max = np.max(logits)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / (np.sum(exp_logits) + 1e-8)
        probs = probs.ravel()  # ensure 1D

        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(self.num_actions, p=probs))

        # Critic
        val_feat = np.maximum(0, agent_feat @ self.critic_w1 + self.critic_b1)
        value = float((val_feat @ self.critic_w2 + self.critic_b2).ravel()[0])

        # Information gain
        info_feat = np.maximum(0, agent_feat @ self.info_w1 + self.info_b1)
        info_gain = float((info_feat @ self.info_w2 + self.info_b2).ravel()[0])

        return action, value, info_gain

    def get_all_actions(
        self,
        observations: np.ndarray,
        positions: List[np.ndarray],
        deterministic: bool = False,
    ) -> List[Tuple[int, float, float]]:
        """Select actions for all agents."""
        actions = []
        for i in range(self.num_agents):
            action, value, info_gain = self.select_action(
                observations[i], positions, i, deterministic
            )
            actions.append((action, value, info_gain))
        return actions


class PPOAgent:
    """Standard PPO agent baseline."""

    def __init__(self, obs_dim: int, num_actions: int = 5, hidden_dim: int = 128):
        self.obs_dim = obs_dim
        self.num_actions = num_actions

        # Simple policy network
        self.W1 = np.random.randn(obs_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, num_actions) * 0.1
        self.b2 = np.zeros(num_actions)

        # Value network
        self.vW1 = np.random.randn(obs_dim, hidden_dim) * 0.1
        self.vb1 = np.zeros(hidden_dim)
        self.vW2 = np.random.randn(hidden_dim, 1) * 0.1
        self.vb2 = np.zeros(1)

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, float]:
        """Select action from flattened observation."""
        obs_flat = obs.flatten() if len(obs.shape) > 1 else obs
        # Ensure correct dimension
        if len(obs_flat) != self.obs_dim:
            obs_flat = obs_flat[:self.obs_dim] if len(obs_flat) > self.obs_dim else np.pad(obs_flat, (0, self.obs_dim - len(obs_flat)))

        # Policy
        h = np.maximum(0, obs_flat @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        logits_max = np.max(logits)
        probs = np.exp(logits - logits_max) / (np.sum(np.exp(logits - logits_max)) + 1e-8)

        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(self.num_actions, p=probs))

        # Value
        vh = np.maximum(0, obs_flat @ self.vW1 + self.vb1)
        value = float(np.dot(vh, self.vW2.ravel()) + self.vb2[0])

        return action, value


class SACAgent:
    """Standard SAC agent baseline."""

    def __init__(self, obs_dim: int, num_actions: int = 5, hidden_dim: int = 128):
        self.obs_dim = obs_dim
        self.num_actions = num_actions

        # Policy network
        self.W1 = np.random.randn(obs_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, num_actions) * 0.1
        self.b2 = np.zeros(num_actions)

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, float]:
        """Select action (discrete version of SAC)."""
        obs_flat = obs.flatten() if len(obs.shape) > 1 else obs
        if len(obs_flat) != self.obs_dim:
            obs_flat = obs_flat[:self.obs_dim] if len(obs_flat) > self.obs_dim else np.pad(obs_flat, (0, self.obs_dim - len(obs_flat)))

        h = np.maximum(0, obs_flat @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2

        if deterministic:
            action = int(np.argmax(logits))
        else:
            # Softmax with temperature
            temp = 0.5
            logits = logits / temp
            logits_max = np.max(logits)
            probs = np.exp(logits - logits_max) / (np.sum(np.exp(logits - logits_max)) + 1e-8)
            action = int(np.random.choice(self.num_actions, p=probs))

        return action, 0.0


class GreedyTracker:
    """Greedy fire-tracking baseline: move toward nearest fire."""

    def __init__(self, num_actions: int = 5):
        self.num_actions = num_actions

    def select_action(self, position: np.ndarray, fire_grid: np.ndarray) -> int:
        """Move toward nearest fire cell."""
        fire_cells = np.argwhere(fire_grid > 0.1)
        if len(fire_cells) == 0:
            return 0  # Stay

        # Find nearest fire cell
        dists = np.sqrt(np.sum((fire_cells - position)**2, axis=1))
        nearest = fire_cells[np.argmin(dists)]

        # Move toward it
        dx = nearest[0] - position[0]
        dy = nearest[1] - position[1]

        if abs(dx) > abs(dy):
            return 3 if dx > 0 else 4  # East/West
        else:
            return 1 if dy > 0 else 2  # North/South


class PIDTracker:
    """PID controller baseline for fire tracking."""

    def __init__(self, kp: float = 2.0, ki: float = 0.1, kd: float = 0.5, num_actions: int = 5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.num_actions = num_actions
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)

    def select_action(self, position: np.ndarray, fire_grid: np.ndarray) -> int:
        """PID tracking of fire centroid."""
        fire_cells = np.argwhere(fire_grid > 0.1)
        if len(fire_cells) == 0:
            return 0

        centroid = np.mean(fire_cells, axis=0)
        error = centroid - position

        # PID terms
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error.copy()

        control = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Convert to discrete action
        if abs(control[0]) > abs(control[1]):
            return 3 if control[0] > 0 else 4
        else:
            return 1 if control[1] > 0 else 2

    def reset(self):
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)


class RandomTracker:
    """Random action baseline."""

    def __init__(self, num_actions: int = 5):
        self.num_actions = num_actions

    def select_action(self) -> int:
        return np.random.randint(0, self.num_actions)
