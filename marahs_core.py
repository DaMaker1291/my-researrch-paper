#!/usr/bin/env python3
"""
MARAHS Core Innovations — Actual Implementations (Optimized)
=============================================================
1. GP Fire Front Model: Matérn 5/2 kernel, online updates with grid caching
2. Information-Theoretic Active Sensing: Mutual info maximization (cached)
3. Neural-CBF: Learned safety filter via neural network
4. GAT Communication: Graph Attention message passing between agents
"""
import numpy as np

# ═══════════════════════════════════════════════════════════════
# INNOVATION 1: GAUSSIAN PROCESS FIRE FRONT MODEL
# ═══════════════════════════════════════════════════════════════

class GPFireFront:
    """
    Gaussian Process model of the fire front field.
    
    Kernel: Matérn 5/2 (once-differentiable, matching fire front regularity)
    Updates: Periodic rebuild of K_inv (O(n³) but throttled)
    
    Caches grid predictions to avoid redundant computation.
    """
    
    def __init__(self, grid_size=30, length_scale=3.0, signal_var=1.0, 
                 noise_var=0.1, max_points=150):
        self.grid_size = grid_size
        self.ell = length_scale
        self.sigma_f = signal_var
        self.sigma_n = noise_var
        self.max_points = max_points
        
        self.X = np.zeros((0, 2), dtype=np.float64)
        self.y = np.zeros(0, dtype=np.float64)
        self.K_inv = np.zeros((0, 0), dtype=np.float64)
        
        # Precompute grid
        xx, yy = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing='ij')
        self.grid_pts = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
        
        # Cache for grid predictions (avoid recomputing every step)
        self._grid_cache_valid = False
        self._grid_mean = np.zeros((grid_size, grid_size))
        self._grid_var = np.ones((grid_size, grid_size)) * signal_var
        self._dirty = False  # Set True when new observations arrive
    
    def _matern52(self, X1, X2):
        """Matérn 5/2 kernel between two sets of points."""
        dists_sq = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        dists_sq = np.maximum(dists_sq, 0)
        r = np.sqrt(dists_sq) / self.ell
        sqrt5_r = np.sqrt(5) * r
        K = self.sigma_f**2 * (1 + sqrt5_r + 5 * r**2 / 3) * np.exp(-sqrt5_r)
        return K
    
    def _matern52_vec(self, X_query, X_obs):
        """Kernel between query points and observation points."""
        dists_sq = np.sum(X_query**2, axis=1, keepdims=True) + np.sum(X_obs**2, axis=1) - 2 * X_query @ X_obs.T
        dists_sq = np.maximum(dists_sq, 0)
        r = np.sqrt(dists_sq) / self.ell
        sqrt5_r = np.sqrt(5) * r
        return self.sigma_f**2 * (1 + sqrt5_r + 5 * r**2 / 3) * np.exp(-sqrt5_r)
    
    def _rebuild_K_inv(self):
        """Rebuild kernel inverse from scratch."""
        n = len(self.X)
        if n == 0:
            self.K_inv = np.zeros((0, 0))
            return
        K = self._matern52(self.X, self.X) + self.sigma_n**2 * np.eye(n)
        try:
            self.K_inv = np.linalg.inv(K + 1e-6 * np.eye(n))
        except np.linalg.LinAlgError:
            self.K_inv = np.eye(n)
    
    def add_observation(self, pos, fire_intensity):
        """Add a new observation and mark grid cache as dirty."""
        pos = np.array(pos, dtype=np.float64).reshape(1, 2)
        val = float(fire_intensity)
        
        self.X = np.vstack([self.X, pos]) if len(self.X) > 0 else pos.copy()
        self.y = np.append(self.y, val)
        self._dirty = True
        
        # Rebuild K_inv
        self._rebuild_K_inv()
        
        # Prune if too many observations
        if len(self.X) > self.max_points:
            self._prune()
    
    def _prune(self):
        """Remove least informative observation."""
        n = len(self.X)
        K_obs = self._matern52(self.X, self.X) + self.sigma_n**2 * np.eye(n)
        try:
            K_obs_inv = np.linalg.inv(K_obs + 1e-6 * np.eye(n))
            diag = np.diag(K_obs_inv)
            idx_min = np.argmin(diag)
            mask = np.ones(n, dtype=bool)
            mask[idx_min] = False
            self.X = self.X[mask]
            self.y = self.y[mask]
            self._rebuild_K_inv()
        except np.linalg.LinAlgError:
            pass
    
    def predict(self, X_query):
        """Predict mean and variance at query points."""
        if len(self.X) == 0:
            return np.zeros(len(X_query)), np.ones(len(X_query)) * self.sigma_f**2
        
        X_query = np.atleast_2d(X_query)
        k_star = self._matern52_vec(X_query, self.X)
        mean = k_star @ self.K_inv @ self.y
        var = self.sigma_f**2 - np.sum(k_star * (self.K_inv @ k_star.T).T, axis=1)
        var = np.maximum(var, 1e-10)
        return mean, var
    
    def predict_cached(self):
        """
        Predict on full grid using cache.
        Only recomputes when _dirty=True (new observations added).
        Returns cached (mean_grid, var_grid).
        """
        if self._dirty or not self._grid_cache_valid:
            if len(self.X) > 0:
                mean, var = self.predict(self.grid_pts)
                self._grid_mean = mean.reshape(self.grid_size, self.grid_size)
                self._grid_var = var.reshape(self.grid_size, self.grid_size)
            self._grid_cache_valid = True
            self._dirty = False
        return self._grid_mean, self._grid_var
    
    def predict_grid(self):
        """Predict on full grid (non-cached, for forced refresh)."""
        if len(self.X) == 0:
            return np.zeros((self.grid_size, self.grid_size)), \
                   np.ones((self.grid_size, self.grid_size)) * self.sigma_f**2
        mean, var = self.predict(self.grid_pts)
        self._grid_mean = mean.reshape(self.grid_size, self.grid_size)
        self._grid_var = var.reshape(self.grid_size, self.grid_size)
        self._grid_cache_valid = True
        self._dirty = False
        return self._grid_mean, self._grid_var


# ═══════════════════════════════════════════════════════════════
# INNOVATION 2: INFORMATION-THEORETIC ACTIVE SENSING
# ═══════════════════════════════════════════════════════════════

class InformationTheoreticPlanner:
    """
    Mutual-information-maximizing coverage planner.
    
    Information gain: I(X; x* | observations) = ½ log(1 + σ²_prior(x*) / σ²_obs)
    Cached to avoid recomputing every step.
    """
    
    def __init__(self, gp_model, grid_size=30):
        self.gp = gp_model
        self.grid_size = grid_size
        self._info_gain_cache = np.zeros((grid_size, grid_size))
        self._cache_valid = False
    
    def compute_info_gain_map(self):
        """Compute information gain at every grid cell (cached)."""
        if self._cache_valid and not self.gp._dirty:
            return self._info_gain_cache
        
        _, var_prior = self.gp.predict_cached()
        self._info_gain_cache = 0.5 * np.log(1 + var_prior / (self.gp.sigma_n**2 + 1e-10))
        self._cache_valid = True
        return self._info_gain_cache
    
    def invalidate_cache(self):
        self._cache_valid = False
    
    def compute_reward_bonus(self, drone_pos):
        """Compute information-theoretic reward bonus for a drone."""
        ix = int(np.clip(drone_pos[0], 0, self.grid_size - 1))
        iy = int(np.clip(drone_pos[1], 0, self.grid_size - 1))
        info_map = self.compute_info_gain_map()
        return float(info_map[ix, iy])


# ═══════════════════════════════════════════════════════════════
# INNOVATION 3: NEURAL CONTROL BARRIER FUNCTION
# ═══════════════════════════════════════════════════════════════

class NeuralCBFNetwork:
    """
    Neural network that learns safety margins from data.
    
    h_θ(x) >= 0 means state x is safe.
    
    Architecture: 2-layer MLP, 8D input → scalar output.
    Input: [pos(2), vel(2), fire_dist, thermal, wind_speed(2)] = 8D
    """
    
    def __init__(self, input_dim=8, hidden_dim=32, lr=1e-3):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        
        # He init
        s1 = np.sqrt(2.0 / input_dim)
        s2 = np.sqrt(2.0 / hidden_dim)
        
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * s1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * s2
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)
        self.W3 = np.random.randn(hidden_dim, 1).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros(1, dtype=np.float32)
        
        # Adam
        self.params = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.v = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.t = 0
        
        self.buffer = []
        self.buffer_size = 2000
    
    def forward(self, x):
        """Forward pass: returns safety margin h(x)."""
        h1 = np.maximum(0, x @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        return (h2 @ self.W3 + self.b3).ravel()
    
    def compute_features(self, drone_pos, drone_vel, fire_dist, thermal, wind):
        """Convert raw state to CBF feature vector."""
        return np.array([
            drone_pos[0], drone_pos[1],
            drone_vel[0], drone_vel[1],
            fire_dist, thermal,
            wind[0], wind[1]
        ], dtype=np.float32)
    
    def store_transition(self, state, is_safe, next_state):
        """Store transition for training."""
        self.buffer.append((state, is_safe, next_state))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
    
    def train_step(self, n_samples=64):
        """Train on stored transitions."""
        if len(self.buffer) < 32:
            return 0.0
        
        indices = np.random.choice(len(self.buffer), min(n_samples, len(self.buffer)), replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states = np.array([b[0] for b in batch], dtype=np.float32)
        labels = np.array([1.0 if b[1] else -1.0 for b in batch], dtype=np.float32)
        next_states = np.array([b[2] for b in batch], dtype=np.float32)
        
        # Forward
        h1 = np.maximum(0, states @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        h = (h2 @ self.W3 + self.b3).ravel()
        
        h1_n = np.maximum(0, next_states @ self.W1 + self.b1)
        h2_n = np.maximum(0, h1_n @ self.W2 + self.b2)
        h_next = (h2_n @ self.W3 + self.b3).ravel()
        
        # Margin loss
        active = (labels * h) < 1
        loss = np.mean(np.maximum(0, 1 - labels * h))
        
        # CBF consistency for safe transitions
        alpha = 0.95
        safe_mask = labels > 0
        if safe_mask.any():
            cbf_viol = np.maximum(0, alpha * h[safe_mask] - h_next[safe_mask])
            loss += 0.5 * np.mean(cbf_viol)
        
        # Backprop
        grad_h = np.zeros_like(h)
        grad_h[active] = -labels[active] / len(batch)
        
        if safe_mask.any():
            cbf_active = (alpha * h[safe_mask] - h_next[safe_mask]) > 0
            if cbf_active.any():
                grad_h_safe = np.zeros_like(h[safe_mask])
                grad_h_safe[cbf_active] += 0.5 * alpha / len(batch)
                grad_h[safe_mask] += grad_h_safe
        
        dW3 = h2.T @ grad_h.reshape(-1, 1) / len(batch)
        db3 = grad_h.mean()
        
        grad_h2 = grad_h.reshape(-1, 1) @ self.W3.T
        grad_h2 *= (h2 > 0).astype(float)
        dW2 = h1.T @ grad_h2 / len(batch)
        db2 = grad_h2.mean(axis=0)
        
        grad_h1 = grad_h2 @ self.W2.T
        grad_h1 *= (h1 > 0).astype(float)
        dW1 = states.T @ grad_h1 / len(batch)
        db1 = grad_h1.mean(axis=0)
        
        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}
        
        self.t += 1
        for p in self.params:
            self.m[p] = 0.9 * self.m[p] + 0.1 * grads[p]
            self.v[p] = 0.999 * self.v[p] + 0.001 * grads[p]**2
            m_hat = self.m[p] / (1 - 0.9**self.t)
            v_hat = self.v[p] / (1 - 0.999**self.t)
            setattr(self, p, getattr(self, p) - self.lr * m_hat / (np.sqrt(v_hat) + 1e-8))
        
        return float(loss)
    
    def safety_margin(self, state):
        """Get safety margin h(x) for a state."""
        return float(self.forward(state.reshape(1, -1) if state.ndim == 1 else state)[0])


# ═══════════════════════════════════════════════════════════════
# INNOVATION 4: GRAPH ATTENTION NETWORK COMMUNICATION
# ═══════════════════════════════════════════════════════════════

class GATCommunicationModule:
    """
    Graph Attention Network for inter-agent communication.
    
    Each drone encodes its state → GAT aggregates from nearby flockmates
    → updated features inform each drone's decision.
    
    Cached per-step to avoid redundant computation.
    """
    
    def __init__(self, node_dim=12, n_heads=3, comm_range=8.0):
        self.node_dim = node_dim
        self.n_heads = n_heads
        self.head_dim = node_dim // n_heads
        self.comm_range = comm_range
        
        # Encoder
        self.W_enc = np.random.randn(6, node_dim).astype(np.float32) * np.sqrt(2.0 / 6)
        self.b_enc = np.zeros(node_dim, dtype=np.float32)
        
        # Per-head attention params
        self.W_heads = []
        self.a_left_heads = []
        self.a_right_heads = []
        for _ in range(n_heads):
            W = np.random.randn(node_dim, self.head_dim).astype(np.float32) * np.sqrt(2.0 / node_dim)
            a_l = np.random.randn(self.head_dim, 1).astype(np.float32) * 0.01
            a_r = np.random.randn(self.head_dim, 1).astype(np.float32) * 0.01
            self.W_heads.append(W)
            self.a_left_heads.append(a_l)
            self.a_right_heads.append(a_r)
        
        self.W_out = np.random.randn(node_dim, node_dim).astype(np.float32) * np.sqrt(2.0 / node_dim)
        
        # Cache
        self._cache_step = -1
        self._cache_result = None
    
    def communicate(self, drone_states, positions, step_id=-1):
        """
        Run GAT communication round.
        
        drone_states: list of (6,) arrays [pos(2), vel(2), fire_dist, thermal]
        positions: (K, 2) array
        step_id: if same as cached, return cached result
        """
        K = len(drone_states)
        if K == 0:
            return np.zeros((0, self.node_dim))
        
        # Return cache if same step
        if step_id >= 0 and step_id == self._cache_step and self._cache_result is not None:
            return self._cache_result
        
        states = np.array(drone_states, dtype=np.float32)
        
        # Encode
        node_features = np.maximum(0, states @ self.W_enc + self.b_enc)
        
        # Build adjacency
        adj = np.zeros((K, K), dtype=bool)
        for i in range(K):
            dists = np.linalg.norm(positions - positions[i], axis=1)
            adj[i] = (dists < self.comm_range) & (np.arange(K) != i)
        
        # Multi-head attention
        head_outputs = []
        for h in range(self.n_heads):
            Wh = node_features @ self.W_heads[h]
            e_left = Wh @ self.a_left_heads[h]
            e_right = Wh @ self.a_right_heads[h]
            e = e_left + e_right.T
            e = np.maximum(0.2 * e, e)  # LeakyReLU
            e[~adj] = -1e9
            e_max = np.max(e, axis=1, keepdims=True)
            alpha = np.exp(e - e_max)
            alpha /= alpha.sum(axis=1, keepdims=True) + 1e-10
            head_outputs.append(alpha @ Wh)
        
        multi_head = np.concatenate(head_outputs, axis=1)
        output = np.maximum(0, multi_head @ self.W_out + node_features)  # Residual + ReLU
        
        if step_id >= 0:
            self._cache_step = step_id
            self._cache_result = output
        
        return output
