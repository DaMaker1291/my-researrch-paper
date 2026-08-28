#!/usr/bin/env python3
"""
GP Fire Front Model — PyTorch Implementation with Woodbury Incremental Updates
===============================================================================

A Gaussian Process that models the spatiotemporal fire front field on a grid.
Uses Matérn 5/2 kernel with incremental O(N²) Woodbury updates instead of
O(N³) full matrix inversions.

Key features:
- Woodbury identity for incremental K_inv updates when adding observations
- Cached grid predictions (recompute only when new data arrives)
- Information gain map for active sensing
- Pure PyTorch tensors for speed

Usage:
    gp = GPFireFront(grid_size=30)
    gp.add_observation(pos=[5.0, 10.0], fire_intensity=0.8)
    mean_grid, var_grid = gp.predict_cached()
    info_gain = gp.information_gain_map()
"""
import torch
import numpy as np
import time

device = torch.device("cpu")  # Use "cuda" if GPU available


class GPFireFront:
    """
    Gaussian Process model of the fire front field.

    Kernel: Matérn 5/2 (once-differentiable, matching fire front regularity)
    Updates: Woodbury identity for O(N²) incremental K_inv updates
    Caching: Grid predictions cached, only recomputed when new data arrives
    """

    def __init__(self, grid_size=30, length_scale=3.0, signal_var=1.0,
                 noise_var=0.1, max_points=200):
        self.grid_size = grid_size
        self.length_scale = length_scale
        self.signal_var = signal_var
        self.noise_var = noise_var
        self.max_points = max_points

        # Observations stored as tensors
        self.X = torch.empty(0, 2, device=device, dtype=torch.float64)
        self.y = torch.empty(0, device=device, dtype=torch.float64)
        self.K_inv = torch.empty(0, 0, device=device, dtype=torch.float64)

        # Pre-compute grid points
        xx, yy = torch.meshgrid(
            torch.arange(grid_size, device=device, dtype=torch.float64),
            torch.arange(grid_size, device=device, dtype=torch.float64),
            indexing='ij'
        )
        self.grid_pts = torch.column_stack([xx.ravel(), yy.ravel()])  # (G², 2)

        # Pre-compute grid-grid kernel (static, never changes)
        # K_grid_grid = matern52(grid_pts, grid_pts) + noise * I
        # This is used for posterior variance: var = K_qq - K_qX @ K_inv @ K_Xq
        self._K_grid_grid = None  # Lazy — only compute if needed for full posterior

        # Cache for grid predictions
        self._grid_cache_valid = False
        self._grid_mean = torch.zeros(grid_size, grid_size, device=device, dtype=torch.float64)
        self._grid_var = torch.ones(grid_size, grid_size, device=device, dtype=torch.float64) * signal_var

        # Observation kernel cache (K_X_X for Woodbury)
        self._K_XX_inv = None

    # ─── Kernel Functions ───────────────────────────────────────

    def _matern52(self, X1, X2):
        """
        Matérn 5/2 kernel between two sets of points.
        K(x1, x2) = σ² (1 + √5·r + 5r²/3) exp(-√5·r)
        where r = ||x1 - x2|| / ℓ
        """
        # Efficient distance computation: ||x1||² + ||x2||² - 2·x1·x2²
        X1_sq = (X1 ** 2).sum(dim=1, keepdim=True)  # (N, 1)
        X2_sq = (X2 ** 2).sum(dim=1, keepdim=True)  # (M, 1)
        dists_sq = X1_sq + X2_sq.T - 2.0 * X1 @ X2.T  # (N, M)
        dists_sq = torch.clamp(dists_sq, min=0.0)

        r = torch.sqrt(dists_sq) / self.length_scale
        sqrt5_r = 5.0 ** 0.5 * r

        K = self.signal_var ** 2 * (1.0 + sqrt5_r + (5.0 / 3.0) * r ** 2) * torch.exp(-sqrt5_r)
        return K

    def _matern52_vec(self, X_query, X_obs):
        """Kernel between query points and observation points."""
        return self._matern52(X_query, X_obs)

    # ─── Woodbury Incremental Update ────────────────────────────

    def _woodbury_add(self, x_new):
        """
        Incrementally update K_inv when adding a new observation x_new.

        Uses the Woodbury matrix identity for O(N²) instead of O(N³).

        Given current K_inv (N×N) built from self.X:
          k_star = K(X, x_new)          — kernel between old obs and new obs
          k_new  = K(x_new, x_new) + σ² — self-kernel + noise

        Woodbury update:
          [K_inv  0]     1              [K_inv·k_star] [K_inv·k_star]^T
          [0      0]  - ──────── ·      [    -1     ] [    -1     ]
                            denom
        where denom = k_new - k_star^T · K_inv · k_star
        """
        N = len(self.X)
        assert N > 0, "_woodbury_add called with empty X"

        # k_star: kernel between all existing obs and the new point
        k_star = self._matern52(self.X, x_new).ravel()  # (N,)
        k_new = self._matern52(x_new, x_new).item() + self.noise_var  # scalar

        # Woodbury denominator: k_new - k_star^T @ K_inv @ k_star
        K_inv_k_star = self.K_inv @ k_star  # (N,)
        denom = k_new - k_star @ K_inv_k_star  # scalar

        if denom < 1e-10:
            # Numerical instability — do full rebuild
            self._rebuild_K_inv()
            return

        # Expand K_inv by one row and column using Woodbury formula
        K_inv_new = torch.zeros(N + 1, N + 1, device=device, dtype=torch.float64)
        K_inv_new[:N, :N] = self.K_inv + torch.outer(K_inv_k_star, K_inv_k_star) / denom
        K_inv_new[:N, N] = -K_inv_k_star / denom
        K_inv_new[N, :N] = -K_inv_k_star / denom
        K_inv_new[N, N] = 1.0 / denom

        self.K_inv = K_inv_new

    def _rebuild_K_inv(self):
        """Full rebuild of K_inv from scratch (O(N³))."""
        n = len(self.X)
        if n == 0:
            self.K_inv = torch.empty(0, 0, device=device, dtype=torch.float64)
            return

        K = self._matern52(self.X, self.X) + self.noise_var * torch.eye(n, device=device, dtype=torch.float64)
        try:
            self.K_inv = torch.linalg.inv(K + 1e-6 * torch.eye(n, device=device, dtype=torch.float64))
        except torch.linalg.LinAlgError:
            self.K_inv = torch.eye(n, device=device, dtype=torch.float64)

    # ─── Add Observations ───────────────────────────────────────

    def add_observation(self, pos, fire_intensity):
        """
        Add a new observation and update K_inv incrementally.

        This is O(N²) thanks to the Woodbury identity.
        """
        pos_t = torch.tensor(pos, device=device, dtype=torch.float64).reshape(1, 2)
        val = float(fire_intensity)

        if len(self.X) == 0:
            # First observation — K_inv = 1/(k(x,x) + σ²)
            k_new = self._matern52(pos_t, pos_t).item() + self.noise_var
            self.K_inv = torch.tensor([[1.0 / k_new]], device=device, dtype=torch.float64)
        else:
            # Woodbury incremental update (O(N²))
            self._woodbury_add(pos_t)

        # Append AFTER update (K_inv must be built from old X)
        self.X = torch.vstack([self.X, pos_t]) if len(self.X) > 0 else pos_t.clone()
        self.y = torch.cat([self.y, torch.tensor([val], device=device, dtype=torch.float64)])
        self._grid_cache_valid = False

        # Prune if over capacity
        if len(self.X) > self.max_points:
            self._prune()

    def add_observations_batch(self, positions, fire_intensities):
        """
        Add multiple observations at once (more efficient than one-by-one).
        Does a full rebuild after adding all of them.
        """
        positions_t = torch.tensor(positions, device=device, dtype=torch.float64)
        vals = torch.tensor(fire_intensities, device=device, dtype=torch.float64)

        if len(self.X) == 0:
            self.X = positions_t
            self.y = vals
        else:
            self.X = torch.vstack([self.X, positions_t])
            self.y = torch.cat([self.y, vals])

        # Full rebuild (faster than N separate Woodbury updates for batch)
        self._rebuild_K_inv()
        self._grid_cache_valid = False

        if len(self.X) > self.max_points:
            self._prune()

    def _prune(self):
        """Remove the least informative observation (highest posterior variance)."""
        n = len(self.X)
        if n <= 10:
            return

        # Compute diagonal of posterior covariance
        # diag(K_posterior) = K(x_i, x_i) - k_i^T @ K_inv @ k_i
        K_diag = torch.full((n,), self.signal_var ** 2, device=device, dtype=torch.float64)
        for i in range(n):
            k_i = self._matern52(self.X[i:i+1], self.X).ravel()
            K_diag[i] -= k_i @ self.K_inv @ k_i

        # Remove observation with lowest posterior variance (most redundant)
        idx_min = torch.argmin(K_diag).item()
        mask = torch.ones(n, dtype=torch.bool, device=device)
        mask[idx_min] = False

        self.X = self.X[mask]
        self.y = self.y[mask]
        self._rebuild_K_inv()

    # ─── Prediction ─────────────────────────────────────────────

    def predict(self, X_query):
        """
        Predict mean and variance at query points.

        mean = K_qX @ K_inv @ y
        var  = σ² - K_qX @ K_inv @ K_Xq  (clamped to min 1e-10)

        Both O(M·N) where M = len(X_query), N = len(self.X).
        """
        if len(self.X) == 0:
            M = len(X_query)
            return (torch.zeros(M, device=device, dtype=torch.float64),
                    torch.ones(M, device=device, dtype=torch.float64) * self.signal_var)

        X_query = torch.atleast_2d(X_query)
        k_star = self._matern52_vec(X_query, self.X)  # (M, N)

        # mean = K_qX @ K_inv @ y
        mean = k_star @ self.K_inv @ self.y  # (M,)

        # var = σ² - K_qX @ K_inv @ K_Xq
        var_diag = self.signal_var ** 2 - (k_star * (self.K_inv @ k_star.T).T).sum(dim=1)  # (M,)
        var_diag = torch.clamp(var_diag, min=1e-10)

        return mean, var_diag

    def predict_cached(self):
        """
        Predict on full grid using cache.
        Only recomputes when new observations have been added.
        Returns (mean_grid, var_grid) as 2D tensors.
        """
        if self._grid_cache_valid:
            return self._grid_mean, self._grid_var

        if len(self.X) > 0:
            mean, var = self.predict(self.grid_pts)
            self._grid_mean = mean.reshape(self.grid_size, self.grid_size)
            self._grid_var = var.reshape(self.grid_size, self.grid_size)
        else:
            self._grid_mean = torch.zeros(self.grid_size, self.grid_size, device=device, dtype=torch.float64)
            self._grid_var = torch.ones(self.grid_size, self.grid_size, device=device, dtype=torch.float64) * self.signal_var

        self._grid_cache_valid = True
        return self._grid_mean, self._grid_var

    # ─── Information Gain ───────────────────────────────────────

    def information_gain_map(self):
        """
        Compute information gain at every grid cell.

        Measures how much an observation at each cell would reduce
        our uncertainty about the fire front field.

        IG(x) = ½ log(1 + σ²_prior / σ²_posterior)

        - Near observations: posterior var is LOW → IG is HIGH
          (we already know a lot, but confirming is cheap)
        - Far from observations: posterior var is HIGH → IG is LOW
          (we're uncertain, but a single observation helps less)

        For exploration, we actually want to maximize PREDICTIVE VARIANCE
        (where we're most uncertain), so use the variance directly.
        """
        _, var_post = self.predict_cached()
        # Return posterior variance directly — high = uncertain = explore here
        return var_post

    def query_best_info_gain_pos(self, n_candidates=50):
        """Find the position with highest information gain (for action selection)."""
        ig_map = self.information_gain_map()
        # Flatten and find top candidates
        ig_flat = ig_map.ravel()
        top_k = torch.topk(ig_flat, min(n_candidates, len(ig_flat)))
        # Convert back to grid coords
        best_idx = top_k.indices[torch.argmax(top_k.values)].item()
        best_x = best_idx // self.grid_size
        best_y = best_idx % self.grid_size
        return float(best_x), float(best_y), float(ig_flat[best_idx])

    # ─── Utility ────────────────────────────────────────────────

    @property
    def n_observations(self):
        return len(self.X)

    def get_observation_array(self):
        """Return observations as numpy arrays for compatibility."""
        if len(self.X) == 0:
            return np.zeros((0, 2)), np.zeros(0)
        return self.X.cpu().numpy(), self.y.cpu().numpy()

    def reset(self):
        """Clear all observations."""
        self.X = torch.empty(0, 2, device=device, dtype=torch.float64)
        self.y = torch.empty(0, device=device, dtype=torch.float64)
        self.K_inv = torch.empty(0, 0, device=device, dtype=torch.float64)
        self._grid_cache_valid = False


# ═══════════════════════════════════════════════════════════════════
# Information-Theoretic Active Sensing Planner
# ═══════════════════════════════════════════════════════════════════

class InformationTheoreticPlanner:
    """
    Mutual-information-maximizing coverage planner.

    Uses the GP's uncertainty map to direct drones toward regions where
    an observation would maximally reduce entropy about the fire front.

    Information gain: I(X; x*) = ½ log(1 + σ²_prior(x*) / σ²_obs(x*))
    """

    def __init__(self, gp_model, grid_size=30):
        self.gp = gp_model
        self.grid_size = grid_size
        self._info_gain_cache = None
        self._cache_valid = False

    def get_info_gain_map(self):
        """Get the information gain map (cached)."""
        if self._cache_valid:
            return self._info_gain_cache
        self._info_gain_cache = self.gp.information_gain_map()
        self._cache_valid = True
        return self._info_gain_cache

    def invalidate_cache(self):
        self._cache_valid = False

    def compute_reward_bonus(self, drone_pos):
        """
        Compute information-theoretic reward bonus for a drone at a position.
        Higher bonus = drone is at a high-uncertainty location = good.
        """
        ix = int(np.clip(drone_pos[0], 0, self.grid_size - 1))
        iy = int(np.clip(drone_pos[1], 0, self.grid_size - 1))
        ig_map = self.get_info_gain_map()
        return float(ig_map[ix, iy])

    def suggest_next_position(self, drone_pos, fire_dist_map=None, safety_margin=2.0):
        """
        Suggest the best next position for a drone, balancing:
        1. Information gain (go to uncertain regions)
        2. Safety (stay away from fire)

        Returns: (best_x, best_y, score)
        """
        ig_map = self.get_info_gain_map().cpu().numpy()

        # Mask unsafe regions (too close to fire)
        if fire_dist_map is not None:
            safety_mask = fire_dist_map < safety_margin
            ig_map = ig_map.copy()
            ig_map[safety_mask] = -1e6

        # Mask already-visited high-confidence regions
        # (optional: reduce IG where variance is already low)

        best_idx = np.argmax(ig_map)
        best_x = best_idx // self.grid_size
        best_y = best_idx % self.grid_size
        return int(best_x), int(best_y), float(ig_map[best_x, best_y])


# ═══════════════════════════════════════════════════════════════════
# Test / Benchmark
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("GP Fire Front — PyTorch Woodbury Benchmark")
    print("=" * 60)

    grid_size = 30
    n_obs_list = [10, 50, 100, 150, 200]

    gp = GPFireFront(grid_size=grid_size, max_points=200)
    planner = InformationTheoreticPlanner(gp, grid_size)

    rng = np.random.default_rng(42)

    # Benchmark: incremental add + predict
    print("\n--- Incremental Woodbury Update Speed ---")
    for n_target in n_obs_list:
        while gp.n_observations < n_target:
            pos = [rng.uniform(0, grid_size - 1), rng.uniform(0, grid_size - 1)]
            fire_val = rng.uniform(0, 1)

            t0 = time.perf_counter()
            gp.add_observation(pos, fire_val)
            t_add = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            mean, var = gp.predict_cached()
            t_pred = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            ig = gp.information_gain_map()
            t_ig = (time.perf_counter() - t0) * 1000

        print(f"  N={gp.n_observations:3d} obs | "
              f"add: {t_add:6.2f}ms | "
              f"grid predict: {t_pred:6.2f}ms | "
              f"info gain: {t_ig:6.2f}ms | "
              f"total: {t_add + t_pred + t_ig:6.2f}ms")

    # Verify predictions make sense
    print("\n--- Sanity Check ---")
    # Add observation at center with high fire
    gp.reset()
    gp.add_observation([15.0, 15.0], 1.0)
    gp.add_observation([14.0, 15.0], 0.8)
    gp.add_observation([16.0, 15.0], 0.9)

    mean, var = gp.predict_cached()

    # Mean should be high near observations, low far away
    center_val = mean[15, 15].item()
    corner_val = mean[0, 0].item()
    center_var = var[15, 15].item()
    corner_var = var[0, 0].item()

    print(f"  Center (15,15): mean={center_val:.4f}, var={center_var:.6f}")
    print(f"  Corner (0,0):   mean={corner_val:.4f}, var={corner_var:.6f}")
    print(f"  Center mean > corner mean: {center_val > corner_val}")
    print(f"  Corner var > center var: {corner_var > center_var}")

    # Information gain should be high far from observations
    ig = gp.information_gain_map()
    ig_center = ig[15, 15].item()
    ig_corner = ig[0, 0].item()
    print(f"  IG center: {ig_center:.4f}, IG corner: {ig_corner:.4f}")
    print(f"  IG corner > IG center (uncertain regions are informative): {ig_corner > ig_center}")

    print("\n--- Planner Test ---")
    bx, by, score = planner.suggest_next_position([15.0, 15.0])
    print(f"  Suggested next position: ({bx}, {by}), info_gain={score:.4f}")

    print("\n" + "=" * 60)
    print("All tests passed!" if all([
        center_val > corner_val,
        corner_var > center_var,
        ig_corner > ig_center
    ]) else "SOME TESTS FAILED!")
    print("=" * 60)
