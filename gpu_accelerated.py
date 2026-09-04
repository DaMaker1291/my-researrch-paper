#!/usr/bin/env python3
"""
=================================================================
GPU-Accelerated PlumeGym-MARL: 32 Parallel Episodes on Dual T4
=================================================================
Drop-in replacement for kaggle_full_run.py training.

KEY OPTIMIZATION: Instead of running 1 episode at a time on CPU,
this runs 32 episodes in parallel on GPU using vectorized tensors.
All fire spread, thermal, observations, and neural network forward
passes happen on GPU simultaneously.

Expected speedup: 10-30x on dual T4 (8h → 20-50 min).

Usage in Kaggle notebook:
    from gpu_accelerated import gpu_run_full_pipeline
    gpu_run_full_pipeline()
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, json, os, math
from collections import defaultdict

# ─── Device Setup ───
if torch.cuda.is_available():
    device = torch.device("cuda")
    n_gpus = torch.cuda.device_count()
    print(f"GPU: {n_gpus}x {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB each", flush=True)
else:
    device = torch.device("cpu")
    print("WARNING: No GPU found, falling back to CPU", flush=True)

np.random.seed(42)
torch.manual_seed(42)

# Strength of the goal-steer action bias (see GoalBiasPolicy). 0.0 = pure
# learned goal-conditioning (the calibrated baseline); tuned per experiment.
GOAL_STEER_BETA = 0.0


# ═══════════════════════════════════════════════════════════════
# SECTION 1: GPU-VECTORIZED ENVIRONMENT
# ═══════════════════════════════════════════════════════════════

class GPUWildfireEnv:
    """Vectorized wildfire environment running N parallel episodes on GPU.
    
    All state (fire grid, thermal, wind, drone positions) is stored as
    torch tensors on GPU. Fire spread, thermal computation, and observation
    gathering are fully vectorized across all N environments.
    
    Memory budget (32 envs, 30x30 grid):
      fire: 32×30×30×4B = 115 KB
      fuel: 32×30×30×4B = 115 KB
      wind: 32×30×30×4B × 2 = 230 KB
      thermal: 32×30×30×4B = 115 KB
      drones: 32×10×2×4B × 2 = 5 KB
      Total: < 1 MB (tiny)
    """
    
    def __init__(self, n_envs=32, grid=30, n_drones=10, max_steps=300, 
                 wind_speed=0.0, obs_r=4):
        self.n_envs = n_envs
        self.grid = grid
        self.n_drones = n_drones
        self.max_steps = max_steps
        self.obs_r = obs_r
        self.obs_size = 2 * obs_r + 1  # 9
        self.obs_channels = 6  # fire, thermal, wind_x, wind_y, visited, fire_dist
        self.local_obs_dim = self.obs_channels * self.obs_size * self.obs_size
        # Global features: 10 scalar features + a small team-visited-map cue
        # (6×6 downsampled fraction of cells the swarm has explored — gives the
        # reactive policy the global memory it needs to sweep instead of re-scan).
        self.global_base_dim = 10
        self.map_cue_size = 6
        self.global_obs_dim = self.global_base_dim + self.map_cue_size ** 2
        self.obs_dim = self.local_obs_dim + self.global_obs_dim
        self.act_dim = 5
        self.base_wind = wind_speed
        
        # Action deltas: stay, right, left, up, down
        self.action_deltas = torch.tensor([[0,0],[0,1],[0,-1],[1,0],[-1,0]], 
                                           dtype=torch.float32, device=device)
        
        # Physics constants
        self.momentum = 0.7
        self.fire_crash_threshold = 0.3
        self.thermal_crash = 15.0
        self.boundary_margin = 1.0
        self.thermal_cap = 25.0
        self.spread_rate = 0.1
        self.wind_amplification = 0.05
        self.fuel_depletion_rate = 0.002
        self.base_fire_radius = 3
        self.fire_intensity_init = 0.8
        self.spotting_prob = 0.02
        
        # Pre-compute coordinate grids on GPU
        yy, xx = torch.meshgrid(
            torch.arange(grid, device=device, dtype=torch.float32),
            torch.arange(grid, device=device, dtype=torch.float32),
            indexing='ij'
        )
        self.yy = yy  # (grid, grid)
        self.xx = xx  # (grid, grid)
        
        # Observation patch indices (pre-computed)
        # For each position, we need to gather a (obs_size × obs_size) patch
        # with channel offsets
        self._precompute_obs_indices()
        
        # Allocate GPU state tensors
        self._alloc_tensors()
    
    def _precompute_obs_indices(self):
        """Pre-compute the flat indices for gathering observation patches."""
        G = self.grid
        r = self.obs_r
        os_ = self.obs_size
        ch = self.obs_channels
        
        # For each possible (ix, iy), compute the patch indices
        # We'll do this dynamically in reset since positions change
        pass  # We'll use vectorized gather in _get_obs instead
    
    def _alloc_tensors(self):
        """Allocate all GPU tensors for N parallel environments."""
        N, G, K = self.n_envs, self.grid, self.n_drones
        
        self.fire = torch.zeros(N, G, G, device=device)
        self.fuel = torch.zeros(N, G, G, device=device)
        self.wind_x = torch.zeros(N, G, G, device=device)
        self.wind_y = torch.zeros(N, G, G, device=device)
        self.thermal = torch.zeros(N, G, G, device=device)
        self.shared_visited = torch.zeros(N, G, G, device=device)
        self.fire_dist = torch.zeros(N, G, G, device=device)
        
        # Drone states
        self.drone_pos = torch.zeros(N, K, 2, device=device)
        self.drone_vel = torch.zeros(N, K, 2, device=device)
        self.drone_alive = torch.ones(N, K, dtype=torch.bool, device=device)
        
        # Tracking
        self.step_count = 0
        self.total_cells_explored = torch.zeros(N, G, G, dtype=torch.bool, device=device)
        self.episode_cells = torch.zeros(N, dtype=torch.float32, device=device)
    
    def reset(self):
        """Reset all N environments in parallel on GPU."""
        N, G, K = self.n_envs, self.grid, self.n_drones
        self.step_count = 0
        self.total_cells_explored.zero_()
        self.episode_cells.zero_()
        
        # Initialize fire: random center for each environment
        r = self.base_fire_radius
        margin = r + 3
        cx = torch.randint(margin, G - margin, (N,), device=device)
        cy = torch.randint(margin, G - margin, (N,), device=device)
        self.fire.zero_()
        
        # Place circular fire patches (vectorized)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx*dx + dy*dy <= r*r:
                    nx = cx + dx  # (N,)
                    ny = cy + dy  # (N,)
                    valid = (nx >= 0) & (nx < G) & (ny >= 0) & (ny < G)
                    # Use advanced indexing
                    env_idx = torch.arange(N, device=device)
                    mask = valid
                    self.fire[env_idx[mask], ny[mask], nx[mask]] = self.fire_intensity_init
        
        self.fire_center_x = cx.float()
        self.fire_center_y = cy.float()
        
        # Fuel
        self.fuel = torch.clip(
            0.8 - 0.3 * torch.randn(N, G, G, device=device), 0.3, 1.0
        ).float()
        
        # Wind field with gusts
        angle = torch.rand(N, device=device) * 2 * math.pi
        self.wind_x = self.base_wind * torch.cos(angle).view(N, 1, 1).expand(N, G, G)
        self.wind_y = self.base_wind * torch.sin(angle).view(N, 1, 1).expand(N, G, G)
        
        # Add gusts
        for k in range(3):
            freq = 0.5 * (k + 1)
            amp = 0.1 * self.base_wind
            phase = torch.rand(N, device=device) * 2 * math.pi
            self.wind_x = self.wind_x + amp * torch.sin(
                2 * math.pi * freq * self.xx.unsqueeze(0) / G + phase.view(N, 1, 1)
            )
            self.wind_y = self.wind_y + amp * torch.sin(
                2 * math.pi * freq * self.yy.unsqueeze(0) / G + phase.view(N, 1, 1) * 0.7
            )
        
        # Shared visited
        self.shared_visited.zero_()
        
        # Initialize drones: scattered across the grid
        for i in range(K):
            px = torch.rand(N, device=device) * (G - 4) + 2
            py = torch.rand(N, device=device) * (G - 4) + 2
            self.drone_pos[:, i, 0] = px
            self.drone_pos[:, i, 1] = py
            self.drone_vel[:, i].zero_()
            self.drone_alive[:, i] = True
        
        # Initialize visited cells
        px_int = self.drone_pos[:, :, 0].long().clamp(0, G-1)
        py_int = self.drone_pos[:, :, 1].long().clamp(0, G-1)
        for i in range(K):
            self.shared_visited[
                torch.arange(N, device=device), py_int[:, i], px_int[:, i]
            ] = 1.0
            self.total_cells_explored[
                torch.arange(N, device=device), py_int[:, i], px_int[:, i]
            ] = True
        
        self._update_thermal()
        self._update_fire_dist()
        
        return self._get_obs()
    
    def _update_thermal(self):
        """Vectorized thermal updraft computation on GPU."""
        fire_mask = (self.fire > 0.2).float()  # (N, G, G)
        fire_sum = fire_mask.sum(dim=(1, 2))  # (N,)
        
        # Initialize thermal to zero
        self.thermal.zero_()
        
        # For each fire cell, compute Gaussian contribution to all grid cells
        # Vectorized: compute distance from each cell to fire center of mass
        fire_intensity = self.fire * fire_mask  # (N, G, G)
        
        # Fire center of mass per environment
        fire_mass = fire_mask.sum(dim=(1, 2)).clamp(min=1)  # (N,)
        fire_cx = (fire_mask * self.xx.unsqueeze(0)).sum(dim=(1, 2)) / fire_mass  # (N,)
        fire_cy = (fire_mask * self.yy.unsqueeze(0)).sum(dim=(1, 2)) / fire_mass  # (N,)
        
        # Distance from each grid cell to fire center of mass
        dx = self.xx.unsqueeze(0) - fire_cx.view(-1, 1, 1)  # (N, G, G)
        dy = self.yy.unsqueeze(0) - fire_cy.view(-1, 1, 1)  # (N, G, G)
        dist_sq = dx*dx + dy*dy  # (N, G, G)
        
        # Thermal = sum of intensities × Gaussian falloff
        # Approximate: use total fire intensity × Gaussian from center
        total_intensity = fire_intensity.sum(dim=(1, 2))  # (N,)
        self.thermal = total_intensity.view(-1, 1, 1) * torch.exp(-dist_sq / 8.0)
        self.thermal = self.thermal.clamp(0, self.thermal_cap)
    
    def _update_fire_dist(self):
        """EXACT Euclidean distance to nearest fire cell, fully vectorized on GPU.
        
        Two-pass separable Euclidean distance transform (the standard
        Felzenszwalb–Huttenlocher decomposition), evaluated exactly over the
        small G×G grid with batched min-plus ops. Each cell gets the true
        Euclidean distance to the closest cell with fire > 0.2 in its own
        environment — same semantics as the original argwhere loop, but with
        no per-env Python loop and no GPU-CPU sync.
        """
        N, G = self.n_envs, self.grid
        fire_mask = (self.fire > 0.2)  # (N, G, G)
        self.fire_dist.fill_(10.0)
        
        has_fire = fire_mask.any(dim=(1, 2))  # (N,)
        if not has_fire.any():
            return
        
        # Squared offset matrix (i - j)^2 for targets i and sources j
        coords = torch.arange(G, device=device, dtype=torch.float32)
        d2 = (coords[:, None] - coords[None, :]) ** 2  # (G, G)
        INF = 1e8
        
        # ── Pass 1 along x (per row): gx[y,x] = min over fire cols j of (x-j)^2 ──
        rows = fire_mask[has_fire].reshape(-1, G)      # (Nf*G, G)
        cand = d2[None, :, :] + torch.where(rows[:, None, :], 0.0, INF)
        gx = cand.min(dim=2).values                    # (Nf*G, G); INF if row has no fire
        
        # ── Pass 2 along y: for each column x, d2[y,x] = min over rows k of (y-k)^2 + gx[k,x] ──
        Nf = int(has_fire.sum().item())
        # cols: one series per (env, column x) over row index k → shape (Nf*G, G)
        cols = gx.reshape(Nf, G, G).transpose(1, 2).reshape(Nf * G, G)
        cand2 = d2[None, :, :] + cols[:, None, :]      # (Nf*G, G_y, G_k)
        dist2 = cand2.min(dim=2).values                # (Nf*G, G): index (col x, row y)
        # Pack as (Nf, col x, row y), then transpose back to (Nf, row y, col x)
        dist2 = dist2.reshape(Nf, G, G).transpose(1, 2)
        
        self.fire_dist[has_fire] = torch.sqrt(dist2 + 1e-8)
    
    def _spread_fire(self):
        """Vectorized fire spread on GPU across all N environments."""
        fire_mask = (self.fire > 0.2)
        new_fire = self.fire.clone()
        
        # Get fire cell locations per environment
        neighbor_offsets = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        
        for dy, dx in neighbor_offsets:
            # Shift fire mask to neighbor positions
            # Source: fire cells; Target: neighbor cells
            src_fire = fire_mask.clone()
            
            # Compute neighbor positions with clamping
            if dy != 0:
                if dy > 0:
                    src_ny = slice(None, -1)
                    dst_ny = slice(1, None)
                else:
                    src_ny = slice(1, None)
                    dst_ny = slice(None, -1)
            else:
                src_ny = slice(None)
                dst_ny = slice(None)
            
            if dx != 0:
                if dx > 0:
                    src_nx = slice(None, -1)
                    dst_nx = slice(1, None)
                else:
                    src_nx = slice(1, None)
                    dst_nx = slice(None, -1)
            else:
                src_nx = slice(None)
                dst_nx = slice(None)
            
            # Source fire intensity at (src_y, src_x)
            src_intensity = self.fuel[:, src_ny, src_nx] * self.fire[:, src_ny, src_nx]
            wind_mag = torch.sqrt(
                self.wind_x[:, src_ny, src_nx]**2 + self.wind_y[:, src_ny, src_nx]**2
            )
            spread_prob = self.spread_rate * (1 + self.wind_amplification * wind_mag) * src_intensity
            
            # Random spread (use dropout as stochastic source)
            rand = torch.rand_like(spread_prob)
            spreads = (rand < spread_prob) & (self.fuel[:, dst_ny, dst_nx] > 0.1)
            
            # Apply spread
            new_fire[:, dst_ny, dst_nx] = new_fire[:, dst_ny, dst_nx] + spreads.float() * 0.1
        
        self.fire = new_fire.clamp(0, 1.0)
        
        # Fuel depletion
        self.fuel = torch.clamp(
            self.fuel - self.fuel_depletion_rate * fire_mask.float(), 0.0, 1.0
        )
    
    def step(self, actions):
        """Vectorized step across all N environments on GPU.
        
        Args:
            actions: (N, K) int tensor of actions for each drone in each env
        
        Returns:
            obs: (N, K, obs_dim) float tensor
            dones: (N, K) bool tensor
            crashed: (N, K) bool tensor
        """
        self.step_count += 1
        N, G, K = self.n_envs, self.grid, self.n_drones
        
        # Get action deltas for all drones
        action_deltas = self.action_deltas[actions]  # (N, K, 2)
        alive = self.drone_alive  # (N, K) — alive at start of this step
        alive_e = alive.unsqueeze(-1)  # (N, K, 1)
        
        # Update velocity with momentum — dead drones stay frozen (vel 0, pos unchanged),
        # matching the reference CPU env semantics (dead agents are skipped entirely).
        new_vel = self.momentum * self.drone_vel + (1 - self.momentum) * action_deltas
        new_vel = torch.where(alive_e, new_vel, torch.zeros_like(new_vel))
        self.drone_vel = new_vel
        
        # Update position (only alive drones move)
        new_pos = torch.where(alive_e, self.drone_pos + new_vel, self.drone_pos)
        new_pos = new_pos.clamp(self.boundary_margin, G - 1 - self.boundary_margin)
        self.drone_pos = new_pos
        
        # ── Mark visited cells (fully vectorized, no per-drone loop) ──
        px_int = new_pos[:, :, 0].long().clamp(0, G-1)  # (N, K)
        py_int = new_pos[:, :, 1].long().clamp(0, G-1)  # (N, K)
        alive = self.drone_alive  # (N, K)
        
        env_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, K)  # (N, K)
        env_flat = env_idx.reshape(-1)
        px_flat = px_int.reshape(-1)
        py_flat = py_int.reshape(-1)
        alive_flat = alive.reshape(-1)
        
        valid = alive_flat
        self.shared_visited[env_flat[valid], py_flat[valid], px_flat[valid]] = 1.0
        self.total_cells_explored[env_flat[valid], py_flat[valid], px_flat[valid]] = True
        
        # ── Check crashes (fully vectorized, no per-drone loop) ──
        ix_all = px_int.clamp(0, G-1)  # (N, K)
        iy_all = py_int.clamp(0, G-1)  # (N, K)
        
        fire_val = self.fire[env_idx, iy_all, ix_all]       # (N, K)
        fire_near = self.fire_dist[env_idx, iy_all, ix_all]  # (N, K)
        thermal_val = self.thermal[env_idx, iy_all, ix_all]  # (N, K)
        
        crash_mask = alive & (
            (fire_val > self.fire_crash_threshold) |
            (fire_near < 0.5) |
            (thermal_val > self.thermal_crash)
        )
        
        crashed = crash_mask.clone()
        self.drone_alive = self.drone_alive & ~crash_mask
        dones = crash_mask | ~alive
        
        # Fire spread every 3 steps
        if self.step_count % 3 == 0:
            self._spread_fire()
            self._update_fire_dist()
            self._update_thermal()
        
        # Episode end
        if self.step_count >= self.max_steps:
            dones[:] = True
        
        # Count explored cells per env
        self.episode_cells = self.total_cells_explored.sum(dim=(1, 2)).float()
        
        obs = self._get_obs()
        
        return obs, dones, crashed
    
    def _get_obs(self):
        """Fully vectorized observation gathering on GPU.
        
        Extracts local patches and computes global features for all N×K agents
        in a single batched operation — no per-env or per-drone Python loops.
        
        Returns:
            obs: (N, K, obs_dim) tensor
        """
        N, G, K = self.n_envs, self.grid, self.n_drones
        r = self.obs_r
        os_ = self.obs_size
        ch = self.obs_channels
        local_dim = ch * os_ * os_
        
        obs = torch.zeros(N, K, self.obs_dim, device=device)
        
        # Channel stack: fire, thermal, wind_x, wind_y, visited, fire_dist
        fire_dist_norm = (self.fire_dist / 10.0).clamp(0, 1.0)
        channels = torch.stack([self.fire, self.thermal, self.wind_x,
                                self.wind_y, self.shared_visited, fire_dist_norm], dim=1)
        # channels: (N, ch, G, G)
        
        # Pad channels with zeros for boundary handling
        channels_padded = F.pad(channels, (r, r, r, r))  # (N, ch, G+2r, G+2r)
        padded_G = G + 2 * r
        
        # All drone positions (N, K)
        px = self.drone_pos[:, :, 0].long().clamp(0, G - 1)  # (N, K)
        py = self.drone_pos[:, :, 1].long().clamp(0, G - 1)  # (N, K)
        
        # Positions in padded coords — the original code indexes padded grid
        # directly with original position (not +r), so the patch [iy:iy+2r+1]
        # includes r cells of zero-padding to the left/top.
        px_pad = px  # (N, K) — use original coords as padded-grid indices
        py_pad = py  # (N, K)
        
        # Create offset grid: (os_, os_)
        oy, ox = torch.meshgrid(
            torch.arange(os_, device=device),
            torch.arange(os_, device=device),
            indexing='ij'
        )
        
        # Compute absolute positions in padded grid for each drone's obs patch
        # Shape: (N, K, os_, os_)
        abs_y = py_pad.unsqueeze(-1).unsqueeze(-1) + oy.unsqueeze(0).unsqueeze(0)
        abs_x = px_pad.unsqueeze(-1).unsqueeze(-1) + ox.unsqueeze(0).unsqueeze(0)
        
        # Flatten to (N, K, os_*os_)
        flat_pos = (abs_y * padded_G + abs_x).reshape(N, K, os_ * os_)
        
        # Gather patches from all channels at once
        flat_channels = channels_padded.reshape(N, ch, -1)  # (N, ch, padded_G^2)
        # Expand to (N, K, ch, padded_G^2) and (N, K, ch, os_*os_)
        flat_channels_exp = flat_channels.unsqueeze(1).expand(N, K, ch, -1)  # (N, K, ch, padded_G^2)
        flat_pos_exp = flat_pos.unsqueeze(2).expand(N, K, ch, os_ * os_)     # (N, K, ch, os_*os_)
        patches = torch.gather(flat_channels_exp, 3, flat_pos_exp)           # (N, K, ch, os_*os_)
        
        # Reshape to (N, K, ch * os_ * os_)
        local_obs = patches.reshape(N, K, local_dim)
        obs[:, :, :local_dim] = local_obs
        
        # ── Global features (all vectorized) ──
        alive = self.drone_alive  # (N, K)
        env_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, K)  # (N, K)
        px_c = px.clamp(0, G-1)
        py_c = py.clamp(0, G-1)
        
        obs[:, :, local_dim + 0] = px.float() / G
        obs[:, :, local_dim + 1] = py.float() / G
        obs[:, :, local_dim + 2] = self.drone_vel[:, :, 0]
        obs[:, :, local_dim + 3] = self.drone_vel[:, :, 1]
        
        # Fire/thermal/wind at drone position: (N, K)
        obs[:, :, local_dim + 4] = self.fire[env_idx, py_c, px_c]
        obs[:, :, local_dim + 5] = self.thermal[env_idx, py_c, px_c] / self.thermal_cap
        
        wx_at = self.wind_x[env_idx, py_c, px_c]
        wy_at = self.wind_y[env_idx, py_c, px_c]
        obs[:, :, local_dim + 6] = torch.sqrt(wx_at**2 + wy_at**2) / 30.0
        obs[:, :, local_dim + 7] = torch.atan2(wy_at, wx_at) / math.pi
        
        # Coverage (same for all drones in same env)
        obs[:, :, local_dim + 8] = self.episode_cells.unsqueeze(1).expand(N, K) / (G * G)
        
        # Fire radius approximation (same for all drones in same env)
        fire_cells = (self.fire > 0.2).sum(dim=(1, 2)).float()  # (N,)
        fr = torch.sqrt(fire_cells.clamp(min=1)) / G
        obs[:, :, local_dim + 9] = fr.unsqueeze(1).expand(N, K)
        
        # ── Team visited-map cue (global memory) ──
        # 6×6 adaptive-average pool of the shared visited map: each entry is the
        # fraction of cells explored in that region. Agents combine this with
        # their own normalized position (global dims 0-1) to steer toward
        # unexplored regions instead of re-scanning visited ones.
        map_dim = self.map_cue_size ** 2
        vis_map = F.adaptive_avg_pool2d(self.shared_visited.unsqueeze(1),  # (N,1,G,G)
                                        (self.map_cue_size, self.map_cue_size))
        vis_map = vis_map.reshape(N, map_dim)  # (N, 36)
        obs[:, :, local_dim + self.global_base_dim:local_dim + self.global_obs_dim] = \
            vis_map.unsqueeze(1).expand(N, K, map_dim)
        
        # Zero out dead agents
        obs[~alive] = 0.0
        
        return obs


# ═══════════════════════════════════════════════════════════════
# SECTION 2: REUSE NETWORKS FROM kaggle_full_run.py
# ═══════════════════════════════════════════════════════════════

# Import the network classes from the main file
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_full_run import (
    MultiHeadAttention, GATCommunication, NoGATCommunication,
    PPONetwork, MAPPOCritic, GPFireFront, NeuralCBFFilter, compute_reward,
    train
)


# ═══════════════════════════════════════════════════════════════
# SECTION 3: BATCHED AGENT
# ═══════════════════════════════════════════════════════════════

class GoalBiasPolicy(nn.Module):
    """PPONetwork + a fixed steering prior over the 5 discrete actions.

    The last 4 dims of the input are the waypoint embedding (cy, cx, dy, dx)
    where (dy, dx) = clamp(waypoint_center - own_pos, -1, 1) in map-normalized
    units. The policy's action logits get a non-learned additive term
    beta * dot(action_delta, toward), so movement is biased toward the chosen
    frontier sector. The learned net sits on top and can override the bias
    (avoid fire, crowding, etc.) — but goal-following no longer depends on PPO
    discovering it through diffuse exploration credit (which it never did).

    forward/evaluate recompute the SAME bias from the stored input, so the
    log-probs used at selection time and at PPO update time stay consistent.
    """

    def __init__(self, net, steer_beta=None, act_deltas=None):
        if steer_beta is None:
            steer_beta = GOAL_STEER_BETA
        super().__init__()
        self.net = net
        self.steer_beta = steer_beta
        # 5 discrete actions, order = env.action_deltas: 0 = stay,
        # 1 = (0,+1), 2 = (0,-1), 3 = (+1,0), 4 = (-1,0) in the same
        # (x=col, y=row) frame the env applies to drone_pos.
        if act_deltas is None:
            act_deltas = [[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]]
        self.register_buffer('_deltas', torch.tensor(act_deltas, dtype=torch.float32))

    def _steer(self, obs):
        # obs: (..., n) with n >= 4; last 4 dims = (cy, cx, dy, dx)
        embed = obs[..., -4:]
        dx = embed[..., 3]            # toward waypoint, x (col) axis
        dy = embed[..., 2]            # toward waypoint, y (row) axis
        norm = torch.sqrt(dx * dx + dy * dy).clamp(min=1e-3)
        ux = dx / norm
        uy = dy / norm
        # (…, 5) dot products of each action delta with the unit toward vector
        steer = (self._deltas[:, 0] * ux.unsqueeze(-1)
                 + self._deltas[:, 1] * uy.unsqueeze(-1))
        return self.steer_beta * steer

    def forward(self, obs):
        logits, values = self.net(obs)
        return logits + self._steer(obs), values

    def evaluate(self, obs, actions):
        logits, values = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return logits, dist.log_prob(actions), dist.entropy(), values


class BatchedGATPPO:
    """PPO agent that processes all N environments × K agents in batched fashion.
    
    Key optimization: instead of calling select_actions() K times per env,
    we stack all observations and do a single batched forward pass through
    GAT and policy networks.
    """
    
    def __init__(self, obs_dim, act_dim=5, use_gat=True, lr=2e-4, comm_range=10.0,
                 map_cue_size=6, global_base_dim=10):
        if use_gat:
            self.gat = GATCommunication(obs_dim, hidden_dim=128, out_dim=64, comm_range=comm_range).to(device)
        else:
            self.gat = NoGATCommunication(obs_dim, out_dim=64).to(device)
        
        # ── Frontier waypoint head (goal-conditioned exploration) ──
        # A small head over the raw map cue + own position emits logits over the
        # 6×6 sectors of the team-visited map. Every WP_INTERVAL steps each drone
        # samples a target sector (its "waypoint"); the actor's policy input is
        # augmented with that target (center + offset), so movement becomes
        # "head toward region (i,j)" instead of a raw local reactive scan. The
        # waypoint head is trained with REINFORCE on the drone's own
        # new-cell credit (its block return), so it learns to pick the sectors
        # where exploration actually pays — the mechanism random wandering
        # exploits implicitly and a memoryless reactive policy lacks.
        self.map_cue_size = map_cue_size
        self.global_base_dim = global_base_dim
        self.cue_dim = map_cue_size * map_cue_size
        # obs layout (built in GPUWildfireEnv._get_obs): local patch, then
        # global_base_dim global features (own pos at +0/+1), then the cue.
        self._glob_start = obs_dim - (global_base_dim + self.cue_dim)
        wp_in_dim = 2 + self.cue_dim            # pos_norm + cue
        self.wp_net = nn.Sequential(
            nn.Linear(wp_in_dim, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, self.cue_dim)).to(device)
        self.wp_interval = 25                    # steps a waypoint is held
        self.wp_entropy_coef = 0.02
        self.wp_prior = 3.0                      # init bias toward unvisited sectors
        # 4 extra policy-input dims: waypoint center + normalized offset from own pos
        self.policy = GoalBiasPolicy(
            PPONetwork(self.gat.enhanced_obs_dim + 4, act_dim)).to(device)
        params = (list(self.gat.parameters()) + list(self.policy.parameters())
                  + list(self.wp_net.parameters()))
        self.optimizer = torch.optim.Adam(params, lr=lr, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=2000, eta_min=1e-5)
        # Tensor buffers for fast storage (avoids per-element dict creation)
        self._obs_buf = []
        self._act_buf = []
        self._rew_buf = []
        self._done_buf = []
        self._lp_buf = []
        self._val_buf = []
        self._aid_buf = []
        # Waypoint buffers (one row per (env, agent) per step, like the others)
        self._wp_in_buf = []       # wp-net input at each step
        self._wp_act_buf = []      # sector id governing each step (-1 = dead/none)
        self._wp_rew_buf = []      # per-step own-new-cell credit (training only)
        self._wp_act_flat = None   # current waypoint per flattened row
        self._wp_step = 0          # steps since begin_rollout()
        self._old_format_policy = False   # True after loading a pre-waypoint ckpt
        self.itse_weight = 0.3
        self.explore_bonus_scale = 2.0

    def begin_rollout(self):
        """Reset waypoint state at the start of a fresh episode batch."""
        self._wp_act_flat = None
        self._wp_step = 0
        self._wp_in_buf.clear()
        self._wp_act_buf.clear()
        self._wp_rew_buf.clear()

    def _select_waypoints(self, wp_in, alive_flat):
        """Sample a fresh target sector for each alive agent (others get -1)."""
        with torch.no_grad():
            logits = self.wp_net(wp_in)
            # Init-bias: start from "prefer unvisited sectors"; the net learns on top.
            logits = logits - self.wp_prior * wp_in[:, -self.cue_dim:]
            dist = torch.distributions.Categorical(logits=logits)
            wp = dist.sample()
            wp[~alive_flat] = -1
            self._wp_act_flat = wp

    def _waypoint_embedding(self, n_rows):
        """(center_y, center_x, dy, dx) of the current waypoint, in map-normalized units."""
        idx = self._wp_act_flat.clamp(min=0)
        sy = (idx // self.map_cue_size).float()
        sx = (idx % self.map_cue_size).float()
        cy = (sy + 0.5) / self.map_cue_size
        cx = (sx + 0.5) / self.map_cue_size
        pos = self._wp_pos_flat                       # (n, 2) normalized own pos
        dy = (cy - pos[:, 0]).clamp(-1.0, 1.0)
        dx = (cx - pos[:, 1]).clamp(-1.0, 1.0)
        return torch.stack([cy, cx, dy, dx], dim=-1)  # (n, 4)

    def select_actions_batched(self, obs, positions, alive_mask, greedy=False):
        """Batched action selection across all envs and agents.
        
        Args:
            obs: (N, K, obs_dim) tensor
            positions: (N, K, 2) tensor
            alive_mask: (N, K) bool tensor
            greedy: if True, take the mode (argmax) action — used for
                deterministic evaluation/reporting, never during training.
        
        Returns:
            actions: (N, K) int tensor
            log_probs: (N, K) float tensor
            values: (N, K) float tensor
            enhanced_obs: (N, K, enhanced_dim) tensor
        """
        N, K, obs_dim = obs.shape
        
        # Flatten to (N*K, obs_dim) for batch processing
        obs_flat = obs.reshape(N * K, obs_dim)
        pos_flat = positions.reshape(N * K, 2)
        alive_flat = alive_mask.reshape(N * K)
        
        # GAT forward pass (batched). Pass env group ids so attention edges are
        # confined to agents inside the same environment: the N parallel envs
        # are independent episodes and must not communicate with each other.
        if isinstance(self.gat, GATCommunication):
            group = torch.arange(N, device=device).repeat_interleave(K)
            enhanced = self.gat(obs_flat, pos_flat, alive_flat, group=group)
        else:
            enhanced = self.gat(obs_flat, pos_flat, alive_flat)
        
        # ── Waypoint selection & goal conditioning ──
        if self._old_format_policy:
            # Legacy (pre-waypoint) checkpoint: the policy net expects the raw
            # enhanced obs with no +4 goal-embedding dims, and we skip waypoint
            # sampling entirely (no extra RNG draws), so replayed evaluation of
            # an old checkpoint is bit-identical to what the old code produced.
            pass
        else:
            wp_in = torch.cat([obs_flat[..., self._glob_start:self._glob_start + 2],
                               obs_flat[..., -self.cue_dim:]], dim=-1).detach()   # (n, 2+cue)
            self._wp_pos_flat = obs_flat[..., self._glob_start:self._glob_start + 2].detach()
            if ((self._wp_step % self.wp_interval == 0)
                    or self._wp_act_flat is None
                    or self._wp_act_flat.shape[0] != N * K):
                self._select_waypoints(wp_in, alive_flat)
            self._wp_step += 1
            # Record (input, governing sector) per step for the waypoint update.
            self._wp_in_buf.append(wp_in)
            self._wp_act_buf.append(self._wp_act_flat.clone())
            # Augment the policy input with the chosen target so movement can be
            # goal-directed ("head toward sector" rather than a blind local scan).
            enhanced = torch.cat([enhanced, self._waypoint_embedding(N * K)], dim=-1)
        
        # Policy forward pass
        with torch.no_grad():
            logits, values = self.policy(enhanced)
            dist = torch.distributions.Categorical(logits=logits)
            if greedy:
                # Deterministic (mode) actions for evaluation/reporting — a
                # published number must not depend on sampling noise.
                actions = logits.argmax(dim=-1)
                log_probs = dist.log_prob(actions)
            else:
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
        
        # Mask dead agents
        actions = actions.reshape(N, K)
        log_probs = log_probs.reshape(N, K)
        values = values.reshape(N, K)
        enhanced_obs = enhanced.reshape(N, K, -1)
        
        # Set dead agent actions to 0 (hover)
        dead = ~alive_mask
        actions[dead] = 0
        log_probs[dead] = 0
        values[dead] = 0
        
        return actions, log_probs, values, enhanced_obs
    
    def store_batched(self, enhanced_obs, actions, rewards, dones, log_probs, values,
                      waypoint_reward=None):
        """Store transitions from all N environments using on-device tensor buffers.
        
        Replaces per-element dict creation with batch tensor ops and avoids a
        per-step GPU→CPU round trip (update() concatenates straight from the buffer).
        Each stored row is one (env, agent) transition; rows are stacked per step.
        waypoint_reward: optional per-agent new-cell credit used by the waypoint
        head's REINFORCE update (training only).
        """
        # Flatten N×K to a single batch dim (row = env * K + agent)
        self._obs_buf.append(enhanced_obs.detach().reshape(-1, enhanced_obs.shape[-1]))
        self._act_buf.append(actions.reshape(-1))
        self._rew_buf.append(rewards.reshape(-1))
        self._done_buf.append(dones.reshape(-1))
        self._lp_buf.append(log_probs.reshape(-1))
        self._val_buf.append(values.reshape(-1))
        if waypoint_reward is not None:
            self._wp_rew_buf.append(waypoint_reward.reshape(-1).detach())
    
    def _wp_loss(self):
        """REINFORCE-with-baseline loss for the waypoint head.

        Blocks = maximal runs of steps governed by one sampled sector. Each block's
        return is the agent's own new-cell credit over those steps; advantage is the
        block return minus that agent's mean block return for the rollout. The head
        therefore learns to choose the sectors whose pursuit actually yields fresh
        ground. Only alive-governed blocks contribute (dead rows carry sector -1).
        """
        if not self._wp_act_buf or not self._wp_rew_buf:
            return None
        S = len(self._wp_act_buf)
        wp_in_all = torch.cat(self._wp_in_buf, dim=0)     # (S*M, wp_in_dim)
        wp_act_all = torch.cat(self._wp_act_buf, dim=0)   # (S*M,)
        wp_rew_all = torch.cat(self._wp_rew_buf, dim=0)   # (S*M,)
        n = wp_act_all.shape[0]
        M = n // S
        A = wp_act_all.reshape(M, S)
        Rw = wp_rew_all.reshape(M, S)
        self._wp_in_buf.clear()
        self._wp_act_buf.clear()
        self._wp_rew_buf.clear()
        in_rows, acts, advs = [], [], []
        for r in range(M):
            ar = A[r]
            if int(ar[0].item()) < 0:
                continue                              # row dead (or never selected)
            change = (ar[1:] != ar[:-1]).nonzero(as_tuple=False).flatten()
            bnd = [0] + (change + 1).tolist() + [S]
            row_in, row_act, row_ret = [], [], []
            for b in range(len(bnd) - 1):
                s0, s1 = bnd[b], bnd[b + 1]
                a = int(ar[s0].item())
                if a < 0:
                    continue
                row_in.append(wp_in_all[r * S + s0])
                row_act.append(a)
                row_ret.append(Rw[r, s0:s1].sum())
            if not row_ret:
                continue
            rets = torch.stack(row_ret)
            in_rows.append(torch.stack(row_in))
            acts.append(torch.tensor(row_act, device=device))
            advs.append(rets - rets.mean())
        if not acts:
            return None
        X = torch.cat(in_rows)
        Ac = torch.cat(acts)
        Ad = torch.cat(advs)
        logits = self.wp_net(X)
        dist = torch.distributions.Categorical(logits=logits)
        lp = dist.log_prob(Ac)
        ent = dist.entropy().mean()
        return -(lp * Ad).mean() - self.wp_entropy_coef * ent

    def update(self, gamma=0.99, gae_lambda=0.95, n_epochs=3, batch_size=2048, clip_eps=0.2, entropy_coef=0.02):
        """PPO update using on-device tensor buffers.
        
        GAE is computed per (env, agent) rollout: the buffer rows are stacked per
        step, so each row is one agent in one environment over time. Advantages are
        therefore bootstraped only from that agent's own environment — NOT chained
        across the independent parallel environments (which injected cross-env
        value noise into every advantage and destabilized training). Fully
        vectorized, no per-transition .item() / GPU-CPU sync.

        The frontier waypoint head is updated in the same call (REINFORCE on
        per-block exploration credit — see _wp_loss).
        """
        if not self._obs_buf:
            return 0.0
        wp_loss = self._wp_loss()
        
        S = len(self._obs_buf)          # rollout steps
        # Per-step stacked rows, step-major (used for the minibatch training)
        all_obs = torch.cat(self._obs_buf, dim=0)
        all_actions = torch.cat(self._act_buf, dim=0).long()
        all_old_lp = torch.cat(self._lp_buf, dim=0)
        all_values = torch.cat(self._val_buf, dim=0)
        n = all_obs.shape[0]
        M = n // S
        
        # Reshape per-step buffers into (M rows, S steps) per-row timelines.
        # Row r = (env r//K, agent r%K) stays constant across steps, so a GAE over
        # its own timeline is a correct per-environment rollout.
        def to_mat(buf):
            return torch.stack(buf, dim=1)          # (M, S)
        R = to_mat(self._rew_buf)
        V = to_mat(self._val_buf)
        D = to_mat(self._done_buf)
        
        # Clear buffers
        self._obs_buf.clear()
        self._act_buf.clear()
        self._rew_buf.clear()
        self._done_buf.clear()
        self._lp_buf.clear()
        self._val_buf.clear()
        
        # delta_t = R_t + gamma * V_{t+1} * (1 - D_{t+1}) - V_t   (past episode end: V=0, D=1)
        Vn = torch.zeros_like(V)
        Dn = torch.ones_like(D)
        Vn[:, :-1] = V[:, 1:]
        Dn[:, :-1] = D[:, 1:]
        delta = R + gamma * Vn * (1 - Dn) - V
        # decay m_t = gamma*lambda*(1 - D_{t+1}); reversed recurrence a = x + r*a_prev
        c = gamma * gae_lambda
        m = c * (1 - Dn)
        x = torch.flip(delta, dims=[1])
        r = torch.flip(m, dims=[1])
        # Masked-scan closed form along each row (S ≤ 300 ⇒ c**(±u) stays in fp32):
        # a_j = x_j + r_j a_{j-1}, where r_j ∈ {0, c} (r=0 at episode/agent boundaries).
        # With u = steps since the last boundary, a_j = c^u * cumsum_{run}(x * c^-u).
        ar = torch.arange(S, device=device, dtype=torch.float32).unsqueeze(0)  # (1, S)
        z = r <= 0.0                                                            # (M, S)
        lastz = torch.cummax(torch.where(z, ar.expand(M, S), ar.new_full((), -1.0)), dim=1).values
        u = ar - lastz.clamp(min=0.0)                                            # (M, S)
        inv = torch.pow(c, -u)
        scl = torch.pow(c, u)
        pref = torch.cumsum(x * inv, dim=1)
        idx = (lastz - 1.0).long().clamp(min=0)
        off = torch.where(lastz >= 1.0, torch.gather(pref, 1, idx), torch.zeros_like(pref))
        y = scl * (pref - off)
        adv_mat = torch.flip(y, dims=[1])                                        # (M, S)
        # Back to step-major ordering (matches all_obs / all_values)
        advantages = adv_mat.transpose(0, 1).reshape(-1)
        returns = advantages + all_values
        all_adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss, count = 0.0, 0
        all_params = list(self.gat.parameters()) + list(self.policy.parameters()) + list(self.wp_net.parameters())
        
        for _ in range(n_epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                _, new_lp, entropy, vals = self.policy.evaluate(all_obs[idx], all_actions[idx])
                ratio = torch.exp(new_lp - all_old_lp[idx])
                s1 = ratio * all_adv[idx]
                s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * all_adv[idx]
                loss = -torch.min(s1, s2).mean() + 0.5 * F.mse_loss(vals, returns[idx]) - entropy_coef * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, 0.5)
                self.optimizer.step()
                total_loss += loss.item()
                count += 1
        
        # Waypoint-head update step (separate gradients; disjoint params)
        if wp_loss is not None:
            self.optimizer.zero_grad()
            wp_loss.backward()
            nn.utils.clip_grad_norm_(self.wp_net.parameters(), 0.5)
            self.optimizer.step()
        
        self.scheduler.step()
        return total_loss / max(1, count)
    
    def save(self, path):
        torch.save({'gat': self.gat.state_dict(), 'policy': self.policy.state_dict(),
                    'wp_net': self.wp_net.state_dict()}, path)
    
    def load(self, path):
        """Load a checkpoint, tolerating both current and pre-waypoint formats.

        Current checkpoints carry the GoalBiasPolicy under 'net.*' plus 'wp_net'.
        Legacy checkpoints (committed 6798a2d and earlier, e.g. from a Kaggle run
        started before the waypoint head existed) carry a plain PPONetwork with the
        raw 596-dim enhanced obs and no wp_net. Those are detected by the missing
        'net.' prefix, rebuilt at the legacy input width, and evaluated exactly as
        the old code would (waypoint handling disabled — see select_actions_batched).
        """
        ckpt = torch.load(path, map_location=device)
        self.gat.load_state_dict(ckpt['gat'])
        pol = ckpt['policy']
        if any(k.startswith('net.') for k in pol):
            # current format (GoalBiasPolicy wrapping PPONetwork)
            self.policy.load_state_dict(pol)
            if 'wp_net' in ckpt:
                self.wp_net.load_state_dict(ckpt['wp_net'])
            self._old_format_policy = False
        else:
            # legacy format: plain PPONetwork, no goal-conditioning dims
            act_dim = int(pol['policy_head.weight'].shape[0])
            inner = PPONetwork(self.gat.enhanced_obs_dim, act_dim).to(device)
            inner.load_state_dict(pol)
            self.policy = inner
            self._old_format_policy = True


# ────────────────────────────────────────────────────────────────────
# FRESH-EPISODE EVALUATION — ONE environment for ALL published rows
#
# batched_train() and gpu_run_full_pipeline() publish numbers from FRESH
# episodes run in the SAME batched env the agent trained in — never
# train-time logged coverage and never a second, independently-implemented
# environment. Random / Greedy / PID baselines are evaluated here too, with
# the identical action envelope (5 discrete actions, same physics) as the
# trained policies, so every row of a comparison table is comparable.
# ────────────────────────────────────────────────────────────────────

def _run_batched_rollouts(action_fn, grid=30, n_drones=10, max_steps=300,
                          n_envs=32, n_eps=None, wind=0.0, seed=None,
                          return_lists=False):
    """Run n_eps fresh episodes in the batched env using action_fn(o, env).

    Every (env, drone) resets from the same seeded stream, so calling with the
    same seed reproduces the same initial conditions for every policy.

    Returns (mean_cov, std_cov, mean_safe, std_safe): coverage = fraction of
    grid cells the swarm visited by episode end; survival = fraction of drones
    alive at episode end (per-episode, pooled).
    """
    if n_eps is None:
        n_eps = n_envs
    covs, safes = [], []
    eps_done = 0
    while eps_done < n_eps:
        b = min(n_envs, n_eps - eps_done)
        if seed is not None:
            torch.manual_seed(seed + eps_done)
        env = GPUWildfireEnv(n_envs=b, grid=grid, n_drones=n_drones,
                             max_steps=max_steps, wind_speed=wind)
        o = env.reset()
        rh = getattr(action_fn, 'reset', None)
        if rh is not None:
            rh()
        for _ in range(max_steps):
            if not env.drone_alive.any():
                break
            acts = action_fn(o, env)
            o, d, _ = env.step(acts)
            if d.all():
                break
        covs.extend((env.total_cells_explored.sum(dim=(1, 2)).float()
                     / (grid * grid) * 100).cpu().tolist())
        safes.extend((env.drone_alive.sum(dim=1).float()
                      / n_drones * 100).cpu().tolist())
        eps_done += b
    if return_lists:
        return covs, safes
    return (float(np.mean(covs)), float(np.std(covs)),
            float(np.mean(safes)), float(np.std(safes)))


def _random_actions(o, env):
    """Uniform-random actions over the same 5-action envelope as the policies."""
    a = torch.randint(0, 5, (env.n_envs, env.n_drones), device=device)
    a[~env.drone_alive] = 0
    return a


def _greedy_baseline_actions(o, env):
    """Move toward the best (unvisited, near-fire) neighbor cell (per drone)."""
    N, G, K = env.n_envs, env.grid, env.n_drones
    ix = env.drone_pos[:, :, 0].long().clamp(0, G - 1)   # (N, K)
    iy = env.drone_pos[:, :, 1].long().clamp(0, G - 1)
    dd = env.action_deltas.long()                         # (5, 2)
    nx_raw = ix.unsqueeze(-1) + dd[:, 0].view(1, 1, 5)   # (N, K, 5)
    ny_raw = iy.unsqueeze(-1) + dd[:, 1].view(1, 1, 5)
    nx = nx_raw.clamp(0, G - 1)
    ny = ny_raw.clamp(0, G - 1)
    env_i = torch.arange(N, device=device).view(N, 1, 1).expand(N, K, 5)
    vis = env.shared_visited[env_i, ny, nx]               # 1.0 if already visited
    fd = env.fire_dist[env_i, ny, nx]
    ok = ((nx_raw >= 0) & (nx_raw < G) & (ny_raw >= 0) & (ny_raw < G)
          & env.drone_alive.unsqueeze(-1))
    v = (vis < 0.5).float() + 2.0 / (fd + 1.0)
    a = torch.where(ok, v, torch.full_like(v, -1e9)).argmax(dim=-1)
    a[~env.drone_alive] = 0
    return a


def _pid_baseline_actions(o, env):
    """Orbit the fire center perpendicular-to-radius (mirrors CPU eval_pid)."""
    N, K = env.n_envs, env.n_drones
    dx = env.drone_pos[:, :, 0] - env.fire_center_x.view(N, 1)
    dy = env.drone_pos[:, :, 1] - env.fire_center_y.view(N, 1)
    take_x = dx.abs() > dy.abs()
    three = torch.full((N, K), 3, dtype=torch.long, device=device)
    four = torch.full((N, K), 4, dtype=torch.long, device=device)
    one = torch.full((N, K), 1, dtype=torch.long, device=device)
    two = torch.full((N, K), 2, dtype=torch.long, device=device)
    a = torch.where(take_x, torch.where(dx < 0, three, four),
                    torch.where(dy < 0, one, two))
    a[~env.drone_alive] = 0
    return a


def _frontier_oracle_actions(o, env):
    """Adaptive-frontier replanner oracle (perfect information).

    Every step each drone targets the nearest cell that is (unvisited AND safe:
    fire_dist >= 1.2, thermal below cap) inside its own row band (deconfliction),
    falling back to the global nearest safe unvisited cell when its band is
    exhausted. Moves by hill-climbing the Manhattan distance to that target with
    the 5 discrete actions, never landing in a dangerous cell (retreats along the
    least-dangerous step if surrounded).

    This is the strong non-learning baseline that defines the environment's
    coverage-safety ceiling: it maximizes coverage subject to safety, so it is
    the bar a learned swarm must approach.
    """
    N, G, K = env.n_envs, env.grid, env.n_drones
    band = max(1, G // K)
    xs = torch.arange(G, device=device).float()
    row_idx = torch.arange(G, device=device).view(1, 1, G, 1)
    visited = env.shared_visited.unsqueeze(1)                      # (N,1,G,G)
    safe = (visited < 0.5) & (env.fire_dist.unsqueeze(1) >= 1.2) \
        & (env.thermal.unsqueeze(1) < env.thermal_cap * 0.95)
    kk = torch.arange(K, device=device).view(1, K, 1, 1)
    band_rows = (row_idx >= kk * band) & (row_idx < kk * band + band)
    band_safe = safe & band_rows
    px = env.drone_pos[:, :, 0].clamp(0, G - 1)
    py = env.drone_pos[:, :, 1].clamp(0, G - 1)
    dx = (px.view(N, K, 1, 1) - xs.view(1, 1, 1, G)).abs()
    dy = (py.view(N, K, 1, 1) - row_idx.float().squeeze(-1).view(1, 1, G, 1)).abs()
    dist = dx + dy                                                 # (N,K,G,G)
    use_band = band_safe.any(dim=(-2, -1), keepdim=True)
    mask = torch.where(use_band, band_safe, safe)
    d_masked = torch.where(mask, dist, torch.full_like(dist, float('inf')))
    targ = d_masked.reshape(N, K, -1).argmin(dim=-1)
    tx = (targ % G).long(); ty = (targ // G).long()
    ix = px.long(); iy = py.long()
    dd = env.action_deltas.float()
    lx = ix.unsqueeze(-1) + dd[:, 0].view(1, 1, 5)
    ly = iy.unsqueeze(-1) + dd[:, 1].view(1, 1, 5)
    ok = (lx >= 0) & (lx < G) & (ly >= 0) & (ly < G) & env.drone_alive.unsqueeze(-1)
    lx_c = lx.clamp(0, G - 1).long(); ly_c = ly.clamp(0, G - 1).long()
    ei = torch.arange(N, device=device).view(N, 1, 1).expand(N, K, 5)
    lfd = env.fire_dist[ei, ly_c, lx_c]
    lth = env.thermal[ei, ly_c, lx_c]
    land_safe = ok & (lfd >= 1.2) & (lth < env.thermal_cap * 0.95)
    stay = torch.zeros(N, K, 1, dtype=torch.bool, device=device)
    land_safe = land_safe | (stay & env.drone_alive.unsqueeze(-1))
    cur = (px - tx.float()).abs() + (py - ty.float()).abs()
    nd = ((ix.unsqueeze(-1) + dd[:, 0].long().view(1, 1, 5)) - tx.unsqueeze(-1)).float().abs() \
         + ((iy.unsqueeze(-1) + dd[:, 1].long().view(1, 1, 5)) - ty.unsqueeze(-1)).float().abs()
    score = torch.where(land_safe, cur.unsqueeze(-1) - nd, torch.full_like(nd, -1e9))
    a = score.argmax(dim=-1)
    no_safe = ~land_safe.any(dim=-1)
    fallback = torch.where(ok, lfd, torch.full_like(lfd, 1e9)).argmin(dim=-1)
    a = torch.where(no_safe, fallback, a)
    a[~env.drone_alive] = 0
    return a


def _trained_agent_actions(agent, greedy=False):
    """Action fn for a trained agent checkpoint.

    Default (greedy=False): SAMPLE from the policy's action distribution — the
    same mechanism used in training/deployment, so the published number is the
    expected coverage/safety of the policy the agent actually implements.
    (Mode/argmax actions collapse to near-stasis for this reactive policy and
    would misrepresent it.) Rollouts are seeded, so results stay reproducible.
    greedy=True is available for an explicit mode-action measurement.
    """
    def fn(o, env):
        with torch.no_grad():
            acts, _, _, _ = agent.select_actions_batched(
                o, env.drone_pos, env.drone_alive, greedy=greedy)
        return acts
    # Reset per-episode waypoint state at every fresh env batch in rollouts.
    fn.reset = agent.begin_rollout
    return fn


def _eval_result(action_fn, grid=30, n_drones=10, max_steps=300,
                 n_envs=32, n_eps=20, wind=0.0, seed=None):
    """Fresh batched-env evaluation, formatted like the CPU baseline dicts."""
    cm, cs, sm, ss = _run_batched_rollouts(
        action_fn, grid=grid, n_drones=n_drones, max_steps=max_steps,
        n_envs=n_envs, n_eps=n_eps, wind=wind, seed=seed)
    return {'safety': sm, 'coverage': cm, 'safety_std': ss, 'coverage_std': cs}


# ═══════════════════════════════════════════════════════════════
# RIGOROUS BENCHMARKING — Pareto frontier + RLiable-style stats
# ═══════════════════════════════════════════════════════════════
# Multi-seed fresh-episode evaluation with Interquartile Mean (IQM),
# stratified-bootstrap 95% CIs (RLiable convention), and Welch's t-test +
# Mann-Whitney U against the random baseline — plus the coverage-vs-safety
# Pareto plot. Every policy is measured on IDENTICAL fresh episodes.

def _iqm(xs):
    """Interquartile mean (mean of the middle 50% of samples)."""
    a = np.sort(np.asarray(xs, dtype=np.float64))
    n = a.size
    lo = max(0, n // 4 - 1)
    hi = min(n, n - n // 4 + 1)
    return float(a[lo:hi].mean()) if hi > lo else float(a.mean())


def _stratified_bootstrap_ci(groups, stat=_iqm, n_boot=2000, seed=0):
    """95% CI of `stat` via stratified resampling (per env-seed stratum).

    groups: list of per-stratum sample arrays. Each bootstrap draw resamples
    every stratum with replacement (same size) and pools, then applies stat.
    """
    rng = np.random.default_rng(seed)
    gs = [np.asarray(g, dtype=np.float64) for g in groups]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pooled = np.concatenate([rng.choice(g, size=g.size, replace=True) for g in gs])
        boots[b] = stat(pooled)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def _pareto_summary(checkpoints=None, grid=30, n_drones=10, max_steps=300,
                    n_envs=32, n_eps=96, wind=0.0, seed=4242, n_boot=2000,
                    run_id="pareto", fig_path="pareto_frontier.png"):
    """Benchmark trained checkpoints + baselines + oracle on identical fresh
    episodes; print an IQM/bootstrap/significance table and save a Pareto plot.

    checkpoints: list of (label, path, use_gat) — e.g. ('GAT best s42',
    'gpu_gat_s42_best.pt', True). Baselines (random/greedy/PID/oracle) are
    always included. Episodes are grouped into n_envs-sized strata (each shares
    an env seed) so bootstrap CIs are stratified. Saves <run_id>_pareto.json
    and the PNG into /kaggle/working (cwd).
    """
    import scipy.stats as sp
    from matplotlib import pyplot as plt

    rows = {}
    base_fns = [("Random", _random_actions), ("Greedy", _greedy_baseline_actions),
                ("PID", _pid_baseline_actions), ("FrontierOracle", _frontier_oracle_actions)]
    for label, fn in base_fns:
        covs, safes = _run_batched_rollouts(fn, grid=grid, n_drones=n_drones,
                                            max_steps=max_steps, n_envs=n_envs,
                                            n_eps=n_eps, wind=wind, seed=seed,
                                            return_lists=True)
        rows[label] = (covs, safes)
    if checkpoints:
        probe = GPUWildfireEnv(n_envs=2, grid=grid, n_drones=n_drones, max_steps=10)
        obs_dim = probe.obs_dim
        for label, path, use_gat in checkpoints:
            ag = BatchedGATPPO(obs_dim=obs_dim, act_dim=5, use_gat=use_gat)
            ag.load(path)
            covs, safes = _run_batched_rollouts(_trained_agent_actions(ag),
                                                grid=grid, n_drones=n_drones,
                                                max_steps=max_steps, n_envs=n_envs,
                                                n_eps=n_eps, wind=wind, seed=seed,
                                                return_lists=True)
            rows[label] = (covs, safes)

    # ── stats per policy ──
    table = {}
    strat = max(1, n_envs)
    for label, (covs, safes) in rows.items():
        cov_groups = [covs[i:i + strat] for i in range(0, len(covs), strat)]
        safe_groups = [safes[i:i + strat] for i in range(0, len(safes), strat)]
        cov_lo, cov_hi = _stratified_bootstrap_ci(cov_groups, n_boot=n_boot, seed=42)
        safe_lo, safe_hi = _stratified_bootstrap_ci(safe_groups, n_boot=n_boot, seed=43)
        table[label] = {
            'coverage_mean': float(np.mean(covs)), 'coverage_iqm': _iqm(covs),
            'coverage_ci': [cov_lo, cov_hi],
            'safety_mean': float(np.mean(safes)), 'safety_iqm': _iqm(safes),
            'safety_ci': [safe_lo, safe_hi],
        }
    # ── significance vs Random ──
    rnd_c = np.asarray(rows['Random'][0]); rnd_s = np.asarray(rows['Random'][1])
    for label in table:
        if label == 'Random':
            continue
        covs = np.asarray(rows[label][0]); safes = np.asarray(rows[label][1])
        t_c = sp.ttest_ind(covs, rnd_c, equal_var=False)
        mw_c = sp.mannwhitneyu(covs, rnd_c, alternative='two-sided')
        t_s = sp.ttest_ind(safes, rnd_s, equal_var=False)
        mw_s = sp.mannwhitneyu(safes, rnd_s, alternative='two-sided')
        table[label]['vs_random'] = {
            'welch_cov_p': float(t_c.pvalue), 'mwu_cov_p': float(mw_c.pvalue),
            'welch_safe_p': float(t_s.pvalue), 'mwu_safe_p': float(mw_s.pvalue),
        }

    # ── print ──
    print("\n=== PARETO & STATISTICS (n=%d fresh episodes/policy, identical envs) ===" % n_eps)
    hdr = f"{'policy':16s} {'cov mean':>9s} {'cov IQM':>8s} {'cov 95%CI':>18s} {'safe mean':>9s} {'safe IQM':>8s} {'safe 95%CI':>18s} {'vsRandom(cov p)':>15s}"
    print(hdr); print('-' * len(hdr))
    for label, t in table.items():
        vp = t['vs_random']['welch_cov_p'] if 'vs_random' in t else 1.0
        print(f"{label:16s} {t['coverage_mean']:9.2f} {t['coverage_iqm']:8.2f} "
              f"[{t['coverage_ci'][0]:7.2f},{t['coverage_ci'][1]:7.2f}] "
              f"{t['safety_mean']:9.2f} {t['safety_iqm']:8.2f} "
              f"[{t['safety_ci'][0]:7.2f},{t['safety_ci'][1]:7.2f}] {vp:15.3g}")

    # ── Pareto plot (safety on x, coverage on y; frontier = step line) ──
    pts = sorted([(t['safety_mean'], t['coverage_mean'], label)
                  for label, t in table.items()], key=lambda p: p[0])
    plt.figure(figsize=(8, 6))
    for sx, cy, label in pts:
        plt.scatter(sx, cy, s=90, zorder=3)
        plt.annotate(label, (sx, cy), textcoords="offset points", xytext=(6, 6))
    # Pareto-optimal set (max coverage for >= given safety): step through left->right
    best = -1.0
    frontier = []
    for sx, cy, label in pts:
        if cy > best:
            best = cy
            frontier.append((sx, cy))
    if frontier:
        fx = [p[0] for p in frontier]; fy = [p[1] for p in frontier]
        plt.step(fx, fy, where='post', ls='--', color='gray', alpha=0.8, label='Pareto frontier')
    plt.xlabel('Survival (%)'); plt.ylabel('Coverage (%)')
    plt.title('Coverage–Safety Pareto frontier (identical fresh episodes)')
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    with open(f'{run_id}_pareto.json', 'w') as f:
        json.dump(table, f, indent=2)
    print(f"Saved {run_id}_pareto.json and {fig_path}")
    return table


# ═══════════════════════════════════════════════════════════════
# SECTION 4: BATCHED TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def batched_train(n_episodes=500, grid=30, n_drones=10, max_steps=300, 
                  n_envs=32, use_gat=True, seed=0, run_id="gpu_gat"):
    """GPU-accelerated training with n_envs parallel environments.
    
    Runs n_envs episodes simultaneously on GPU. Each training step
    collects n_envs × K transitions in one batched forward pass.
    
    Speedup: ~10-30x vs single-env CPU training.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    tag = "GPU-GAT-ITSE" if use_gat else "GPU-No-GAT"
    print(f"\n{'=' * 70}", flush=True)
    print(f"{tag} | seed={seed} | {n_episodes} eps | {n_envs} parallel envs | {n_drones} drones | {grid}×{grid}", flush=True)
    print(f"Device: {device} | Theoretical speedup: {n_envs}x over CPU", flush=True)
    print(f"{'=' * 70}", flush=True)
    
    env = GPUWildfireEnv(n_envs=n_envs, grid=grid, n_drones=n_drones, max_steps=max_steps)
    agent = BatchedGATPPO(obs_dim=env.obs_dim, act_dim=env.act_dim, use_gat=use_gat)
    
    rewards_h = []
    coverage_h = []
    safety_h = []
    best_cov = -float('inf')
    t0 = time.time()
    episodes_done = 0
    
    try:
        while episodes_done < n_episodes:
            # Run one batch of n_envs parallel episodes
            obs = env.reset()
            agent._obs_buf.clear()
            agent._act_buf.clear()
            agent._rew_buf.clear()
            agent._done_buf.clear()
            agent._lp_buf.clear()
            agent._val_buf.clear()
            agent._aid_buf.clear()
            agent.begin_rollout()
            batch_crashes = torch.zeros(n_envs, n_drones, device=device)
            batch_rewards = torch.zeros(n_envs, device=device)
            batch_coverages = torch.zeros(n_envs, device=device)
            
            for step in range(max_steps):
                am = env.drone_alive  # (N, K) — alive at the START of this step
                pos = env.drone_pos  # (N, K, 2)
                
                if not am.any():
                    break
                
                actions, log_probs, values, enhanced = agent.select_actions_batched(obs, pos, am)
                
                prev_visited = env.shared_visited.clone()
                
                obs_next, dones, crashed = env.step(actions)
                is_terminal = bool(dones.all().item())
                
                # ── Rewards: per-agent personal credit (no env-shared free-riding) ──
                px_now = env.drone_pos[:, :, 0].long().clamp(0, grid - 1)  # (N, K)
                py_now = env.drone_pos[:, :, 1].long().clamp(0, grid - 1)
                env_idx_all = torch.arange(n_envs, device=device).unsqueeze(1).expand(n_envs, n_drones)
                # Own new cell: this drone's current cell was unvisited at step start.
                # (Rewarding the ENV-wide count to every agent let drones free-ride on
                # others' exploration and collapse to hovering at the fire edge.)
                own_new = (prev_visited[env_idx_all, py_now, px_now] < 0.5) & am
                
                rewards = torch.zeros(n_envs, n_drones, device=device)
                rewards += 60.0 * own_new.float()             # personal exploration credit
                rewards += 1.5 * am.float()                   # survival tick (per step alive)
                
                # Fire-approach shaping: mild penalty for hugging the fire edge so
                # the agent learns boundaries from the obs channels instead of by
                # crashing, without paying drones to loiter at fd≈2.5.
                fd_now = env.fire_dist[env_idx_all, py_now, px_now]
                rewards -= 2.0 * ((fd_now < 1.5) & am).float()
                
                # Crowding penalty: without a dispersal incentive the team herds
                # into one cluster (final pairwise drone distance ~1-2 cells) and
                # covers a single corridor instead of sweeping the map.
                dp = env.drone_pos.unsqueeze(1) - env.drone_pos.unsqueeze(2)      # (N,K,K,2)
                dd = (dp * dp).sum(-1).sqrt()                                     # (N,K,K)
                eye = torch.eye(n_drones, dtype=torch.bool, device=device)
                close = (dd < 3.0) & ~eye.unsqueeze(0) & am.unsqueeze(1) & am.unsqueeze(2)
                n_neigh = close.sum(dim=2).float()                                 # (N,K)
                rewards -= 2.5 * n_neigh
                
                # Crash penalty
                rewards[crashed] = -50.0
                
                # Episode-end coverage bonus (only on the final stored step)
                cur_coverage = env.total_cells_explored.sum(dim=(1, 2)).float() / (grid * grid) * 100  # (N,)
                if is_terminal:
                    rewards += 100.0 * (cur_coverage / 100.0).unsqueeze(1).expand(n_envs, n_drones) * am.float()
                
                # Dead rows (post-crash) contribute nothing
                rewards[~am] = 0.0
                
                batch_crashes += crashed.float()
                
                # Store transitions (waypoint credit = own new cells, for the wp head)
                agent.store_batched(enhanced, actions, rewards, dones.float(), log_probs, values,
                                    waypoint_reward=own_new.float())
                
                batch_rewards += rewards.sum(dim=1)
                obs = obs_next
                
                if dones.all():
                    break
            
            # PPO update (gentler: fewer epochs, larger minibatches for ~96k-transition batches)
            agent.update(gamma=0.99, gae_lambda=0.95, n_epochs=3, batch_size=2048, entropy_coef=0.04)
            
            # Record stats for this batch (vectorized)
            covs = env.total_cells_explored.sum(dim=(1, 2)).float() / (grid * grid) * 100  # (N,)
            safes = (1.0 - batch_crashes.sum(dim=1) / n_drones) * 100  # (N,)
            rewards_h.extend(batch_rewards.cpu().tolist())
            coverage_h.extend(covs.cpu().tolist())
            safety_h.extend(safes.cpu().tolist())
            
            episodes_done += n_envs
            
            # Best-checkpoint tracking on a smoothed window of the last ~3 batches
            # (per-batch, so the best policy mid-run is captured — not just the
            # sparse log points every ~100 episodes).
            w = min(3 * n_envs, len(coverage_h))
            win_cov = float(np.mean(coverage_h[-w:]))
            if win_cov > best_cov:
                best_cov = win_cov
                agent.save(f'{run_id}_best.pt')
            
            # Log every 100 episodes
            if episodes_done % 100 <= n_envs:
                window = min(100, len(coverage_h))
                avg_r = np.mean(rewards_h[-window:])
                avg_cov = np.mean(coverage_h[-window:])
                avg_saf = np.mean(safety_h[-window:])
                elapsed = time.time() - t0
                eps_per_sec = episodes_done / elapsed
                eta_min = (n_episodes - episodes_done) / max(eps_per_sec, 1e-6) / 60.0
                exp_speed = np.mean(np.diff(coverage_h[-window:])) if len(coverage_h) > 1 else 0
                
                print(f"Ep {episodes_done:5d}/{n_episodes} | R: {avg_r:7.1f} | Cov: {avg_cov:5.1f}% | Safe: {avg_saf:4.0f}% | Speed: {exp_speed:+.3f}%/ep | {eps_per_sec:.1f} ep/s | ETA {eta_min:.0f}m", flush=True)
    
    except (KeyboardInterrupt, SystemExit, Exception) as e:
        print(f"\nTraining interrupted at ep {episodes_done}: {e}", flush=True)
    
    agent.save(f'{run_id}_final.pt')
    
    # ── Fresh-env evaluation of the FINAL checkpoint ──
    # The number tables publish for this run is measured on FRESH episodes in
    # the SAME batched env used for training, sampling from the policy's own
    # action distribution (seeded for reproducibility) — exactly like the
    # Random/Greedy/PID rows it is compared against — never train-time logged
    # coverage (which mixes shaped-reward episodes into the number) and never
    # a second, differently-implemented environment.
    train_final_cov = float(np.mean(coverage_h[-100:])) if len(coverage_h) >= 100 else float(np.mean(coverage_h))
    train_final_saf = float(np.mean(safety_h[-100:])) if len(safety_h) >= 100 else float(np.mean(safety_h))
    eval_cov_m, eval_cov_s, eval_saf_m, eval_saf_s = _run_batched_rollouts(
        _trained_agent_actions(agent), grid=grid, n_drones=n_drones,
        max_steps=max_steps, n_envs=n_envs, n_eps=n_envs, wind=0.0,
        seed=seed + 9000)
    
    results = {
        'n_episodes': len(coverage_h),
        'seed': seed,
        'final_reward': float(np.mean(rewards_h[-100:])) if len(rewards_h) >= 100 else float(np.mean(rewards_h)),
        # Fresh-env numbers — what comparison tables publish.
        'final_coverage': float(eval_cov_m),
        'coverage_std': float(eval_cov_s),
        'final_safety': float(eval_saf_m),
        'safety_std': float(eval_saf_s),
        # Train-time logged stats, kept for diagnostics (labeled as such).
        'train_final_coverage': train_final_cov,
        'train_final_safety': train_final_saf,
        'rewards': [float(x) for x in rewards_h],
        'coverages': [float(x) for x in coverage_h],
        'safety': [float(x) for x in safety_h],
    }
    
    with open(f'{run_id}_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    total_time = time.time() - t0
    print(f"\nDone in {total_time:.0f}s ({total_time/60:.1f}m) | Reward: {results['final_reward']:.1f} | "
          f"Coverage: {results['final_coverage']:.1f}% (fresh sampled, n={n_envs}) | "
          f"train-time: {train_final_cov:.1f}% | Safety: {results['final_safety']:.0f}%", flush=True)
    print(f"Speed: {len(coverage_h)/total_time:.1f} episodes/sec ({n_envs} parallel envs)", flush=True)
    
    return agent, results


# ═══════════════════════════════════════════════════════════════
# SECTION 5: FULL GPU PIPELINE
# ═══════════════════════════════════════════════════════════════

def gpu_run_full_pipeline():
    """Full research pipeline optimized for dual T4 GPU.
    
    All 9 phases from the paper, with GPU-accelerated training.
    Estimated time: 30-60 min on dual T4 (vs 8+ hours on CPU).
    """
    total_start = time.time()
    
    print("=" * 80, flush=True)
    print("PlumeGym-MARL v5: GPU-Accelerated Full Pipeline (Dual T4)", flush=True)
    print("=" * 80, flush=True)
    
    grid, n_drones, max_steps = 30, 10, 300
    n_envs = 32
    train_eps = 500
    
    # Import everything from the main module
    from kaggle_full_run import (
        train_mappo, train_ippo, statistical_analysis, generate_figures
    )
    
    # ═══ PHASE 1: Train GAT-ITSE × 3 seeds (GPU) ═══
    print("\n" + "=" * 60, flush=True)
    print("PHASE 1: Training GAT-ITSE × 3 seeds (GPU, 32 parallel envs)", flush=True)
    print("=" * 60, flush=True)
    gat_results = []
    gat_agents = []
    for seed in [42, 123, 777]:
        agent, res = batched_train(train_eps, grid, n_drones, max_steps, n_envs=n_envs,
                                   use_gat=True, seed=seed, run_id=f"gpu_gat_s{seed}")
        gat_results.append(res)
        gat_agents.append(agent)
    
    # ═══ PHASE 2: Train ablations (GPU) ═══
    print("\n" + "=" * 60, flush=True)
    print("PHASE 2: Training Ablations (GPU)", flush=True)
    print("=" * 60, flush=True)
    nogat_results = []
    for seed in [42, 123]:
        _, res = batched_train(train_eps, grid, n_drones, max_steps, n_envs=n_envs,
                               use_gat=False, seed=seed, run_id=f"gpu_nogat_s{seed}")
        nogat_results.append(res)
    
    # No-GP and No-CBF use CPU training (require per-step GP/CBF logic)
    print("\n  Training No-GP ablation (CPU)...", flush=True)
    _, nogp_res = train(200, grid, n_drones, max_steps, use_gat=True, seed=42,
                        run_id="gpu_nogp", use_gp=False, use_cbf=True)
    print("  Training No-CBF ablation (CPU)...", flush=True)
    _, nocbf_res = train(200, grid, n_drones, max_steps, use_gat=True, seed=42,
                         run_id="gpu_nocbf", use_gp=True, use_cbf=False)
    
    # ═══ PHASE 3: Train MAPPO (CPU) ═══
    print("\n" + "=" * 60, flush=True)
    print("PHASE 3: Training MAPPO × 2 seeds", flush=True)
    print("=" * 60, flush=True)
    mappo_res = []
    for seed in [42, 123]:
        _, res = train_mappo(train_eps, grid, n_drones, max_steps, seed=seed)
        mappo_res.append(res)
    
    # ═══ PHASE 4: Train IPPO (CPU) ═══
    print("\n" + "=" * 60, flush=True)
    print("PHASE 4: Training IPPO × 2 seeds", flush=True)
    print("=" * 60, flush=True)
    ippo_res = []
    for seed in [42, 123]:
        _, res = train_ippo(400, grid, n_drones, max_steps, seed=seed)
        ippo_res.append(res)
    
    # ═══ PHASE 5: Non-learned baselines ═══
    print("\n" + "=" * 60, flush=True)
    print("PHASE 5: Evaluating Baselines (fresh episodes, batched env)", flush=True)
    print("=" * 60, flush=True)
    # Baselines are evaluated in the SAME batched env as the trained agents
    # (identical physics and the same 5-action envelope) so every row of the
    # published table comes from one environment. Same seed => all methods see
    # the identical set of initial fire/spawn states.
    random_res = _eval_result(_random_actions, grid, n_drones, max_steps,
                              n_envs=n_envs, n_eps=20, seed=4242)
    greedy_res = _eval_result(_greedy_baseline_actions, grid, n_drones, max_steps,
                              n_envs=n_envs, n_eps=20, seed=4242)
    pid_res = _eval_result(_pid_baseline_actions, grid, n_drones, max_steps,
                           n_envs=n_envs, n_eps=20, seed=4242)
    
    # ═══ AGGREGATE ALL RESULTS ═══
    all_results = {}
    
    gat_covs = [r['final_coverage'] for r in gat_results]
    gat_safs = [r['final_safety'] for r in gat_results]
    all_results['GAT-ITSE'] = {
        'coverage': float(np.mean(gat_covs)), 'coverage_std': float(np.std(gat_covs)),
        'safety': float(np.mean(gat_safs)), 'safety_std': float(np.std(gat_safs)),
        'coverages': gat_covs,
    }
    
    nogat_covs = [r['final_coverage'] for r in nogat_results]
    nogat_safs = [r['final_safety'] for r in nogat_results]
    all_results['No-GAT'] = {
        'coverage': float(np.mean(nogat_covs)), 'coverage_std': float(np.std(nogat_covs)),
        'safety': float(np.mean(nogat_safs)), 'safety_std': float(np.std(nogat_safs)),
        'coverages': nogat_covs,
    }
    
    for name, res_data in [('No-GP', nogp_res), ('No-CBF', nocbf_res)]:
        all_results[name] = {
            'coverage': res_data['final_coverage'], 'coverage_std': 0,
            'safety': res_data['final_safety'], 'safety_std': 0,
            'coverages': [res_data['final_coverage']],
        }
    
    mappo_covs = [r['final_coverage'] for r in mappo_res]
    mappo_safs = [r['final_safety'] for r in mappo_res]
    all_results['MAPPO'] = {
        'coverage': float(np.mean(mappo_covs)), 'coverage_std': float(np.std(mappo_covs)),
        'safety': float(np.mean(mappo_safs)), 'safety_std': float(np.std(mappo_safs)),
        'coverages': mappo_covs,
    }
    
    ippo_covs = [r['final_coverage'] for r in ippo_res]
    ippo_safs = [r['final_safety'] for r in ippo_res]
    all_results['IPPO'] = {
        'coverage': float(np.mean(ippo_covs)), 'coverage_std': float(np.std(ippo_covs)),
        'safety': float(np.mean(ippo_safs)), 'safety_std': float(np.std(ippo_safs)),
        'coverages': ippo_covs,
    }
    
    for name, res_data in [('Random', random_res), ('Greedy', greedy_res), ('PID', pid_res)]:
        all_results[name] = res_data
    
    # Print summary
    print(f"\n{'=' * 70}", flush=True)
    print(f"{'Method':<15s} {'Safety':>10s} {'Coverage':>10s}", flush=True)
    print(f"{'-' * 70}", flush=True)
    for m in ['GAT-ITSE', 'No-GAT', 'No-GP', 'No-CBF', 'MAPPO', 'IPPO', 'Random', 'Greedy', 'PID']:
        r = all_results.get(m, {})
        print(f"{m:<15s} {r.get('safety', 0):>8.1f}% {r.get('coverage', 0):>8.1f}%", flush=True)
    print(f"{'=' * 70}", flush=True)
    
    # ═══ PHASE 6: Wind Robustness Sweep ═══
    print("\n" + "=" * 60, flush=True)
    print("PHASE 6: Wind Robustness Sweep", flush=True)
    print("=" * 60, flush=True)
    # Load best GAT agent for evaluation
    best_seed_idx = int(np.argmax(gat_covs))
    best_agent = gat_agents[best_seed_idx]
    wind_results = {}
    # Wind sweeps run in the same batched env (it supports wind_speed) with the
    # same fresh-episode protocol as the main table.
    for wind in [5, 10, 15, 20, 25]:
        print(f"  Wind = {wind} m/s", flush=True)
        wind_results[wind] = {}
        wind_results[wind]['GAT-ITSE'] = _eval_result(
            _trained_agent_actions(best_agent), grid, n_drones, max_steps,
            n_envs=n_envs, n_eps=20, wind=wind, seed=4242 + wind)
        wind_results[wind]['Random'] = _eval_result(
            _random_actions, grid, n_drones, max_steps,
            n_envs=n_envs, n_eps=20, wind=wind, seed=4242 + wind)
        wind_results[wind]['Greedy'] = _eval_result(
            _greedy_baseline_actions, grid, n_drones, max_steps,
            n_envs=n_envs, n_eps=20, wind=wind, seed=4242 + wind)
        wind_results[wind]['PID'] = _eval_result(
            _pid_baseline_actions, grid, n_drones, max_steps,
            n_envs=n_envs, n_eps=20, wind=wind, seed=4242 + wind)
        print(f"    GAT-ITSE: Safety={wind_results[wind]['GAT-ITSE']['safety']:.1f}%, Cov={wind_results[wind]['GAT-ITSE']['coverage']:.1f}%", flush=True)
    
    # ═══ PHASE 7: Scalability ═══
    print("\n" + "=" * 60, flush=True)
    print("PHASE 7: Scalability Analysis", flush=True)
    print("=" * 60, flush=True)
    scalability_results = {'swarm': {}, 'grid': {}}
    for n_d in [5, 10, 20]:
        print(f"  Swarm = {n_d} drones", flush=True)
        scalability_results['swarm'][n_d] = {}
        scalability_results['swarm'][n_d]['GAT-ITSE'] = _eval_result(
            _trained_agent_actions(best_agent), grid, n_d, max_steps,
            n_envs=10, n_eps=10, wind=0, seed=4242 + n_d)
        scalability_results['swarm'][n_d]['Random'] = _eval_result(
            _random_actions, grid, n_d, max_steps,
            n_envs=10, n_eps=10, wind=0, seed=4242 + n_d)
    for gs in [20, 30, 50]:
        print(f"  Grid = {gs}×{gs}", flush=True)
        scalability_results['grid'][gs] = {}
        scalability_results['grid'][gs]['GAT-ITSE'] = _eval_result(
            _trained_agent_actions(best_agent), gs, n_drones, max_steps,
            n_envs=10, n_eps=10, wind=0, seed=4242 + gs)
        scalability_results['grid'][gs]['Random'] = _eval_result(
            _random_actions, gs, n_drones, max_steps,
            n_envs=10, n_eps=10, wind=0, seed=4242 + gs)
    
    # ═══ PHASE 8: Statistical Analysis ═══
    print("\n" + "=" * 60, flush=True)
    print("PHASE 8: Statistical Analysis", flush=True)
    print("=" * 60, flush=True)
    statistical_analysis(all_results)
    
    # ═══ PHASE 9: Publication Figures ═══
    print("\n" + "=" * 60, flush=True)
    print("PHASE 9: Generating Publication Figures", flush=True)
    print("=" * 60, flush=True)
    training_histories = {}
    for name, res_list in [('GAT-ITSE', gat_results), ('No-GAT', nogat_results),
                           ('MAPPO', mappo_res), ('IPPO', ippo_res)]:
        if res_list:
            last = res_list[-1]
            training_histories[name] = {
                'coverages': last.get('coverages', []),
                'safety': last.get('safety', []),
                'rewards': last.get('rewards', []),
                'attention_entropy': last.get('attention_entropy', []),
            }
    generate_figures(all_results, training_histories, wind_results, scalability_results)
    
    # ═══ SAVE ALL RESULTS ═══
    final_results = {
        'benchmark': {k: {kk: vv for kk, vv in v.items() if kk != 'coverages'} for k, v in all_results.items()},
        'wind_robustness': {str(k): v for k, v in wind_results.items()},
        'scalability': {k: {str(kk): vv for kk, vv in v.items()} for k, v in scalability_results.items()},
    }
    with open('gpu_benchmark_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    total_time = time.time() - total_start
    print(f"\n{'=' * 80}", flush=True)
    print(f"GPU PIPELINE COMPLETE — ALL 9 PHASES DONE", flush=True)
    print(f"Total time: {total_time / 3600:.1f} hours ({total_time / 60:.0f} minutes)", flush=True)
    print(f"{'=' * 80}", flush=True)


if __name__ == "__main__":
    gpu_run_full_pipeline()
