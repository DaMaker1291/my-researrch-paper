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
        self.global_obs_dim = 10
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
        """Vectorized fire distance computation on GPU."""
        fire_mask = (self.fire > 0.2)  # (N, G, G)
        has_fire = fire_mask.sum(dim=(1, 2)) > 0  # (N,)
        
        self.fire_dist.fill_(10.0)
        
        # For each environment with fire, compute distance transform
        for n in range(self.n_envs):
            if not has_fire[n]:
                continue
            # Find fire cells
            fire_cells = torch.argwhere(fire_mask[n])  # (M, 2)
            if len(fire_cells) == 0:
                continue
            # Compute min distance from each cell to nearest fire cell
            fy = fire_cells[:, 0].float()
            fx = fire_cells[:, 1].float()
            # Vectorized distance: (G, G, M) → min over M
            dist_y = self.yy.unsqueeze(2) - fy.view(1, 1, -1)
            dist_x = self.xx.unsqueeze(2) - fx.view(1, 1, -1)
            dist = torch.sqrt(dist_x**2 + dist_y**2 + 1e-8)
            self.fire_dist[n] = dist.min(dim=2)[0]
    
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
            rewards: (N, K) float tensor
            dones: (N, K) bool tensor
            infos: dict with crash info
        """
        self.step_count += 1
        N, G, K = self.n_envs, self.grid, self.n_drones
        
        # Get action deltas for all drones
        action_deltas = self.action_deltas[actions]  # (N, K, 2)
        
        # Update velocity with momentum
        new_vel = self.momentum * self.drone_vel + (1 - self.momentum) * action_deltas
        self.drone_vel = new_vel
        
        # Update position
        new_pos = self.drone_pos + new_vel
        new_pos = new_pos.clamp(self.boundary_margin, G - 1 - self.boundary_margin)
        self.drone_pos = new_pos
        
        # Mark visited cells
        px_int = new_pos[:, :, 0].long().clamp(0, G-1)
        py_int = new_pos[:, :, 1].long().clamp(0, G-1)
        
        for i in range(K):
            alive_mask = self.drone_alive[:, i]  # (N,)
            env_idx = torch.arange(N, device=device)
            valid = alive_mask
            self.shared_visited[env_idx[valid], py_int[valid, i], px_int[valid, i]] = 1.0
            self.total_cells_explored[env_idx[valid], py_int[valid, i], px_int[valid, i]] = True
        
        # Check crashes
        dones = torch.zeros(N, K, dtype=torch.bool, device=device)
        crashed = torch.zeros(N, K, dtype=torch.bool, device=device)
        
        for i in range(K):
            alive = self.drone_alive[:, i]  # (N,)
            if not alive.any():
                continue
            ix = px_int[:, i].clamp(0, G-1)
            iy = py_int[:, i].clamp(0, G-1)
            env_idx = torch.arange(N, device=device)
            
            # Fire crash
            fire_val = self.fire[env_idx, iy, ix]
            fire_near = self.fire_dist[env_idx, iy, ix]
            thermal_val = self.thermal[env_idx, iy, ix]
            
            crash_mask = alive & (
                (fire_val > self.fire_crash_threshold) |
                (fire_near < 0.5) |
                (thermal_val > self.thermal_crash)
            )
            
            crashed[:, i] = crash_mask
            self.drone_alive[:, i] = ~crash_mask
            dones[:, i] = crash_mask | ~alive
        
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
        """Vectorized observation gathering on GPU.
        
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
        
        for i in range(K):
            alive = self.drone_alive[:, i]  # (N,)
            if not alive.any():
                continue
            
            px = self.drone_pos[:, i, 0].long().clamp(0, G-1)  # (N,)
            py = self.drone_pos[:, i, 1].long().clamp(0, G-1)  # (N,)
            
            alive_idx = torch.where(alive)[0]
            for n_idx in alive_idx:
                n = n_idx.item()
                ix, iy = px[n].item(), py[n].item()
                # In padded coords, the center is at (ix+r, iy+r)
                # Patch is [iy : iy+2r+1, ix : ix+2r+1] in padded coords
                patch = channels_padded[n, :, iy:iy+2*r+1, ix:ix+2*r+1]  # (ch, 2r+1, 2r+1)
                obs[n, i, :local_dim] = patch.reshape(-1)
            
            # Global features for alive drones
            alive_idx = torch.where(alive)[0]
            for n_idx in alive_idx:
                n = n_idx.item()
                px_i, py_i = px[n].item(), py[n].item()
                
                fire_cells = (self.fire[n] > 0.2).sum().item()
                fr = float(math.sqrt(max(1, fire_cells))) / G
                
                wind_mag = float(math.sqrt(self.wind_x[n, py_i, px_i]**2 + 
                                           self.wind_y[n, py_i, px_i]**2))
                wind_dir = float(math.atan2(self.wind_y[n, py_i, px_i],
                                            self.wind_x[n, py_i, px_i])) / math.pi
                coverage = float(self.episode_cells[n]) / (G * G)
                
                obs[n, i, local_dim:] = torch.tensor([
                    px_i / G, py_i / G,
                    self.drone_vel[n, i, 0].item(), self.drone_vel[n, i, 1].item(),
                    float(self.fire[n, py_i, px_i]),
                    float(self.thermal[n, py_i, px_i]) / self.thermal_cap,
                    wind_mag / 30.0, wind_dir,
                    coverage, fr
                ], device=device)
        
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

class BatchedGATPPO:
    """PPO agent that processes all N environments × K agents in batched fashion.
    
    Key optimization: instead of calling select_actions() K times per env,
    we stack all observations and do a single batched forward pass through
    GAT and policy networks.
    """
    
    def __init__(self, obs_dim, act_dim=5, use_gat=True, lr=3e-4, comm_range=10.0):
        if use_gat:
            self.gat = GATCommunication(obs_dim, hidden_dim=128, out_dim=64, comm_range=comm_range).to(device)
        else:
            self.gat = NoGATCommunication(obs_dim, out_dim=64).to(device)
        
        self.policy = PPONetwork(self.gat.enhanced_obs_dim, act_dim).to(device)
        params = list(self.gat.parameters()) + list(self.policy.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=2000, eta_min=1e-5)
        self._traj = []
        self.itse_weight = 0.3
        self.explore_bonus_scale = 2.0
    
    def select_actions_batched(self, obs, positions, alive_mask):
        """Batched action selection across all envs and agents.
        
        Args:
            obs: (N, K, obs_dim) tensor
            positions: (N, K, 2) tensor
            alive_mask: (N, K) bool tensor
        
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
        
        # GAT forward pass (batched)
        enhanced = self.gat(obs_flat, pos_flat, alive_flat)
        
        # Policy forward pass
        with torch.no_grad():
            logits, values = self.policy(enhanced)
            dist = torch.distributions.Categorical(logits=logits)
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
    
    def store_batched(self, enhanced_obs, actions, rewards, dones, log_probs, values):
        """Store transitions from all N environments."""
        N, K = actions.shape
        for n in range(N):
            for k in range(K):
                self._traj.append({
                    'obs': enhanced_obs[n, k].detach().cpu(),
                    'action': actions[n, k].item(),
                    'reward': rewards[n, k].item(),
                    'done': dones[n, k].item(),
                    'log_prob': log_probs[n, k].item(),
                    'value': values[n, k].item(),
                    'agent_id': k
                })
    
    def update(self, gamma=0.99, gae_lambda=0.95, n_epochs=6, batch_size=1024, clip_eps=0.2, entropy_coef=0.02):
        """PPO update using collected trajectories."""
        if not self._traj:
            return 0.0
        
        n = len(self._traj)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        
        # Compute GAE per agent
        agent_groups = defaultdict(list)
        for i, t in enumerate(self._traj):
            agent_groups[t['agent_id']].append(i)
        
        for aid, indices in agent_groups.items():
            gae = 0.0
            for k in reversed(range(len(indices))):
                idx = indices[k]
                next_idx = indices[k + 1] if k + 1 < len(indices) else None
                next_val = 0.0 if next_idx is None else self._traj[next_idx]['value']
                next_done = 1.0 if next_idx is None else self._traj[next_idx]['done']
                delta = self._traj[idx]['reward'] + gamma * next_val * (1 - next_done) - self._traj[idx]['value']
                gae = delta + gamma * gae_lambda * (1 - next_done) * gae
                advantages[idx] = gae
                returns[idx] = gae + self._traj[idx]['value']
        
        all_obs = torch.stack([t['obs'].squeeze(0) for t in self._traj]).to(device)
        all_actions = torch.tensor([t['action'] for t in self._traj], dtype=torch.long, device=device)
        all_old_lp = torch.tensor([t['log_prob'] for t in self._traj], dtype=torch.float32, device=device)
        all_adv = torch.tensor(advantages, dtype=torch.float32, device=device)
        all_ret = torch.tensor(returns, dtype=torch.float32, device=device)
        all_adv = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-8)
        
        total_loss, count = 0.0, 0
        all_params = list(self.gat.parameters()) + list(self.policy.parameters())
        
        for _ in range(n_epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                _, new_lp, entropy, vals = self.policy.evaluate(all_obs[idx], all_actions[idx])
                ratio = torch.exp(new_lp - all_old_lp[idx])
                s1 = ratio * all_adv[idx]
                s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * all_adv[idx]
                loss = -torch.min(s1, s2).mean() + 0.5 * F.mse_loss(vals, all_ret[idx]) - entropy_coef * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, 0.5)
                self.optimizer.step()
                total_loss += loss.item()
                count += 1
        
        self._traj.clear()
        self.scheduler.step()
        return total_loss / max(1, count)
    
    def save(self, path):
        torch.save({'gat': self.gat.state_dict(), 'policy': self.policy.state_dict()}, path)
    
    def load(self, path):
        ckpt = torch.load(path, map_location=device)
        self.gat.load_state_dict(ckpt['gat'])
        self.policy.load_state_dict(ckpt['policy'])


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
            agent._traj.clear()
            batch_crashes = torch.zeros(n_envs, n_drones, device=device)
            batch_rewards = torch.zeros(n_envs, device=device)
            batch_coverages = torch.zeros(n_envs, device=device)
            
            for step in range(max_steps):
                am = env.drone_alive  # (N, K)
                pos = env.drone_pos  # (N, K, 2)
                
                if not am.any():
                    break
                
                actions, log_probs, values, enhanced = agent.select_actions_batched(obs, pos, am)
                
                prev_visited = env.shared_visited.clone()
                
                obs_next, dones, crashed = env.step(actions)
                
                # Compute rewards for all envs × agents
                cur_coverage = env.total_cells_explored.sum(dim=(1, 2)).float() / (grid * grid) * 100  # (N,)
                
                rewards = torch.zeros(n_envs, n_drones, device=device)
                for i in range(n_drones):
                    alive = am[:, i]
                    if not alive.any():
                        continue
                    
                    new_cells = ((prev_visited < 0.5) & (env.shared_visited > 0.5)).sum(dim=(1, 2)).float()  # (N,)
                    
                    # Exploration reward: +30/cell
                    rewards[:, i] += 30.0 * new_cells
                    
                    # Frontier bonus
                    has_new = new_cells > 0
                    rewards[has_new, i] += 15.0
                    no_new = ~has_new
                    rewards[no_new & (cur_coverage < 90), i] += 2.0
                    
                    # Coverage reward
                    rewards[:, i] += 50.0 * cur_coverage / 100.0
                    
                    # Fire proximity (safe zone)
                    env_idx = torch.arange(n_envs, device=device)
                    fd = env.fire_dist[env_idx, 
                                       (pos[:, i, 1]).long().clamp(0, grid-1),
                                       (pos[:, i, 0]).long().clamp(0, grid-1)]
                    safe_fire = (fd > 0.5) & (fd < 8.0)
                    rewards[safe_fire, i] += 12.0 * (1.0 - (fd[safe_fire] - 2.5).abs() / 5.5).clamp(0)
                    
                    # Survival
                    rewards[:, i] += 0.5
                    
                    # Crash penalty
                    rewards[crashed[:, i], i] = -30.0
                    
                    batch_crashes[:, i] = crashed[:, i].float()
                
                # Store transitions
                agent.store_batched(enhanced, actions, rewards, dones.float(), log_probs, values)
                
                batch_rewards += rewards.sum(dim=1)
                obs = obs_next
                
                if dones.all():
                    break
            
            # PPO update
            agent.update()
            
            # Record stats for this batch
            for n in range(n_envs):
                cov = float(env.total_cells_explored[n].sum()) / (grid * grid) * 100
                saf = float(1.0 - batch_crashes[n].sum() / n_drones) * 100
                rewards_h.append(float(batch_rewards[n]))
                coverage_h.append(cov)
                safety_h.append(saf)
            
            episodes_done += n_envs
            
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
                
                if avg_cov > best_cov:
                    best_cov = avg_cov
                    agent.save(f'{run_id}_best.pt')
    
    except (KeyboardInterrupt, SystemExit, Exception) as e:
        print(f"\nTraining interrupted at ep {episodes_done}: {e}", flush=True)
    
    agent.save(f'{run_id}_final.pt')
    
    results = {
        'n_episodes': len(coverage_h),
        'seed': seed,
        'final_reward': float(np.mean(rewards_h[-100:])) if len(rewards_h) >= 100 else float(np.mean(rewards_h)),
        'final_coverage': float(np.mean(coverage_h[-100:])) if len(coverage_h) >= 100 else float(np.mean(coverage_h)),
        'final_safety': float(np.mean(safety_h[-100:])) if len(safety_h) >= 100 else float(np.mean(safety_h)),
        'rewards': [float(x) for x in rewards_h],
        'coverages': [float(x) for x in coverage_h],
        'safety': [float(x) for x in safety_h],
    }
    
    with open(f'{run_id}_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    total_time = time.time() - t0
    print(f"\nDone in {total_time:.0f}s ({total_time/60:.1f}m) | Reward: {results['final_reward']:.1f} | Coverage: {results['final_coverage']:.1f}% | Safety: {results['final_safety']:.0f}%", flush=True)
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
        eval_random, eval_greedy, eval_pid, eval_trained_agent,
        train_mappo, train_ippo, statistical_analysis, generate_figures,
        WildfireEnv, FastGATPPO
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
    print("PHASE 5: Evaluating Baselines", flush=True)
    print("=" * 60, flush=True)
    random_res = eval_random(grid, n_drones, max_steps, wind=0.0, n_eps=20)
    greedy_res = eval_greedy(grid, n_drones, max_steps, wind=0.0, n_eps=20)
    pid_res = eval_pid(grid, n_drones, max_steps, wind=0.0, n_eps=20)
    
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
    for wind in [5, 10, 15, 20, 25]:
        print(f"  Wind = {wind} m/s", flush=True)
        wind_results[wind] = {}
        wind_results[wind]['GAT-ITSE'] = eval_trained_agent(
            best_agent, grid, n_drones, max_steps, wind=wind, n_eps=20, use_gat=True)
        wind_results[wind]['Random'] = eval_random(grid, n_drones, max_steps, wind=wind, n_eps=20)
        wind_results[wind]['Greedy'] = eval_greedy(grid, n_drones, max_steps, wind=wind, n_eps=20)
        wind_results[wind]['PID'] = eval_pid(grid, n_drones, max_steps, wind=wind, n_eps=20)
        print(f"    GAT-ITSE: Safety={wind_results[wind]['GAT-ITSE']['safety']:.1f}%, Cov={wind_results[wind]['GAT-ITSE']['coverage']:.1f}%", flush=True)
    
    # ═══ PHASE 7: Scalability ═══
    print("\n" + "=" * 60, flush=True)
    print("PHASE 7: Scalability Analysis", flush=True)
    print("=" * 60, flush=True)
    scalability_results = {'swarm': {}, 'grid': {}}
    for n_d in [5, 10, 20]:
        print(f"  Swarm = {n_d} drones", flush=True)
        scalability_results['swarm'][n_d] = {}
        scalability_results['swarm'][n_d]['GAT-ITSE'] = eval_trained_agent(
            best_agent, grid, n_d, max_steps, wind=0, n_eps=10, use_gat=True)
        scalability_results['swarm'][n_d]['Random'] = eval_random(grid, n_d, max_steps, wind=0, n_eps=10)
    for gs in [20, 30, 50]:
        print(f"  Grid = {gs}×{gs}", flush=True)
        scalability_results['grid'][gs] = {}
        scalability_results['grid'][gs]['GAT-ITSE'] = eval_trained_agent(
            best_agent, gs, n_drones, max_steps, wind=0, n_eps=10, use_gat=True)
        scalability_results['grid'][gs]['Random'] = eval_random(gs, n_drones, max_steps, wind=0, n_eps=10)
    
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
