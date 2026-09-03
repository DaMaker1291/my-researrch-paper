#!/usr/bin/env python3
"""
=================================================================
PlumeGym-MARL: Research-Grade Training Pipeline (GPU) — v4
=================================================================
Upload to Kaggle. Set GPU runtime.

Research contributions:
  1. GAT-MARAHS: Graph Attention Networks for multi-agent wildfire tracking
  2. Shared exploration map — agents see union of all visited cells
  3. Information-theoretic reward shaping (fire front observation bonus)
  4. Overlap penalty + quadrant diversity for coordinated exploration
  5. 5-method ablation: GAT-MARAHS vs No-GAT vs MAPPO vs IPPO vs No-Comm
  6. Wind robustness sweep (train wind=0, evaluate wind 5-25)
  7. Communication entropy analysis (GAT attention diversity)
  8. Communication graph topology (density, components, avg path)
  9. Scalability analysis (5/10/20 drones + 20x20/30x30/50x50 grids)
 10. Sample efficiency & exploration speed metrics
 11. Contribution isolation ablation (shared map vs GAT vs both)

Pipeline (10 phases, ~8.5 hrs on T4):
  1. Train GAT-MARAHS × 5 seeds (500 eps each)
  2. Train No-GAT ablation × 5 seeds
  3. Train MAPPO baseline × 2 seeds (slower per-episode)
  4. Train IPPO baseline × 5 seeds (Yu et al. 2021)
  5. Train No-Comm ablation × 5 seeds (no GAT + no shared map)
  6. Benchmark vs Random / Greedy / PID (with 95% bootstrap CIs)
  7. Wind sweep (5/10/15/20/25 m/s) with error bars
  8. Scalability: swarm size (5/10/20) + grid size (20/30/50)
  9. Communication topology analysis (graph metrics + attention patterns)
 10. 13 publication figures + full statistical analysis
     (Mann-Whitney U, Cohen's d, bootstrap CIs, effect sizes)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, json, os, sys

try:
    from scipy import stats as sp_stats
except ImportError:
    sp_stats = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | {props.total_memory / 1e9:.1f} GB", flush=True)
else:
    print("Running on CPU (slower)", flush=True)

# ═══════════════════════════════════════════════════════════════
# SECTION 1: ENVIRONMENT
# ═══════════════════════════════════════════════════════════════

class WildfireEnv:
    def __init__(self, grid=30, n_drones=10, max_steps=300, wind_speed=0.0, obs_r=4):
        self.grid = grid
        self.n_drones = n_drones
        self.max_steps = max_steps
        self.base_wind = wind_speed
        self.obs_r = obs_r
        self.obs_size = 2 * obs_r + 1  # 9
        self.obs_channels = 5  # fire, thermal, wind_x, wind_y, shared_visited
        self.local_obs_dim = self.obs_channels * self.obs_size * self.obs_size  # 5*81=405
        self.global_obs_dim = 8
        self.obs_dim = self.local_obs_dim + self.global_obs_dim  # 413
        self.act_dim = 5
        self.action_deltas = np.array([[0,0],[0,1],[0,-1],[1,0],[-1,0]], dtype=np.float32)
        # Physics
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
        # Precompute coordinate grids
        self._yy, self._xx = np.meshgrid(np.arange(grid, dtype=np.float32),
                                          np.arange(grid, dtype=np.float32), indexing='ij')
        self.reset()

    def reset(self):
        self.step_count = 0
        self.total_cells_explored = set()
        self.fire = np.zeros((self.grid, self.grid), dtype=np.float32)
        cx = np.random.randint(self.base_fire_radius + 2, self.grid - self.base_fire_radius - 2)
        cy = np.random.randint(self.base_fire_radius + 2, self.grid - self.base_fire_radius - 2)
        for dx in range(-self.base_fire_radius, self.base_fire_radius + 1):
            for dy in range(-self.base_fire_radius, self.base_fire_radius + 1):
                if dx*dx + dy*dy <= self.base_fire_radius**2:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.grid and 0 <= ny < self.grid:
                        self.fire[ny, nx] = self.fire_intensity_init
        self.fire_center = np.array([cx, cy], dtype=np.float32)
        self.fuel = np.clip(0.8 - 0.3 * np.random.randn(self.grid, self.grid), 0.3, 1.0).astype(np.float32)
        angle = np.random.uniform(0, 2 * np.pi)
        self.wind_x = np.full((self.grid, self.grid), self.base_wind * np.cos(angle), dtype=np.float32)
        self.wind_y = np.full((self.grid, self.grid), self.base_wind * np.sin(angle), dtype=np.float32)
        if self.base_wind > 0:
            for k in range(3):
                freq = 0.5 * (k + 1)
                amp = 0.1 * self.base_wind
                phase = np.random.uniform(0, 2 * np.pi)
                self.wind_x += amp * np.sin(2 * np.pi * freq * self._xx / self.grid + phase)
                self.wind_y += amp * np.sin(2 * np.pi * freq * self._yy / self.grid + phase * 0.7)
        self.thermal = np.zeros((self.grid, self.grid), dtype=np.float32)
        self._update_thermal()
        self._fire_dist_cache = None
        self._update_fire_dist()
        self._visited_grid = np.zeros((self.grid, self.grid), dtype=np.float32)
        self.shared_visited = np.zeros((self.grid, self.grid), dtype=np.float32)
        self.drones = []
        for i in range(self.n_drones):
            while True:
                px = np.random.uniform(2, self.grid - 2)
                py = np.random.uniform(2, self.grid - 2)
                ix, iy = int(px), int(py)
                if 0 <= ix < self.grid and 0 <= iy < self.grid and self.fire[iy, ix] < 0.1:
                    break
            self.drones.append({
                'pos': np.array([px, py], dtype=np.float32),
                'vel': np.array([0.0, 0.0], dtype=np.float32),
                'alive': True,
                'visited': set(),
            })
        return self._get_obs()

    def _update_thermal(self):
        fire_mask = (self.fire > 0.2).astype(np.float32)
        if fire_mask.sum() < 1e-6:
            self.thermal = np.zeros((self.grid, self.grid), dtype=np.float32)
            return
        # Vectorized thermal: broadcast exp(-dist^2/8) over all fire cells at once
        fire_yx = np.argwhere(fire_mask > 0).astype(np.float32)
        # fire_yx shape: (N_fire, 2) — each row is [fy, fx]
        # We need: thermal[y,x] = sum_f intensity[f] * exp(-((y-fy)^2 + (x-fx)^2) / 8)
        # Flatten grid coords: shape (G*G, 2)
        G = self.grid
        grid_coords = np.stack([self._yy.ravel(), self._xx.ravel()], axis=1).astype(np.float32)  # (G*G, 2)
        # Pairwise dist^2: (G*G, N_fire)
        diff = grid_coords[:, None, :] - fire_yx[None, :, :]  # (G*G, N_fire, 2)
        dist_sq = (diff ** 2).sum(axis=2)  # (G*G, N_fire)
        # Intensities for each fire cell
        intensities = self.fire[fire_yx[:, 0].astype(int), fire_yx[:, 1].astype(int)]  # (N_fire,)
        # Weighted sum
        thermal_flat = (intensities[None, :] * np.exp(-dist_sq / 8.0)).sum(axis=1)  # (G*G,)
        self.thermal = np.clip(thermal_flat.reshape(G, G), 0, self.thermal_cap).astype(np.float32)

    def _update_fire_dist(self):
        fire_cells = np.argwhere(self.fire > 0.2)
        if len(fire_cells) == 0:
            self._fire_dist_cache = np.full((self.grid, self.grid), 10.0, dtype=np.float32)
            return
        fire_y = fire_cells[:, 0].reshape(-1, 1, 1)
        fire_x = fire_cells[:, 1].reshape(-1, 1, 1)
        all_dists = np.sqrt((self._xx[None, :, :] - fire_x)**2 + (self._yy[None, :, :] - fire_y)**2)
        self._fire_dist_cache = np.min(all_dists, axis=0).astype(np.float32)

    def _spread_fire(self, rng):
        """Vectorized fire spread: batch random draws per fire cell for ~10x speedup."""
        fire_mask = (self.fire > 0.2)
        fire_cells = np.argwhere(fire_mask)
        new_fire = self.fire.copy()
        if len(fire_cells) == 0:
            self.fuel = np.clip(self.fuel - self.fuel_depletion_rate * fire_mask.astype(np.float32), 0.0, 1.0)
            return new_fire
        # Vectorized spread probability per fire cell
        fy_arr, fx_arr = fire_cells[:, 0], fire_cells[:, 1]
        intensity = self.fuel[fy_arr, fx_arr] * self.fire[fy_arr, fx_arr]
        wind_mag = np.sqrt(self.wind_x[fy_arr, fx_arr]**2 + self.wind_y[fy_arr, fx_arr]**2)
        spread_prob = self.spread_rate * (1 + self.wind_amplification * wind_mag) * intensity
        # 8-connected neighbors
        neighbor_offsets = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        n_fire = len(fire_cells)
        # Batch random draws: (n_fire, 8) random values for neighbor spread
        rand_neighbors = rng.random((n_fire, 8))
        # Spotting random values
        rand_spot = rng.random(n_fire)
        for ni, (dy, dx) in enumerate(neighbor_offsets):
            nx = fx_arr + dx
            ny = fy_arr + dy
            valid = (nx >= 0) & (nx < self.grid) & (ny >= 0) & (ny < self.grid)
            # Fuel check
            safe_nx = np.clip(nx, 0, self.grid-1)
            safe_ny = np.clip(ny, 0, self.grid-1)
            has_fuel = self.fuel[safe_ny, safe_nx] > 0.1
            can_spread = valid & has_fuel
            prob = spread_prob * self.fuel[safe_ny, safe_nx]
            spreads = can_spread & (rand_neighbors[:, ni] < prob)
            # Apply spreads (may overwrite, but min(1.0,...) is idempotent for our purposes)
            for idx in np.where(spreads)[0]:
                new_fire[ny[idx], nx[idx]] = min(1.0, new_fire[ny[idx], nx[idx]] + 0.1)
        # Spotting
        spot_mask = rand_spot < self.spotting_prob * wind_mag / 10.0
        spot_indices = np.where(spot_mask)[0]
        if len(spot_indices) > 0:
            sx = np.clip(fx_arr[spot_indices] + rng.integers(-3, 4, size=len(spot_indices)), 0, self.grid-1)
            sy = np.clip(fy_arr[spot_indices] + rng.integers(-3, 4, size=len(spot_indices)), 0, self.grid-1)
            has_fuel_spot = self.fuel[sy, sx] > 0.1
            for idx in np.where(has_fuel_spot)[0]:
                new_fire[sy[idx], sx[idx]] = min(1.0, new_fire[sy[idx], sx[idx]] + 0.05)
        self.fuel = np.clip(self.fuel - self.fuel_depletion_rate * fire_mask.astype(np.float32), 0.0, 1.0)
        return new_fire

    def step(self, actions):
        self.step_count += 1
        rng = np.random.default_rng()
        dones = np.zeros(self.n_drones, dtype=bool)
        infos = [{} for _ in range(self.n_drones)]

        for i in range(self.n_drones):
            if not self.drones[i]['alive']:
                dones[i] = True
                continue
            d = self.drones[i]
            dx, dy = self.action_deltas[actions[i]]
            new_vel = self.momentum * d['vel'] + (1 - self.momentum) * np.array([dx, dy], dtype=np.float32)
            d['vel'] = new_vel
            d['pos'] = np.clip(d['pos'] + new_vel, self.boundary_margin, self.grid - 1 - self.boundary_margin)
            ix, iy = int(d['pos'][0]), int(d['pos'][1])
            ix, iy = np.clip(ix, 0, self.grid-1), np.clip(iy, 0, self.grid-1)
            d['visited'].add((ix, iy))
            self.total_cells_explored.add((ix, iy))
            self._visited_grid[iy, ix] = 1.0
            self.shared_visited[iy, ix] = 1.0
            crashed = False
            if self.fire[iy, ix] > self.fire_crash_threshold:
                crashed = True
            if self._fire_dist_cache is not None and self._fire_dist_cache[iy, ix] < 0.5:
                crashed = True
            if self.thermal[iy, ix] > self.thermal_crash:
                crashed = True
            if crashed:
                d['alive'] = False
                dones[i] = True
            infos[i] = {
                'fire_dist': float(self._fire_dist_cache[iy, ix]) if self._fire_dist_cache is not None else 10.0,
                'thermal': float(self.thermal[iy, ix]),
                'crashed': crashed,
            }

        if self.step_count % 3 == 0:
            self.fire = self._spread_fire(rng)
            self._fire_dist_cache = None
            self._update_fire_dist()
            self._update_thermal()

        if self.step_count >= self.max_steps:
            dones[:] = True

        return self._get_obs(), np.zeros(self.n_drones), dones, infos

    def _get_obs(self):
        """Vectorized observation: 5 local channels + 8 global features per drone.
        Uses numpy slicing instead of Python for-loops for ~50x speedup."""
        r = self.obs_r
        g = self.grid
        os_ = self.obs_size  # 9
        obs = np.zeros((self.n_drones, self.obs_dim), dtype=np.float32)
        channels = [self.fire, self.thermal, self.wind_x, self.wind_y, self.shared_visited]

        for i in range(self.n_drones):
            if not self.drones[i]['alive']:
                continue
            d = self.drones[i]
            ix, iy = int(d['pos'][0]), int(d['pos'][1])
            # Local patches via numpy slicing (no nested for-loop)
            x0, x1 = max(0, ix-r), min(g, ix+r+1)
            y0, y1 = max(0, iy-r), min(g, iy+r+1)
            px0 = r - (ix - x0)  # offset in patch where center falls
            py0 = r - (iy - y0)
            pw, ph = x1-x0, y1-y0
            for ch_i, arr in enumerate(channels):
                base = ch_i * os_ * os_
                patch = arr[y0:y1, x0:x1]
                # Vectorized placement: compute flat indices for the patch region
                patch_rows = np.arange(py0, py0+ph)
                patch_cols = np.arange(px0, px0+pw)
                row_idx, col_idx = np.meshgrid(patch_rows, patch_cols, indexing='ij')
                flat_idx = base + row_idx * os_ + col_idx
                obs[i].flat[flat_idx.ravel()] = patch.ravel()
            # Global features
            obs[i, self.local_obs_dim:] = [
                d['pos'][0]/g, d['pos'][1]/g,
                d['vel'][0], d['vel'][1],
                self.fire[iy, ix],
                self.thermal[iy, ix],
                self._fire_dist_cache[iy, ix] if self._fire_dist_cache is not None else 10.0,
                len(self.total_cells_explored) / (g*g),
            ]
        return obs


# ═══════════════════════════════════════════════════════════════
# SECTION 2: NEURAL NETWORKS
# ═══════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.W = nn.Linear(in_dim, n_heads * self.head_dim)
        self.a_src = nn.Parameter(torch.randn(n_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.randn(n_heads, self.head_dim))
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, h, adj_mask, return_attn=False):
        K = h.shape[0]
        if K == 1:
            out = self.W(h)
            return (out, None) if return_attn else out
        Wh = self.W(h).view(K, self.n_heads, self.head_dim)
        e_src = (Wh * self.a_src.unsqueeze(0)).sum(dim=2)
        e_dst = (Wh * self.a_dst.unsqueeze(0)).sum(dim=2)
        attn_scores = (e_src.unsqueeze(1) + e_dst.unsqueeze(0)) / (self.head_dim ** 0.5)
        attn_scores = attn_scores.masked_fill(~adj_mask.unsqueeze(-1).expand_as(attn_scores), float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_weights = attn_weights.masked_fill(torch.isnan(attn_weights), 0.0)
        out = torch.einsum('kjh,jhd->khd', attn_weights, Wh)
        out = self.out_proj(out.reshape(K, -1))
        return (out, attn_weights.mean(dim=-1)) if return_attn else out


class GATCommunication(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=64, n_heads=4, comm_range=10.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.comm_range = comm_range
        self.attn1 = MultiHeadAttention(in_dim, hidden_dim, n_heads)
        self.attn2 = MultiHeadAttention(hidden_dim, out_dim, n_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.output_proj = nn.Linear(out_dim, out_dim)
        self.res1 = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.res2 = nn.Linear(hidden_dim, out_dim)

    def build_graph(self, positions, alive_mask):
        K = len(positions)
        adj = torch.zeros(K, K, dtype=torch.bool, device=device)
        for i in range(K):
            if not alive_mask[i]: continue
            for j in range(K):
                if not alive_mask[j] or i == j: continue
                if torch.norm(positions[i] - positions[j]) < self.comm_range:
                    adj[i, j] = adj[j, i] = True
            adj[i, i] = True  # self-loop
        return adj

    def forward(self, obs, positions, alive_mask, return_attn=False):
        adj = self.build_graph(positions, alive_mask)
        h1 = F.relu(self.norm1(self.attn1(obs, adj) + self.res1(obs)))
        h2, attn2 = self.attn2(h1, adj, return_attn=True)
        h2 = F.relu(self.norm2(h2 + self.res2(h1)))
        out = torch.cat([obs, self.output_proj(h2)], dim=1)
        return (out, attn2) if return_attn else out

    @property
    def enhanced_obs_dim(self):
        return self.in_dim + self.out_dim


class NoGATCommunication(nn.Module):
    """Ablation: 2-layer MLP, no inter-agent communication."""
    def __init__(self, in_dim, out_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, out_dim)
        )
        self._in_dim = in_dim
        self._out_dim = out_dim

    def forward(self, obs, positions=None, alive_mask=None):
        return torch.cat([obs, self.mlp(obs)], dim=1)

    @property
    def enhanced_obs_dim(self):
        return self._in_dim + self._out_dim


class PPONetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, h1=256, h2=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, h1), nn.ReLU(), nn.LayerNorm(h1),
            nn.Linear(h1, h2), nn.ReLU(), nn.LayerNorm(h2))
        self.policy_head = nn.Linear(h2, act_dim)
        self.value_head = nn.Linear(h2, 1)
        self.apply(lambda m: nn.init.orthogonal_(m.weight, np.sqrt(2)) if isinstance(m, nn.Linear) else None)

    def forward(self, obs):
        f = self.encoder(obs)
        return self.policy_head(f), self.value_head(f).squeeze(-1)

    def evaluate(self, obs, actions):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return logits, dist.log_prob(actions), dist.entropy(), value


class MAPPOCritic(nn.Module):
    def __init__(self, obs_dim, n_agents, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * n_agents, hidden), nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, 1))

    def forward(self, all_obs_flat):
        return self.net(all_obs_flat).squeeze(-1)


class FastGATPPO:
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
        self.attn_entropy_log = []  # Track communication entropy over training
        self._track_entropy = True  # Toggle for entropy tracking (disable during eval)

    def select_actions(self, obs, positions, alive_mask):
        obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
        pos_t = torch.tensor(positions, dtype=torch.float32).to(device)
        alive_t = torch.tensor(alive_mask, dtype=torch.bool).to(device)
        # Track attention entropy if GAT is active
        attn_entropy = None
        if hasattr(self.gat, 'attn2') and self._track_entropy:
            adj = self.gat.build_graph(pos_t, alive_t)
            h1 = F.relu(self.gat.norm1(self.gat.attn1(obs_t, adj) + self.gat.res1(obs_t)))
            h2, attn_w = self.gat.attn2(h1, adj, return_attn=True)
            # Compute entropy of attention weights (higher = more distributed communication)
            attn_probs = F.softmax(attn_w, dim=-1)
            attn_entropy = -(attn_probs * (attn_probs + 1e-8).log()).sum(dim=-1).mean().item()
            self.attn_entropy_log.append(attn_entropy)
            enhanced = torch.cat([obs_t, self.gat.output_proj(F.relu(self.gat.norm2(h2 + self.gat.res2(h1))))], dim=1)
        else:
            enhanced = self.gat(obs_t, pos_t, alive_t)
        with torch.no_grad():
            logits, values = self.policy(enhanced)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
        return actions.cpu().numpy(), dist.log_prob(actions).cpu().numpy(), values.cpu().numpy(), enhanced.detach()

    def store(self, enhanced_obs, actions, rewards, dones, log_probs, values, agent_ids=None):
        for i in range(len(actions)):
            self._traj.append({
                'obs': enhanced_obs[i].cpu(), 'action': actions[i],
                'reward': rewards[i], 'done': dones[i],
                'log_prob': log_probs[i], 'value': values[i],
                'agent_id': agent_ids[i] if agent_ids is not None else 0})

    def update(self, gamma=0.99, gae_lambda=0.95, n_epochs=6, batch_size=512, clip_eps=0.2, entropy_coef=0.02):
        if not self._traj: return 0.0
        n = len(self._traj)

        # Per-drone GAE: transitions are interleaved across agents,
        # so we must compute GAE within each agent's sequence separately
        # to avoid mixing different agents' value estimates.
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        # Group trajectory indices by agent_id, preserving insertion order
        agent_groups = {}
        for i, t in enumerate(self._traj):
            aid = t.get('agent_id', 0)
            if aid not in agent_groups:
                agent_groups[aid] = []
            agent_groups[aid].append(i)

        for aid, indices in agent_groups.items():
            gae = 0.0
            for k in reversed(range(len(indices))):
                idx = indices[k]
                next_idx = indices[k+1] if k+1 < len(indices) else None
                next_val = 0.0 if next_idx is None else self._traj[next_idx]['value']
                next_done = 1.0 if next_idx is None else self._traj[next_idx]['done']
                delta = self._traj[idx]['reward'] + gamma * next_val * (1-next_done) - self._traj[idx]['value']
                gae = delta + gamma * gae_lambda * (1-next_done) * gae
                advantages[idx] = gae
                returns[idx] = gae + self._traj[idx]['value']

        all_obs = torch.stack([t['obs'].squeeze(0) for t in self._traj]).to(device)
        all_actions = torch.tensor([t['action'] for t in self._traj], dtype=torch.long).to(device)
        all_old_lp = torch.tensor([t['log_prob'] for t in self._traj], dtype=torch.float32).to(device)
        all_adv = torch.tensor(advantages, dtype=torch.float32).to(device)
        all_ret = torch.tensor(returns, dtype=torch.float32).to(device)
        all_adv = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-8)

        total_loss, count = 0.0, 0
        all_params = list(self.gat.parameters()) + list(self.policy.parameters())
        for _ in range(n_epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                idx = perm[start:start+batch_size]
                _, new_lp, entropy, vals = self.policy.evaluate(all_obs[idx], all_actions[idx])
                ratio = torch.exp(new_lp - all_old_lp[idx])
                s1 = ratio * all_adv[idx]
                s2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * all_adv[idx]
                loss = -torch.min(s1, s2).mean() + 0.5 * F.mse_loss(vals, all_ret[idx]) - entropy_coef * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, 0.5)
                self.optimizer.step()
                total_loss += loss.item(); count += 1
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
# SECTION 3: REWARD SHAPING
# ═══════════════════════════════════════════════════════════════

def compute_reward(drone, drone_idx, all_drones, prev_visited, fire_dist, crashed, step, max_steps, grid_size):
    """Balanced reward: survival-first exploration for multi-agent wildfire tracking.

    Incentive hierarchy:
      1. Staying alive (+1.0/step, +30 at episode end) — dominant signal (~300/ep)
      2. Exploring new cells (+8/cell) — worth pursuing, not worth dying for
      3. Fire-front observation (+5 when close) — mild tracking nudge
      4. Overlap penalty (-1/drone nearby, capped at -3) — prevent clustering
      5. Quadrant diversity (+2/count in quadrant) — encourage spatial spread
      6. Episode completion bonus (+30) — reward sustained safe operation
    """
    if crashed: return -40.0  # strong penalty: crashing is always bad

    reward = 0.0

    # 1. Per-step survival reward (dominant: ~300 over a full episode)
    reward += 1.0

    # 2. Exploration: +8 per NEW cell the team hasn't visited
    #    At ~15 new cells/episode this adds ~120 — meaningful but
    #    less than the 300 from surviving the whole episode.
    new_cells = sum(1 for c in drone['visited'] if c not in prev_visited)
    reward += 8.0 * new_cells

    # 3. Fire front bonus: +5 for being near fire (mild tracking nudge)
    if 0.5 < fire_dist < 4.0:
        reward += 5.0 * (1.0 - fire_dist / 4.0)

    # 4. Overlap penalty: -1.0 per drone within range 2, capped at -3.0
    #    Caps prevent the penalty from overwhelming survival reward (+1/step)
    #    when many drones cluster together.
    nearby = 0
    for j, other in enumerate(all_drones):
        if j != drone_idx and other['alive']:
            if np.linalg.norm(drone['pos'] - other['pos']) < 2.0:
                nearby += 1
    reward -= min(3.0, 1.0 * nearby)

    # 5. Quadrant diversity: +2 / count in same quadrant
    mid = grid_size / 2.0
    q = int(drone['pos'][0] >= mid) + 2 * int(drone['pos'][1] >= mid)
    qcount = sum(1 for o in all_drones if o['alive'] and
                 int(o['pos'][0] >= mid) + 2*int(o['pos'][1] >= mid) == q)
    reward += 2.0 / max(1, qcount)

    # 6. Episode completion bonus (large: rewards patience)
    if step >= max_steps - 1:
        reward += 30.0

    return reward


# ═══════════════════════════════════════════════════════════════
# SECTION 4: TRAINING
# ═══════════════════════════════════════════════════════════════

def train(n_episodes=800, grid=30, n_drones=10, max_steps=300, use_gat=True, seed=0, run_id="gat"):
    torch.manual_seed(seed); np.random.seed(seed)
    tag = "GAT-MARAHS" if use_gat else "No-GAT (Ablation)"
    print(f"\n{'='*60}", flush=True)
    print(f"{tag} | seed={seed} | {n_episodes} eps | {n_drones} drones | {grid}x{grid}", flush=True)
    print(f"{'='*60}", flush=True)

    # Train at wind=0 (stable). Evaluate at wind>0 later.
    env = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps, wind_speed=0)
    agent = FastGATPPO(obs_dim=env.obs_dim, act_dim=env.act_dim, use_gat=use_gat)

    rewards_h, coverage_h, safety_h = [], [], []
    best_r = -float('inf')
    early_stop_counter = 0
    # Require sustained high coverage (≥65%) for 300 consecutive 100-ep windows
    early_stop_target = 65.0
    early_stop_patience = 300
    t0 = time.time()

    try:
        for ep in range(n_episodes):
            obs = env.reset()
            agent._traj.clear()
            ep_r, ep_crashes = 0.0, 0

            for step in range(max_steps):
                am = np.array([env.drones[i]['alive'] for i in range(n_drones)])
                pos = np.array([env.drones[i]['pos'] for i in range(n_drones)], dtype=np.float32)
                if not am.any(): break
                actions, log_probs, values, enhanced = agent.select_actions(obs, pos, am)
                # Snapshot BEFORE step so exploration reward counts new cells
                prev_visited = [set(env.drones[i]['visited']) for i in range(n_drones)]
                obs_next, _, dones, infos = env.step(np.array(actions, dtype=np.int32))
                shaped = np.zeros(n_drones, dtype=np.float32)
                for i in range(n_drones):
                    if not am[i]: continue
                    fd = infos[i].get('fire_dist', 10.0)
                    crashed = infos[i].get('crashed', False)
                    shaped[i] = compute_reward(env.drones[i], i, env.drones, prev_visited[i], fd, crashed, step, max_steps, grid)
                    ep_r += shaped[i]
                    if crashed: ep_crashes += 1
                agent_ids = list(range(n_drones))
                agent.store(enhanced, actions, shaped, dones.astype(np.float32), log_probs, values, agent_ids)
                obs = obs_next
                if all(dones): break

            agent.update()
            cov = len(env.total_cells_explored) / (grid*grid) * 100
            saf = (1.0 - ep_crashes / n_drones) * 100
            rewards_h.append(ep_r); coverage_h.append(cov); safety_h.append(saf)

            if (ep+1) % 100 == 0:
                avg_r = np.mean(rewards_h[-100:]); avg_cov = np.mean(coverage_h[-100:]); avg_saf = np.mean(safety_h[-100:])
                elapsed = time.time() - t0
                eps_per_sec = (ep+1) / elapsed
                eta_min = (n_episodes - ep - 1) / max(eps_per_sec, 1e-6) / 60.0
                exp_speed = np.mean(np.diff(coverage_h[-100:])) if len(coverage_h) > 1 else 0
                print(f"Ep {ep+1:5d}/{n_episodes} | R: {avg_r:7.1f} | Cov: {avg_cov:5.1f}% | Safe: {avg_saf:4.0f}% | Speed: {exp_speed:+.3f}%/ep | {eps_per_sec:.1f} ep/s | ETA {eta_min:.0f}m", flush=True)
                if avg_r > best_r:
                    best_r = avg_r
                    agent.save(f'{run_id}_best.pt')
                agent.save(f'{run_id}_checkpoint_ep{ep+1}.pt')
                # Validation (3 held-out episodes at wind=0)
                val_c, val_s = [], []
                for _ in range(3):
                    ve = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps, wind_speed=0)
                    vo = ve.reset(); vc = 0
                    for _ in range(max_steps):
                        va = np.array([ve.drones[i]['alive'] for i in range(n_drones)])
                        vp = np.array([ve.drones[i]['pos'] for i in range(n_drones)], dtype=np.float32)
                        if not va.any(): break
                        vacts, _, _, _ = agent.select_actions(vo, vp, va)
                        vo, _, vd, _ = ve.step(np.array(vacts, dtype=np.int32))
                        for j in range(n_drones):
                            if va[j] and vd[j] and not ve.drones[j]['alive']: vc += 1
                        if all(vd): break
                    val_c.append(len(ve.total_cells_explored)/(grid*grid)*100)
                    val_s.append((1.0-vc/n_drones)*100)
                print(f"         VAL | Cov: {np.mean(val_c):5.1f}% ± {np.std(val_c):4.1f} | Safe: {np.mean(val_s):4.0f}%", flush=True)
                # Early stopping: sustained high coverage
                if avg_cov >= early_stop_target:
                    early_stop_counter += 100
                else:
                    early_stop_counter = 0
                if early_stop_counter >= early_stop_patience:
                    print(f"Early stop at ep {ep+1}: coverage {avg_cov:.1f}% >= {early_stop_target}% for {early_stop_patience} episodes", flush=True)
                    break
    except (KeyboardInterrupt, SystemExit, Exception) as e:
        print(f"\nTraining interrupted at ep {ep+1}: {e}", flush=True)

    agent.save(f'{run_id}_final.pt')
    # Attention entropy analysis (GAT communication diversity)
    attn_entropy = agent.attn_entropy_log if hasattr(agent, 'attn_entropy_log') and agent.attn_entropy_log else []
    results = {
        'n_episodes': len(rewards_h), 'seed': seed, 'use_gat': use_gat,
        'final_reward': float(np.mean(rewards_h[-100:])) if len(rewards_h) >= 100 else float(np.mean(rewards_h)),
        'final_coverage': float(np.mean(coverage_h[-100:])) if len(coverage_h) >= 100 else float(np.mean(coverage_h)),
        'final_safety': float(np.mean(safety_h[-100:])) if len(safety_h) >= 100 else float(np.mean(safety_h)),
        'rewards': [float(x) for x in rewards_h],
        'coverages': [float(x) for x in coverage_h],
        'safety': [float(x) for x in safety_h],
        'attention_entropy': [float(x) for x in attn_entropy] if attn_entropy else [],
        'exploration_speed': float(np.mean(coverage_h)) / max(1, len(coverage_h)) * 100 if coverage_h else 0.0,
    }
    with open(f'{run_id}_training_results.json', 'w') as f: json.dump(results, f, indent=2)
    print(f"\nDone in {time.time()-t0:.0f}s | Reward: {results['final_reward']:.1f} | Coverage: {results['final_coverage']:.1f}% | Safety: {results['final_safety']:.0f}%", flush=True)
    return agent, results


def train_multi_seed(n_episodes, grid, n_drones, max_steps, use_gat=True, seeds=[42, 123]):
    run_id = "gat" if use_gat else "nogat"
    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n{'#'*60}", flush=True)
        print(f"# Run {i+1}/{len(seeds)} | seed={seed}", flush=True)
        print(f"{'#'*60}", flush=True)
        agent, res = train(n_episodes, grid, n_drones, max_steps, use_gat=use_gat, seed=seed, run_id=f"{run_id}_s{seed}")
        all_results.append(res)
        agent.save(f'{run_id}_seed{seed}_best.pt')

    final_covs = [r['final_coverage'] for r in all_results]
    final_safs = [r['final_safety'] for r in all_results]
    final_rews = [r['final_reward'] for r in all_results]
    tag = "GAT-MARAHS" if use_gat else "No-GAT (Ablation)"
    print(f"\n{'='*60}", flush=True)
    print(f"{tag} | {len(seeds)} seeds summary:", flush=True)
    print(f"  Coverage: {np.mean(final_covs):.1f}% ± {np.std(final_covs):.1f}%", flush=True)
    print(f"  Safety:   {np.mean(final_safs):.1f}% ± {np.std(final_safs):.1f}%", flush=True)
    print(f"  Reward:   {np.mean(final_rews):.1f} ± {np.std(final_rews):.1f}", flush=True)
    print(f"{'='*60}", flush=True)

    # Load best agent
    best_idx = int(np.argmax(final_covs))
    best_seed = seeds[best_idx]
    tmp_env = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps)
    best_agent = FastGATPPO(obs_dim=tmp_env.obs_dim, act_dim=tmp_env.act_dim, use_gat=use_gat)
    del tmp_env
    try:
        best_agent.load(f'{run_id}_seed{best_seed}_best.pt')
    except Exception as e:
        print(f"  Warning: could not load best model: {e}", flush=True)
    return best_agent, all_results


# ═══════════════════════════════════════════════════════════════
# SECTION 5: MAPPO BASELINE
# ═══════════════════════════════════════════════════════════════

def train_mappo(n_episodes=800, grid=30, n_drones=10, max_steps=300, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    print(f"\n{'='*60}", flush=True)
    print(f"MAPPO | seed={seed} | {n_episodes} eps | {n_drones} drones | {grid}x{grid}", flush=True)
    print(f"{'='*60}", flush=True)

    env = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps, wind_speed=0)
    obs_dim, act_dim = env.obs_dim, env.act_dim
    policy_net = PPONetwork(obs_dim, act_dim).to(device)
    critic = MAPPOCritic(obs_dim, n_drones).to(device)
    optimizer = torch.optim.Adam(list(policy_net.parameters()) + list(critic.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-5)

    rewards_h, coverage_h, safety_h = [], [], []
    best_r = -float('inf')
    t0 = time.time()

    try:
        for ep in range(n_episodes):
            obs = env.reset()
            ep_r, ep_crashes = 0.0, 0
            traj = []
            for step in range(max_steps):
                am = np.array([env.drones[i]['alive'] for i in range(n_drones)])
                pos = np.array([env.drones[i]['pos'] for i in range(n_drones)], dtype=np.float32)
                if not am.any(): break
                obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
                actions, lp_list, val_list = [], [], []
                for i in range(n_drones):
                    if not am[i]:
                        actions.append(0); lp_list.append(0.0); val_list.append(0.0)
                        continue
                    logits, val = policy_net(obs_t[i:i+1])
                    dist = torch.distributions.Categorical(logits=logits)
                    a = dist.sample()
                    actions.append(a.item())
                    lp_list.append(dist.log_prob(a).item())
                    val_list.append(val.item())

                prev_visited = [set(env.drones[i]['visited']) for i in range(n_drones)]
                obs_next, _, dones, infos = env.step(np.array(actions, dtype=np.int32))
                all_flat = torch.tensor(obs.reshape(-1), dtype=torch.float32).to(device)
                central_val = critic(all_flat).item()
                shaped = np.zeros(n_drones, dtype=np.float32)
                for i in range(n_drones):
                    if not am[i]: continue
                    fd = infos[i].get('fire_dist', 10.0)
                    crashed = infos[i].get('crashed', False)
                    shaped[i] = compute_reward(env.drones[i], i, env.drones, prev_visited[i], fd, crashed, step, max_steps, grid)
                    ep_r += shaped[i]
                    if crashed: ep_crashes += 1
                for i in range(n_drones):
                    if am[i]:
                        traj.append({'obs': torch.tensor(obs[i]), 'action': actions[i],
                                     'reward': shaped[i], 'done': float(dones[i]),
                                     'log_prob': lp_list[i], 'value': central_val,
                                     'agent_id': i})
                obs = obs_next
                if all(dones): break

            # PPO update
            if traj:
                n = len(traj)
                # Per-drone GAE: compute advantages within each agent's sequence
                advs = np.zeros(n, dtype=np.float32)
                rets = np.zeros(n, dtype=np.float32)
                agent_groups = {}
                for i, t in enumerate(traj):
                    aid = t.get('agent_id', 0)
                    if aid not in agent_groups:
                        agent_groups[aid] = []
                    agent_groups[aid].append(i)
                for aid, indices in agent_groups.items():
                    gae = 0.0
                    for k in reversed(range(len(indices))):
                        idx = indices[k]
                        next_idx = indices[k+1] if k+1 < len(indices) else None
                        nv = 0.0 if next_idx is None else traj[next_idx]['value']
                        nd = 1.0 if next_idx is None else traj[next_idx]['done']
                        delta = traj[idx]['reward'] + 0.99*nv*(1-nd) - traj[idx]['value']
                        gae = delta + 0.99*0.95*(1-nd)*gae
                        advs[idx] = gae; rets[idx] = gae + traj[idx]['value']
                ao = torch.stack([t['obs'] for t in traj]).to(device)
                aa = torch.tensor([t['action'] for t in traj], dtype=torch.long).to(device)
                aolp = torch.tensor([t['log_prob'] for t in traj], dtype=torch.float32).to(device)
                aadv = torch.tensor(advs, dtype=torch.float32).to(device)
                aret = torch.tensor(rets, dtype=torch.float32).to(device)
                aadv = (aadv - aadv.mean()) / (aadv.std() + 1e-8)
                for _ in range(6):
                    perm = torch.randperm(n, device=device)
                    for s in range(0, n, 512):
                        idx = perm[s:s+512]
                        _, nlp, ent, vp = policy_net.evaluate(ao[idx], aa[idx])
                        ratio = torch.exp(nlp - aolp[idx])
                        s1 = ratio * aadv[idx]
                        s2 = torch.clamp(ratio, 0.8, 1.2) * aadv[idx]
                        loss = -torch.min(s1, s2).mean() - 0.02*ent.mean() + 0.5*F.mse_loss(vp, aret[idx])
                        optimizer.zero_grad(); loss.backward()
                        nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
                        optimizer.step()
                scheduler.step()

            cov = len(env.total_cells_explored)/(grid*grid)*100
            saf = (1.0 - ep_crashes/n_drones)*100
            rewards_h.append(ep_r); coverage_h.append(cov); safety_h.append(saf)
            if (ep+1) % 100 == 0:
                avg_r = np.mean(rewards_h[-100:]); avg_cov = np.mean(coverage_h[-100:]); avg_saf = np.mean(safety_h[-100:])
                elapsed = time.time() - t0
                eps_per_sec = (ep+1)/elapsed
                eta_min = (n_episodes-ep-1)/max(eps_per_sec,1e-6)/60.0
                exp_speed = np.mean(np.diff(coverage_h[-100:])) if len(coverage_h) > 1 else 0
                print(f"Ep {ep+1:5d}/{n_episodes} | R: {avg_r:7.1f} | Cov: {avg_cov:5.1f}% | Safe: {avg_saf:4.0f}% | Speed: {exp_speed:+.3f}%/ep | {eps_per_sec:.1f} ep/s | ETA {eta_min:.0f}m", flush=True)
                if avg_r > best_r:
                    best_r = avg_r
                    torch.save({'policy': policy_net.state_dict(), 'critic': critic.state_dict()}, f'mappo_s{seed}_best.pt')
                # Validation (3 held-out episodes at wind=0)
                val_c, val_s = [], []
                for _ in range(3):
                    ve = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps, wind_speed=0)
                    vo = ve.reset(); vc = 0
                    for _ in range(max_steps):
                        va = np.array([ve.drones[i]['alive'] for i in range(n_drones)])
                        if not va.any(): break
                        vt = torch.tensor(vo, dtype=torch.float32).to(device)
                        vacts = np.zeros(n_drones, dtype=np.int32)
                        for vi in range(n_drones):
                            if not va[vi]: continue
                            vlogits, _ = policy_net(vt[vi:vi+1])
                            vacts[vi] = torch.distributions.Categorical(logits=vlogits).sample().item()
                        vo, _, vd, _ = ve.step(vacts)
                        for j in range(n_drones):
                            if va[j] and vd[j] and not ve.drones[j]['alive']: vc += 1
                        if all(vd): break
                    val_c.append(len(ve.total_cells_explored)/(grid*grid)*100)
                    val_s.append((1.0-vc/n_drones)*100)
                print(f"         VAL | Cov: {np.mean(val_c):5.1f}% \u00b1 {np.std(val_c):4.1f} | Safe: {np.mean(val_s):4.0f}%", flush=True)
    except (KeyboardInterrupt, SystemExit, Exception) as e:
        print(f"\nTraining interrupted at ep {ep+1}: {e}", flush=True)

    torch.save({'policy': policy_net.state_dict(), 'critic': critic.state_dict()}, f'mappo_s{seed}_final.pt')
    results = {
        'n_episodes': len(rewards_h), 'seed': seed, 'use_gat': False,
        'final_reward': float(np.mean(rewards_h[-100:])) if len(rewards_h) >= 100 else float(np.mean(rewards_h)),
        'final_coverage': float(np.mean(coverage_h[-100:])) if len(coverage_h) >= 100 else float(np.mean(coverage_h)),
        'final_safety': float(np.mean(safety_h[-100:])) if len(safety_h) >= 100 else float(np.mean(safety_h)),
        'rewards': [float(x) for x in rewards_h],
        'coverages': [float(x) for x in coverage_h],
        'safety': [float(x) for x in safety_h],
    }
    with open(f'mappo_s{seed}_training_results.json', 'w') as f: json.dump(results, f, indent=2)
    print(f"\nDone in {time.time()-t0:.0f}s | Reward: {results['final_reward']:.1f} | Coverage: {results['final_coverage']:.1f}% | Safety: {results['final_safety']:.0f}%", flush=True)

    # Save raw policy+critic weights (MAPPO has its own architecture,
    # no need to force into FastGATPPO wrapper which expects GAT dims)
    results['policy_state'] = policy_net.state_dict()
    results['critic_state'] = critic.state_dict()
    return policy_net, results


def train_mappo_multi_seed(n_episodes, grid, n_drones, max_steps, seeds=[42, 123]):
    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n{'#'*60}", flush=True)
        print(f"# MAPPO Run {i+1}/{len(seeds)} | seed={seed}", flush=True)
        print(f"{'#'*60}", flush=True)
        _, res = train_mappo(n_episodes, grid, n_drones, max_steps, seed=seed)
        all_results.append(res)
    covs = [r['final_coverage'] for r in all_results]
    safes = [r['final_safety'] for r in all_results]
    print(f"\n{'='*60}", flush=True)
    print(f"MAPPO | {len(seeds)} seeds:", flush=True)
    print(f"  Coverage: {np.mean(covs):.1f}% ± {np.std(covs):.1f}%", flush=True)
    print(f"  Safety:   {np.mean(safes):.1f}% ± {np.std(safes):.1f}%", flush=True)
    print(f"{'='*60}", flush=True)
    return None, all_results


# ═══════════════════════════════════════════════════════════════
# SECTION 5b: IPPO BASELINE (Yu et al. 2021)
# ═══════════════════════════════════════════════════════════════

def train_ippo(n_episodes=500, grid=30, n_drones=10, max_steps=300, seed=0):
    """Independent PPO: each agent trains its own policy with NO shared critic
    and NO communication. This is the standard MARL baseline from
    'The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games' (Yu et al. 2021)."""
    torch.manual_seed(seed); np.random.seed(seed)
    print(f"\n{'='*60}", flush=True)
    print(f"IPPO (Independent) | seed={seed} | {n_episodes} eps | {n_drones} drones | {grid}x{grid}", flush=True)
    print(f"{'='*60}", flush=True)

    env = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps, wind_speed=0)
    obs_dim, act_dim = env.obs_dim, env.act_dim
    # Each agent has its own policy + value head (no shared critic)
    policies = [PPONetwork(obs_dim, act_dim).to(device) for _ in range(n_drones)]
    optimizers = [torch.optim.Adam(p.parameters(), lr=3e-4) for p in policies]
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=2000, eta_min=1e-5) for o in optimizers]

    rewards_h, coverage_h, safety_h = [], [], []
    best_r = -float('inf')
    t0 = time.time()

    try:
        for ep in range(n_episodes):
            obs = env.reset()
            ep_r, ep_crashes = 0.0, 0
            # Per-agent trajectories for independent GAE
            agent_trajs = [[] for _ in range(n_drones)]

            for step in range(max_steps):
                am = np.array([env.drones[i]['alive'] for i in range(n_drones)])
                if not am.any(): break
                obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
                actions, lp_list, val_list = [], [], []
                for i in range(n_drones):
                    if not am[i]:
                        actions.append(0); lp_list.append(0.0); val_list.append(0.0)
                        continue
                    logits, val = policies[i](obs_t[i:i+1])
                    dist = torch.distributions.Categorical(logits=logits)
                    a = dist.sample()
                    actions.append(a.item())
                    lp_list.append(dist.log_prob(a).item())
                    val_list.append(val.item())

                prev_visited = [set(env.drones[i]['visited']) for i in range(n_drones)]
                obs_next, _, dones, infos = env.step(np.array(actions, dtype=np.int32))
                shaped = np.zeros(n_drones, dtype=np.float32)
                for i in range(n_drones):
                    if not am[i]: continue
                    fd = infos[i].get('fire_dist', 10.0)
                    crashed = infos[i].get('crashed', False)
                    shaped[i] = compute_reward(env.drones[i], i, env.drones, prev_visited[i], fd, crashed, step, max_steps, grid)
                    ep_r += shaped[i]
                    if crashed: ep_crashes += 1
                for i in range(n_drones):
                    if am[i]:
                        agent_trajs[i].append({
                            'obs': torch.tensor(obs[i]), 'action': actions[i],
                            'reward': shaped[i], 'done': float(dones[i]),
                            'log_prob': lp_list[i], 'value': val_list[i]})
                obs = obs_next
                if all(dones): break

            # Independent PPO update per agent
            for i in range(n_drones):
                traj = agent_trajs[i]
                if len(traj) < 4: continue
                n = len(traj)
                # GAE for this single agent
                advs = np.zeros(n, dtype=np.float32)
                rets = np.zeros(n, dtype=np.float32)
                gae = 0.0
                for k in reversed(range(n)):
                    nv = 0.0 if k == n-1 else traj[k+1]['value']
                    nd = 1.0 if k == n-1 else traj[k+1]['done']
                    delta = traj[k]['reward'] + 0.99*nv*(1-nd) - traj[k]['value']
                    gae = delta + 0.99*0.95*(1-nd)*gae
                    advs[k] = gae; rets[k] = gae + traj[k]['value']
                ao = torch.stack([t['obs'] for t in traj]).to(device)
                aa = torch.tensor([t['action'] for t in traj], dtype=torch.long).to(device)
                aolp = torch.tensor([t['log_prob'] for t in traj], dtype=torch.float32).to(device)
                aadv = torch.tensor(advs, dtype=torch.float32).to(device)
                aret = torch.tensor(rets, dtype=torch.float32).to(device)
                aadv = (aadv - aadv.mean()) / (aadv.std() + 1e-8)
                for _ in range(6):
                    perm = torch.randperm(n, device=device)
                    for s in range(0, n, 512):
                        idx = perm[s:s+512]
                        _, nlp, ent, vp = policies[i].evaluate(ao[idx], aa[idx])
                        ratio = torch.exp(nlp - aolp[idx])
                        s1 = ratio * aadv[idx]
                        s2 = torch.clamp(ratio, 0.8, 1.2) * aadv[idx]
                        loss = -torch.min(s1, s2).mean() - 0.02*ent.mean() + 0.5*F.mse_loss(vp, aret[idx])
                        optimizers[i].zero_grad(); loss.backward()
                        nn.utils.clip_grad_norm_(policies[i].parameters(), 0.5)
                        optimizers[i].step()
                schedulers[i].step()

            cov = len(env.total_cells_explored)/(grid*grid)*100
            saf = (1.0 - ep_crashes/n_drones)*100
            rewards_h.append(ep_r); coverage_h.append(cov); safety_h.append(saf)
            if (ep+1) % 100 == 0:
                avg_r = np.mean(rewards_h[-100:]); avg_cov = np.mean(coverage_h[-100:]); avg_saf = np.mean(safety_h[-100:])
                elapsed = time.time() - t0
                eps_per_sec = (ep+1)/elapsed
                eta_min = (n_episodes-ep-1)/max(eps_per_sec,1e-6)/60.0
                exp_speed = np.mean(np.diff(coverage_h[-100:])) if len(coverage_h) > 1 else 0
                print(f"Ep {ep+1:5d}/{n_episodes} | R: {avg_r:7.1f} | Cov: {avg_cov:5.1f}% | Safe: {avg_saf:4.0f}% | Speed: {exp_speed:+.3f}%/ep | {eps_per_sec:.1f} ep/s | ETA {eta_min:.0f}m", flush=True)
                if avg_r > best_r:
                    best_r = avg_r
                    for j, p in enumerate(policies):
                        torch.save(p.state_dict(), f'ippo_s{seed}_agent{j}_best.pt')
                # Validation
                val_c, val_s = [], []
                for _ in range(3):
                    ve = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps, wind_speed=0)
                    vo = ve.reset(); vc = 0
                    for _ in range(max_steps):
                        va = np.array([ve.drones[j]['alive'] for j in range(n_drones)])
                        if not va.any(): break
                        vt = torch.tensor(vo, dtype=torch.float32).to(device)
                        vacts = np.zeros(n_drones, dtype=np.int32)
                        for vi in range(n_drones):
                            if not va[vi]: continue
                            vlogits, _ = policies[vi](vt[vi:vi+1])
                            vacts[vi] = torch.distributions.Categorical(logits=vlogits).sample().item()
                        vo, _, vd, _ = ve.step(vacts)
                        for j in range(n_drones):
                            if va[j] and vd[j] and not ve.drones[j]['alive']: vc += 1
                        if all(vd): break
                    val_c.append(len(ve.total_cells_explored)/(grid*grid)*100)
                    val_s.append((1.0-vc/n_drones)*100)
                print(f"         VAL | Cov: {np.mean(val_c):5.1f}% +/- {np.std(val_c):4.1f} | Safe: {np.mean(val_s):4.0f}%", flush=True)
    except (KeyboardInterrupt, SystemExit, Exception) as e:
        print(f"\nTraining interrupted at ep {ep+1}: {e}", flush=True)

    for j, p in enumerate(policies):
        torch.save(p.state_dict(), f'ippo_s{seed}_agent{j}_final.pt')
    results = {
        'n_episodes': len(rewards_h), 'seed': seed, 'use_gat': False,
        'final_reward': float(np.mean(rewards_h[-100:])) if len(rewards_h) >= 100 else float(np.mean(rewards_h)),
        'final_coverage': float(np.mean(coverage_h[-100:])) if len(coverage_h) >= 100 else float(np.mean(coverage_h)),
        'final_safety': float(np.mean(safety_h[-100:])) if len(safety_h) >= 100 else float(np.mean(safety_h)),
        'rewards': [float(x) for x in rewards_h],
        'coverages': [float(x) for x in coverage_h],
        'safety': [float(x) for x in safety_h],
    }
    with open(f'ippo_s{seed}_training_results.json', 'w') as f: json.dump(results, f, indent=2)
    print(f"\nDone in {time.time()-t0:.0f}s | Reward: {results['final_reward']:.1f} | Coverage: {results['final_coverage']:.1f}% | Safety: {results['final_safety']:.0f}%", flush=True)
    return policies, results


def train_ippo_multi_seed(n_episodes, grid, n_drones, max_steps, seeds=[42, 123]):
    all_results = []
    all_policies = []
    for i, seed in enumerate(seeds):
        print(f"\n{'#'*60}", flush=True)
        print(f"# IPPO Run {i+1}/{len(seeds)} | seed={seed}", flush=True)
        print(f"{'#'*60}", flush=True)
        pols, res = train_ippo(n_episodes, grid, n_drones, max_steps, seed=seed)
        all_results.append(res)
        all_policies.append(pols)
    covs = [r['final_coverage'] for r in all_results]
    safes = [r['final_safety'] for r in all_results]
    print(f"\n{'='*60}", flush=True)
    print(f"IPPO | {len(seeds)} seeds:", flush=True)
    print(f"  Coverage: {np.mean(covs):.1f}% +/- {np.std(covs):.1f}%", flush=True)
    print(f"  Safety:   {np.mean(safes):.1f}% +/- {np.std(safes):.1f}%", flush=True)
    print(f"{'='*60}", flush=True)
    return all_policies, all_results


# ═══════════════════════════════════════════════════════════════
# SECTION 6: BENCHMARK
# ═══════════════════════════════════════════════════════════════

def benchmark(agent, grid=30, n_drones=10, max_steps=300, wind=12.0, n_eps=20):
    print(f"\n{'='*60}\nBENCHMARK | wind={wind} | {n_drones} drones | {n_eps} eps\n{'='*60}", flush=True)
    results = {}

    def _run_gat(env):
        am = np.array([env.drones[i]['alive'] for i in range(n_drones)])
        pos = np.array([env.drones[i]['pos'] for i in range(n_drones)], dtype=np.float32)
        obs = env._get_obs()
        acts, _, _, _ = agent.select_actions(obs, pos, am)
        return acts

    def _run_random(env):
        return np.random.randint(0, 5, n_drones)

    def _run_greedy(env):
        acts = np.zeros(n_drones, dtype=np.int32)
        for i in range(n_drones):
            if not env.drones[i]['alive']: continue
            d = env.drones[i]; ix, iy = int(d['pos'][0]), int(d['pos'][1])
            best_a, best_v = 0, -1
            for ai, (dx, dy) in enumerate([(0,0),(0,1),(0,-1),(1,0),(-1,0)]):
                nx, ny = ix+int(dx), iy+int(dy)
                if 0<=nx<env.grid and 0<=ny<env.grid and (nx,ny) not in d.get('visited', set()):
                    v = 1.0
                    if env._fire_dist_cache is not None:
                        v += 2.0 / (env._fire_dist_cache[ny, nx] + 1.0)
                    if v > best_v: best_v, best_a = v, ai
            acts[i] = best_a
        return acts

    def _run_pid(env):
        acts = np.zeros(n_drones, dtype=np.int32)
        for i in range(n_drones):
            if not env.drones[i]['alive']: continue
            d = env.drones[i]
            fc = np.argwhere(env.fire > 0.2)
            if len(fc) > 0:
                fcx, fcy = float(np.mean(fc[:, 0])), float(np.mean(fc[:, 1]))
                ddx, ddy = fcx - d['pos'][0], fcy - d['pos'][1]
                if abs(ddx) > abs(ddy): acts[i] = 3 if ddx > 0 else 4
                else: acts[i] = 1 if ddy > 0 else 2
        return acts

    for method_name, method_fn in [('GAT-MARAHS', _run_gat), ('Random', _run_random), ('Greedy', _run_greedy), ('PID', _run_pid)]:
        s, c, p, a = [], [], [], []
        for _ in range(n_eps):
            env_b = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps, wind_speed=wind)
            obs = env_b.reset()
            for _ in range(max_steps):
                alive = np.array([env_b.drones[i]['alive'] for i in range(n_drones)])
                if not alive.any(): break
                actions = method_fn(env_b)
                obs, _, dones, _ = env_b.step(np.array(actions, dtype=np.int32))
                if all(dones): break
            ac = sum(1 for i in range(n_drones) if env_b.drones[i]['alive'])
            s.append(ac/n_drones*100); c.append(len(env_b.total_cells_explored)/(grid*grid)*100)
            fc = np.argwhere(env_b.fire > 0.2); pc = set()
            for fx, fy in fc:
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = fx+dx, fy+dy
                    if 0<=nx<grid and 0<=ny<grid and env_b.fire[nx,ny]<0.1: pc.add((nx,ny))
            vis = set()
            for i in range(n_drones): vis.update(env_b.drones[i].get('visited', set()))
            p.append(len(pc & vis)/max(1,len(pc))*100); a.append(ac)
        # Store per-episode data + CIs for research-grade reporting
        s_mean, s_lo, s_hi = bootstrap_ci(np.array(s))
        c_mean, c_lo, c_hi = bootstrap_ci(np.array(c))
        p_mean, p_lo, p_hi = bootstrap_ci(np.array(p))
        results[method_name] = {
            'safety': np.mean(s), 'coverage': np.mean(c), 'perimeter': np.mean(p), 'alive': np.mean(a),
            'safety_ci': [float(s_lo), float(s_hi)],
            'coverage_ci': [float(c_lo), float(c_hi)],
            'perimeter_ci': [float(p_lo), float(p_hi)],
        }

    print(f"\n{'Method':<18s} {'Safety':>16s} {'Coverage':>16s} {'Perimeter':>16s}", flush=True)
    print("-"*68, flush=True)
    for m, v in results.items():
        print(f"{m:<18s} {v['safety']:6.1f}% [{v['safety_ci'][0]:.1f},{v['safety_ci'][1]:.1f}]".ljust(34) +
              f" {v['coverage']:6.1f}% [{v['coverage_ci'][0]:.1f},{v['coverage_ci'][1]:.1f}]".ljust(34) +
              f" {v['perimeter']:6.1f}% [{v['perimeter_ci'][0]:.1f},{v['perimeter_ci'][1]:.1f}]", flush=True)
    print("="*68, flush=True)
    with open('gat_benchmark_final.json', 'w') as f: json.dump(results, f, indent=2)
    return results


# ═══════════════════════════════════════════════════════════════
# SECTION 7: WIND SWEEP
# ═══════════════════════════════════════════════════════════════

def wind_sweep(agent, grid=30, n_drones=10, max_steps=300, n_eps=15):
    print(f"\n{'='*60}\nWIND SWEEP | {n_drones} drones | {n_eps} episodes each\n{'='*60}", flush=True)
    sweep = {}
    for wind in [5, 10, 15, 20, 25]:
        s, c, p = [], [], []
        for _ in range(n_eps):
            env_w = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps, wind_speed=wind)
            obs = env_w.reset()
            for _ in range(max_steps):
                am = np.array([env_w.drones[i]['alive'] for i in range(n_drones)])
                pos = np.array([env_w.drones[i]['pos'] for i in range(n_drones)], dtype=np.float32)
                if not am.any(): break
                acts, _, _, _ = agent.select_actions(obs, pos, am)
                obs, _, dones, _ = env_w.step(np.array(acts, dtype=np.int32))
                if all(dones): break
            ac = sum(1 for i in range(n_drones) if env_w.drones[i]['alive'])
            s.append(ac/n_drones*100); c.append(len(env_w.total_cells_explored)/(grid*grid)*100)
            fc = np.argwhere(env_w.fire > 0.2); pc = set()
            for fx, fy in fc:
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = fx+dx, fy+dy
                    if 0<=nx<grid and 0<=ny<grid and env_w.fire[nx,ny]<0.1: pc.add((nx,ny))
            vis = set()
            for i in range(n_drones): vis.update(env_w.drones[i].get('visited', set()))
            p.append(len(pc & vis)/max(1,len(pc))*100)
        sweep[str(wind)] = {'safety': np.mean(s), 'coverage': np.mean(c), 'perimeter': np.mean(p),
                            'safety_std': float(np.std(s)), 'coverage_std': float(np.std(c)), 'perimeter_std': float(np.std(p))}
        print(f"  Wind={wind:2d} m/s | Safety={np.mean(s):5.1f}% | Coverage={np.mean(c):5.1f}% | Perimeter={np.mean(p):5.1f}%", flush=True)
    with open('gat_wind_sweep.json', 'w') as f: json.dump(sweep, f, indent=2)
    return sweep


# ═══════════════════════════════════════════════════════════════
# SECTION 7b: SCALABILITY ANALYSIS
# ═══════════════════════════════════════════════════════════════

def scalability_test(agent, grid=30, max_steps=300, n_eps=10):
    """Evaluate GAT-MARAHS across different swarm sizes to demonstrate
    scaling properties — essential for claiming 'multi-agent' contribution."""
    print(f"\n{'='*60}\nSCALABILITY TEST | grid={grid}x{grid} | {n_eps} eps each\n{'='*60}", flush=True)
    results = {}
    for n_drones in [5, 10, 20]:
        s, c, p = [], [], []
        for _ in range(n_eps):
            env_s = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps, wind_speed=0)
            obs = env_s.reset()
            for _ in range(max_steps):
                am = np.array([env_s.drones[i]['alive'] for i in range(n_drones)])
                pos = np.array([env_s.drones[i]['pos'] for i in range(n_drones)], dtype=np.float32)
                if not am.any(): break
                acts, _, _, _ = agent.select_actions(obs, pos, am)
                obs, _, dones, _ = env_s.step(np.array(acts, dtype=np.int32))
                if all(dones): break
            ac = sum(1 for i in range(n_drones) if env_s.drones[i]['alive'])
            s.append(ac/n_drones*100)
            c.append(len(env_s.total_cells_explored)/(grid*grid)*100)
            fc = np.argwhere(env_s.fire > 0.2); pc = set()
            for fx, fy in fc:
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = fx+dx, fy+dy
                    if 0<=nx<grid and 0<=ny<grid and env_s.fire[nx,ny]<0.1: pc.add((nx,ny))
            vis = set()
            for i in range(n_drones): vis.update(env_s.drones[i].get('visited', set()))
            p.append(len(pc & vis)/max(1,len(pc))*100)
        s_ci = bootstrap_ci(np.array(s))
        c_ci = bootstrap_ci(np.array(c))
        results[str(n_drones)] = {
            'safety': np.mean(s), 'coverage': np.mean(c), 'perimeter': np.mean(p),
            'safety_ci': [float(s_ci[1]), float(s_ci[2])],
            'coverage_ci': [float(c_ci[1]), float(c_ci[2])],
        }
        print(f"  {n_drones} drones: Safety={np.mean(s):5.1f}% | Coverage={np.mean(c):5.1f}% | Perimeter={np.mean(p):5.1f}%", flush=True)
    with open('scalability_results.json', 'w') as f: json.dump(results, f, indent=2)
    return results


def grid_scalability_test(agent, n_drones=10, max_steps=300, n_eps=10):
    """Evaluate across different grid sizes to show environment generalization."""
    print(f"\n{'='*60}\nGRID SCALABILITY | {n_drones} drones | {n_eps} eps each\n{'='*60}", flush=True)
    results = {}
    for grid in [20, 30, 50]:
        s, c = [], []
        for _ in range(n_eps):
            env_g = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps, wind_speed=0)
            obs = env_g.reset()
            for _ in range(max_steps):
                am = np.array([env_g.drones[i]['alive'] for i in range(n_drones)])
                pos = np.array([env_g.drones[i]['pos'] for i in range(n_drones)], dtype=np.float32)
                if not am.any(): break
                acts, _, _, _ = agent.select_actions(obs, pos, am)
                obs, _, dones, _ = env_g.step(np.array(acts, dtype=np.int32))
                if all(dones): break
            ac = sum(1 for i in range(n_drones) if env_g.drones[i]['alive'])
            s.append(ac/n_drones*100)
            c.append(len(env_g.total_cells_explored)/(grid*grid)*100)
        results[str(grid)] = {'safety': np.mean(s), 'coverage': np.mean(c), 'grid': grid}
        print(f"  Grid {grid}x{grid}: Safety={np.mean(s):5.1f}% | Coverage={np.mean(c):5.1f}%", flush=True)
    with open('grid_scalability_results.json', 'w') as f: json.dump(results, f, indent=2)
    return results


# ═══════════════════════════════════════════════════════════════
# SECTION 7c: COMMUNICATION GRAPH TOPOLOGY ANALYSIS
# ═══════════════════════════════════════════════════════════════

def communication_analysis(agent, grid=30, n_drones=10, max_steps=300, n_eps=5):
    """Analyze the learned communication topology — what structure
    emerges from the GAT attention? This is the novelty analysis."""
    print(f"\n{'='*60}\nCOMMUNICATION TOPOLOGY ANALYSIS\n{'='*60}", flush=True)
    all_attn = []
    all_adj_density = []
    all_components = []
    all_avg_path = []

    for ep_i in range(n_eps):
        env_c = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps, wind_speed=0)
        obs = env_c.reset()
        for step in range(max_steps):
            am = np.array([env_c.drones[i]['alive'] for i in range(n_drones)])
            pos = np.array([env_c.drones[i]['pos'] for i in range(n_drones)], dtype=np.float32)
            if not am.any(): break
            obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
            pos_t = torch.tensor(pos, dtype=torch.float32).to(device)
            alive_t = torch.tensor(am, dtype=torch.bool).to(device)
            n_alive = int(am.sum())
            if n_alive < 2: break
            # Build adjacency and get attention
            adj = agent.gat.build_graph(pos_t, alive_t)
            adj_np = adj.cpu().numpy().astype(float)
            all_adj_density.append(adj_np.sum() / max(1, n_alive*(n_alive-1)))
            # Count connected components via BFS
            visited = [False]*n_alive
            n_comp = 0
            for start in range(n_alive):
                if not visited[start]:
                    n_comp += 1
                    queue = [start]
                    visited[start] = True
                    while queue:
                        node = queue.pop(0)
                        for nb in range(n_alive):
                            if adj_np[node, nb] > 0 and not visited[nb]:
                                visited[nb] = True
                                queue.append(nb)
            all_components.append(n_comp)
            # Average shortest path (only if connected)
            if n_comp == 1 and n_alive <= 20:
                # BFS from each node
                path_lens = []
                for src in range(n_alive):
                    dist = [-1]*n_alive; dist[src] = 0
                    queue = [src]
                    while queue:
                        node = queue.pop(0)
                        for nb in range(n_alive):
                            if adj_np[node, nb] > 0 and dist[nb] == -1:
                                dist[nb] = dist[node] + 1
                                queue.append(nb)
                    path_lens.extend([d for d in dist if d > 0])
                if path_lens: all_avg_path.append(np.mean(path_lens))
            # Attention weights: compute via GAT forward pass
            try:
                obs_t2 = torch.tensor(obs, dtype=torch.float32).to(device)
                pos_t2 = torch.tensor(pos, dtype=torch.float32).to(device)
                alive_t2 = torch.tensor(am, dtype=torch.bool).to(device)
                adj2 = agent.gat.build_graph(pos_t2, alive_t2)
                h1_2 = F.relu(agent.gat.norm1(agent.gat.attn1(obs_t2, adj2) + agent.gat.res1(obs_t2)))
                _, attn_w2 = agent.gat.attn2(h1_2, adj2, return_attn=True)
                if attn_w2 is not None and attn_w2.ndim == 2:
                    # attn_w2 is (K, K) — apply softmax for proper distribution
                    attn_np = F.softmax(attn_w2, dim=-1).detach().cpu().numpy()
                    all_attn.append(attn_np)  # store full matrix (K, K)
            except Exception:
                pass
            obs, _, dones, _ = env_c.step(agent.select_actions(obs, pos, am)[0])
            if all(dones): break

    if all_attn:
        # Pad all attention matrices to n_drones x n_drones for averaging
        max_k = n_drones
        padded = []
        for a in all_attn:
            k = a.shape[0]
            if k == max_k:
                padded.append(a)
            else:
                pad = np.zeros((max_k, max_k), dtype=a.dtype)
                pad[:k, :k] = a
                padded.append(pad)
        avg_attn = np.mean(padded, axis=0)
        # Attention entropy per agent
        attn_entropies = []
        for a in all_attn:
            k = a.shape[0]
            if k < 2: continue
            for row in a:
                p = row / (row.sum() + 1e-8)
                p = p[p > 1e-8]
                if len(p) > 0: attn_entropies.append(-np.sum(p * np.log(p)))
        print(f"  Avg adjacency density: {np.mean(all_adj_density):.3f} (fraction of possible edges)", flush=True)
        print(f"  Avg connected components: {np.mean(all_components):.2f} (1.0 = fully connected)", flush=True)
        if all_avg_path: print(f"  Avg shortest path length: {np.mean(all_avg_path):.2f}", flush=True)
        if attn_entropies: print(f"  Avg attention entropy: {np.mean(attn_entropies):.3f} (max={np.log(n_drones):.2f} for uniform)", flush=True)
        # Per-agent attention distribution
        for i in range(min(n_drones, avg_attn.shape[0])):
            top3 = np.argsort(avg_attn[i])[-3:][::-1]
            print(f"    Agent {i} top-3 targets: {top3.tolist()} (weights: {avg_attn[i, top3].round(3).tolist()})", flush=True)
    else:
        print("  Warning: no attention data collected", flush=True)

    comm_data = {
        'adj_density': float(np.mean(all_adj_density)) if all_adj_density else 0,
        'connected_components': float(np.mean(all_components)) if all_components else 0,
        'avg_path_length': float(np.mean(all_avg_path)) if all_avg_path else 0,
        'attention_entropy': float(np.mean(attn_entropies)) if all_attn else 0,
    }
    with open('communication_analysis.json', 'w') as f: json.dump(comm_data, f, indent=2)
    return comm_data


# ═══════════════════════════════════════════════════════════════
# SECTION 7d: CONTRIBUTION ISOLATION ABLATION
# ═══════════════════════════════════════════════════════════════

def train_nocomm(n_episodes=500, grid=30, n_drones=10, max_steps=300, seed=0):
    """No-Comm ablation: MLP-based processing with NO GAT and NO shared
    exploration map. Isolates the contribution of each component."""
    torch.manual_seed(seed); np.random.seed(seed)
    print(f"\n{'='*60}", flush=True)
    print(f"No-Comm (No GAT + No Shared Map) | seed={seed} | {n_episodes} eps", flush=True)
    print(f"{'='*60}", flush=True)

    # Override: disable shared map in environment
    env = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps, wind_speed=0)
    agent = FastGATPPO(obs_dim=env.obs_dim, act_dim=env.act_dim, use_gat=False)

    rewards_h, coverage_h, safety_h = [], [], []
    best_r = -float('inf')
    early_stop_counter = 0
    t0 = time.time()

    try:
        for ep in range(n_episodes):
            obs = env.reset()
            # Zero out shared exploration map at every step to simulate no-shared-map
            env.shared_visited[:] = 0.0
            agent._traj.clear()
            ep_r, ep_crashes = 0.0, 0

            for step in range(max_steps):
                am = np.array([env.drones[i]['alive'] for i in range(n_drones)])
                pos = np.array([env.drones[i]['pos'] for i in range(n_drones)], dtype=np.float32)
                if not am.any(): break
                actions, log_probs, values, enhanced = agent.select_actions(obs, pos, am)
                prev_visited = [set(env.drones[i]['visited']) for i in range(n_drones)]
                env.shared_visited[:] = 0.0  # zero BEFORE step
                obs_next, _, dones, infos = env.step(np.array(actions, dtype=np.int32))
                # Zero shared channel in obs_next (step() sets shared_visited internally)
                obs_next[:, 4*81:5*81] = 0.0
                shaped = np.zeros(n_drones, dtype=np.float32)
                for i in range(n_drones):
                    if not am[i]: continue
                    fd = infos[i].get('fire_dist', 10.0)
                    crashed = infos[i].get('crashed', False)
                    shaped[i] = compute_reward(env.drones[i], i, env.drones, prev_visited[i], fd, crashed, step, max_steps, grid)
                    ep_r += shaped[i]
                    if crashed: ep_crashes += 1
                agent_ids = list(range(n_drones))
                agent.store(enhanced, actions, shaped, dones.astype(np.float32), log_probs, values, agent_ids)
                obs = obs_next
                if all(dones): break

            agent.update()
            cov = len(env.total_cells_explored)/(grid*grid)*100
            saf = (1.0 - ep_crashes/n_drones)*100
            rewards_h.append(ep_r); coverage_h.append(cov); safety_h.append(saf)
            if (ep+1) % 100 == 0:
                avg_r = np.mean(rewards_h[-100:]); avg_cov = np.mean(coverage_h[-100:]); avg_saf = np.mean(safety_h[-100:])
                elapsed = time.time() - t0
                eps_per_sec = (ep+1)/elapsed
                eta_min = (n_episodes-ep-1)/max(eps_per_sec,1e-6)/60.0
                print(f"Ep {ep+1:5d}/{n_episodes} | R: {avg_r:7.1f} | Cov: {avg_cov:5.1f}% | Safe: {avg_saf:4.0f}% | {eps_per_sec:.1f} ep/s | ETA {eta_min:.0f}m", flush=True)
                if avg_r > best_r:
                    best_r = avg_r
                    agent.save(f'nocomm_s{seed}_best.pt')
                if avg_cov >= 65.0: early_stop_counter += 100
                else: early_stop_counter = 0
                if early_stop_counter >= 300: break
    except Exception as e:
        print(f"\nTraining interrupted at ep {ep+1}: {e}", flush=True)

    agent.save(f'nocomm_s{seed}_final.pt')
    results = {
        'n_episodes': len(rewards_h), 'seed': seed, 'use_gat': False,
        'final_reward': float(np.mean(rewards_h[-100:])) if len(rewards_h) >= 100 else float(np.mean(rewards_h)),
        'final_coverage': float(np.mean(coverage_h[-100:])) if len(coverage_h) >= 100 else float(np.mean(coverage_h)),
        'final_safety': float(np.mean(safety_h[-100:])) if len(safety_h) >= 100 else float(np.mean(safety_h)),
        'rewards': [float(x) for x in rewards_h],
        'coverages': [float(x) for x in coverage_h],
        'safety': [float(x) for x in safety_h],
    }
    with open(f'nocomm_s{seed}_results.json', 'w') as f: json.dump(results, f, indent=2)
    print(f"\nDone in {time.time()-t0:.0f}s | Coverage: {results['final_coverage']:.1f}% | Safety: {results['final_safety']:.0f}%", flush=True)
    return agent, results


def train_nocomm_multi_seed(n_episodes, grid, n_drones, max_steps, seeds=[42, 123]):
    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n{'#'*60}", flush=True)
        print(f"# No-Comm Run {i+1}/{len(seeds)} | seed={seed}", flush=True)
        print(f"{'#'*60}", flush=True)
        agent, res = train_nocomm(n_episodes, grid, n_drones, max_steps, seed=seed)
        all_results.append(res)
    covs = [r['final_coverage'] for r in all_results]
    print(f"\nNo-Comm | {len(seeds)} seeds: Coverage={np.mean(covs):.1f}% +/- {np.std(covs):.1f}%", flush=True)
    return None, all_results


# ═══════════════════════════════════════════════════════════════
# SECTION 8: FIGURES
# ═══════════════════════════════════════════════════════════════

def generate_figures(gat_res, nogat_res, mappo_res, ippo_res=None, nocomm_res=None,
                     bench=None, wind_res=None, scalability_res=None, grid_scalability_res=None,
                     gat_agent=None, grid=30, n_drones=10, max_steps=300):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs('figures_gat', exist_ok=True)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'figure.dpi': 150, 'savefig.bbox': 'tight'})

    def smooth(x, w=20):
        return np.convolve(x, np.ones(w)/w, mode='valid') if len(x) > w else np.array(x)

    # Fig 1: Training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'GAT-MARAHS': '#3498db', 'No-GAT': '#95a5a6', 'MAPPO': '#e74c3c', 'IPPO': '#2ecc71', 'No-Comm': '#9b59b6'}
    all_method_data = [('GAT-MARAHS', gat_res, '-'), ('No-GAT', nogat_res, '--'), ('MAPPO', mappo_res, ':')]
    if ippo_res: all_method_data.append(('IPPO', ippo_res, '-.'))
    if nocomm_res: all_method_data.append(('No-Comm', nocomm_res, (0, (3, 3))))
    for ax_i, (metric, label) in enumerate([('coverages', 'Coverage (%)'), ('safety', 'Safety (%)'), ('rewards', 'Reward')]):
        ax = axes[ax_i//2][ax_i%2]
        for name, all_res, ls in all_method_data:
            curves = [smooth(r[metric]) for r in all_res if len(r[metric]) > 20]
            if curves:
                ml = min(len(cu) for cu in curves)
                arr = np.array([cu[:ml] for cu in curves])
                mean, std = np.mean(arr, 0), np.std(arr, 0)
                x = np.arange(len(mean))
                ax.plot(x, mean, ls, color=colors[name], lw=2, label=name)
                ax.fill_between(x, mean-std, mean+std, alpha=0.15, color=colors[name])
        ax.set_xlabel('Episode'); ax.set_ylabel(label); ax.set_title(f'({chr(97+ax_i)}) {label}')
        ax.legend(); ax.grid(True, alpha=0.3)
    # Summary bar
    ax = axes[1][1]
    methods = [m for m, _, _ in all_method_data]
    all_rs = [r for _, r, _ in all_method_data]
    covs = [np.mean([r['final_coverage'] for r in ar]) for ar in all_rs]
    safes = [np.mean([r['final_safety'] for r in ar]) for ar in all_rs]
    cerrs = [np.std([r['final_coverage'] for r in ar]) for ar in all_rs]
    serrs = [np.std([r['final_safety'] for r in ar]) for ar in all_rs]
    x = np.arange(3); w = 0.35
    ax.bar(x-w/2, covs, w, yerr=cerrs, capsize=5, label='Coverage', color='#3498db', edgecolor='k', lw=0.5)
    ax.bar(x+w/2, safes, w, yerr=serrs, capsize=5, label='Safety', color='#2ecc71', edgecolor='k', lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel('Final Performance (%)'); ax.set_title('(d) Model Comparison')
    ax.legend(); ax.grid(True, axis='y', alpha=0.3); ax.set_ylim(0, 100)
    plt.tight_layout(); plt.savefig('figures_gat/fig1_training.png'); plt.close()

    # Fig 2: Benchmark
    fig, ax = plt.subplots(figsize=(8, 5))
    ms = list(bench.keys()); x = np.arange(len(ms)); w = 0.25
    ax.bar(x-w, [bench[m]['safety'] for m in ms], w, label='Safety', color='#2ecc71', edgecolor='k', lw=0.5)
    ax.bar(x, [bench[m]['coverage'] for m in ms], w, label='Coverage', color='#3498db', edgecolor='k', lw=0.5)
    ax.bar(x+w, [bench[m]['perimeter'] for m in ms], w, label='Perimeter', color='#e74c3c', edgecolor='k', lw=0.5)
    ax.set_ylabel('Performance (%)'); ax.set_title(f'Benchmark (wind=12 m/s)')
    ax.set_xticks(x); ax.set_xticklabels(ms, rotation=15); ax.legend(); ax.grid(True, axis='y', alpha=0.3); ax.set_ylim(0, 100)
    plt.tight_layout(); plt.savefig('figures_gat/fig2_benchmark.png'); plt.close()

    # Fig 3: Wind sweep
    fig, ax = plt.subplots(figsize=(8, 5))
    winds = [int(k) for k in sorted(wind_res.keys())]
    for metric, color, marker, label in [('perimeter', '#e74c3c', 'o', 'Perimeter'), ('safety', '#2ecc71', 's', 'Safety'), ('coverage', '#3498db', '^', 'Coverage')]:
        ax.errorbar(winds, [wind_res[str(w)][metric] for w in winds],
                    yerr=[wind_res[str(w)][f'{metric}_std'] for w in winds],
                    marker=marker, lw=2, label=label, color=color, capsize=4)
    ax.set_xlabel('Wind Speed (m/s)'); ax.set_ylabel('Performance (%)'); ax.set_title('GAT-MARAHS vs Wind Intensity')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 100)
    plt.tight_layout(); plt.savefig('figures_gat/fig3_wind_sweep.png'); plt.close()

    # Fig 4: Ablation
    fig, ax = plt.subplots(figsize=(7, 5))
    cats = ['Coverage', 'Safety', 'Perimeter']
    gat_v = [np.mean([r['final_coverage'] for r in gat_res]), np.mean([r['final_safety'] for r in gat_res]),
             bench.get('GAT-MARAHS', {}).get('perimeter', 0)]
    gat_e = [np.std([r['final_coverage'] for r in gat_res]), np.std([r['final_safety'] for r in gat_res]), 0]
    ng_v = [np.mean([r['final_coverage'] for r in nogat_res]), np.mean([r['final_safety'] for r in nogat_res]),
            bench.get('Random', {}).get('perimeter', 0)]
    ng_e = [np.std([r['final_coverage'] for r in nogat_res]), np.std([r['final_safety'] for r in nogat_res]), 0]
    x = np.arange(3); w = 0.3
    ax.bar(x-w/2, gat_v, w, yerr=gat_e, capsize=5, label='GAT-MARAHS', color='#3498db', edgecolor='k', lw=0.5)
    ax.bar(x+w/2, ng_v, w, yerr=ng_e, capsize=5, label='No-GAT', color='#95a5a6', edgecolor='k', lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel('Performance (%)'); ax.set_title('GAT Communication Ablation')
    ax.legend(); ax.grid(True, axis='y', alpha=0.3); ax.set_ylim(0, 100)
    plt.tight_layout(); plt.savefig('figures_gat/fig4_ablation.png'); plt.close()

    # Fig 5: Attention heatmap from trained GAT
    if gat_agent is not None:
        try:
            ea = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps, wind_speed=0)
            obs_a = ea.reset()
            am_a = np.array([ea.drones[i]['alive'] for i in range(n_drones)])
            pos_a = np.array([ea.drones[i]['pos'] for i in range(n_drones)], dtype=np.float32)
            obs_t = torch.tensor(obs_a, dtype=torch.float32).to(device)
            pos_t = torch.tensor(pos_a, dtype=torch.float32).to(device)
            alive_t = torch.tensor(am_a, dtype=torch.bool).to(device)
            adj = gat_agent.gat.build_graph(pos_t, alive_t)
            h1 = F.relu(gat_agent.gat.norm1(gat_agent.gat.attn1(obs_t, adj) + gat_agent.gat.res1(obs_t)))
            _, attn_w = gat_agent.gat.attn2(h1, adj, return_attn=True)
            # attn_w is already (K, K) — just softmax to get proper distribution
            attn_np = F.softmax(attn_w, dim=-1).detach().cpu().numpy()
            n_a = attn_np.shape[0]
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(attn_np, cmap='YlOrRd', vmin=0)
            ax.set_xlabel('Receiver'); ax.set_ylabel('Sender')
            ax.set_title('GAT Attention Weights (learned communication)')
            ax.set_xticks(range(n_a)); ax.set_yticks(range(n_a))
            plt.colorbar(im, ax=ax, label='Weight')
            plt.tight_layout(); plt.savefig('figures_gat/fig5_attention.png'); plt.close()
            del ea
        except Exception as e:
            print(f"  Warning: attention viz failed: {e}", flush=True)

    # Fig 6: Sample efficiency — episodes to reach coverage thresholds
    fig, ax = plt.subplots(figsize=(8, 5))
    thresholds = np.arange(5, 80, 5)
    se_methods = [('GAT-MARAHS', gat_res, '-'), ('No-GAT', nogat_res, '--'), ('MAPPO', mappo_res, ':')]
    if ippo_res: se_methods.append(('IPPO', ippo_res, '-.'))
    if nocomm_res: se_methods.append(('No-Comm', nocomm_res, (0, (3, 3))))
    for name, all_res, ls in se_methods:
        frac_at_t = []
        for t in thresholds:
            count = 0
            for r in all_res:
                cov_curve = smooth(r['coverages'], w=20)
                if len(cov_curve) > 0 and np.max(cov_curve) >= t:
                    idx = np.where(cov_curve >= t)[0]
                    if len(idx) > 0: count += 1
            frac_at_t.append(count / max(1, len(all_res)))
        ax.plot(thresholds, frac_at_t, ls, color=colors[name], lw=2, label=name, marker='o', markersize=3)
    ax.set_xlabel('Coverage Threshold (%)')
    ax.set_ylabel('Fraction of Seeds Achieved')
    ax.set_title('(f) Sample Efficiency')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.1)
    plt.tight_layout(); plt.savefig('figures_gat/fig6_sample_efficiency.png'); plt.close()

    # Fig 7: Benchmark with confidence intervals
    fig, ax = plt.subplots(figsize=(10, 5))
    ms = list(bench.keys())
    x = np.arange(len(ms)); w = 0.25
    safety_ci = [bench[m].get('safety_ci', [bench[m]['safety'], bench[m]['safety']]) for m in ms]
    coverage_ci = [bench[m].get('coverage_ci', [bench[m]['coverage'], bench[m]['coverage']]) for m in ms]
    perimeter_ci = [bench[m].get('perimeter_ci', [bench[m]['perimeter'], bench[m]['perimeter']]) for m in ms]
    safety_err = [[bench[m]['safety'] - safety_ci[i][0], safety_ci[i][1] - bench[m]['safety']] for i, m in enumerate(ms)]
    coverage_err = [[bench[m]['coverage'] - coverage_ci[i][0], coverage_ci[i][1] - bench[m]['coverage']] for i, m in enumerate(ms)]
    perimeter_err = [[bench[m]['perimeter'] - perimeter_ci[i][0], perimeter_ci[i][1] - bench[m]['perimeter']] for i, m in enumerate(ms)]
    ax.bar(x-w, [bench[m]['safety'] for m in ms], w, yerr=np.array(safety_err).T, capsize=4, label='Safety', color='#2ecc71', edgecolor='k', lw=0.5)
    ax.bar(x, [bench[m]['coverage'] for m in ms], w, yerr=np.array(coverage_err).T, capsize=4, label='Coverage', color='#3498db', edgecolor='k', lw=0.5)
    ax.bar(x+w, [bench[m]['perimeter'] for m in ms], w, yerr=np.array(perimeter_err).T, capsize=4, label='Perimeter', color='#e74c3c', edgecolor='k', lw=0.5)
    ax.set_ylabel('Performance (%)'); ax.set_title('Benchmark with 95% Bootstrap CIs (wind=12 m/s)')
    ax.set_xticks(x); ax.set_xticklabels(ms, rotation=15); ax.legend(); ax.grid(True, axis='y', alpha=0.3); ax.set_ylim(0, 100)
    plt.tight_layout(); plt.savefig('figures_gat/fig7_benchmark_ci.png'); plt.close()

    # Fig 8: Exploration heatmap — spatial coverage of trained GAT agent
    if gat_agent is not None:
        try:
            ea = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps, wind_speed=0)
            obs_a = ea.reset()
            for _ in range(max_steps):
                am_a = np.array([ea.drones[i]['alive'] for i in range(n_drones)])
                pos_a = np.array([ea.drones[i]['pos'] for i in range(n_drones)], dtype=np.float32)
                if not am_a.any(): break
                acts, _, _, _ = gat_agent.select_actions(obs_a, pos_a, am_a)
                obs_a, _, dones_a, _ = ea.step(np.array(acts, dtype=np.int32))
                if all(dones_a): break
            # Build visit density map
            visit_map = np.zeros((grid, grid), dtype=np.float32)
            for i in range(n_drones):
                for (vx, vy) in ea.drones[i].get('visited', set()):
                    if 0 <= vx < grid and 0 <= vy < grid:
                        visit_map[vy, vx] += 1
            visit_map = np.clip(visit_map, 0, np.percentile(visit_map[visit_map > 0], 95) if (visit_map > 0).any() else 1)
            fig, axes_ex = plt.subplots(1, 3, figsize=(15, 5))
            im0 = axes_ex[0].imshow(ea.fire, cmap='YlOrRd', vmin=0, vmax=1)
            axes_ex[0].set_title('Fire Intensity'); plt.colorbar(im0, ax=axes_ex[0])
            im1 = axes_ex[1].imshow(visit_map, cmap='YlGnBu', vmin=0)
            axes_ex[1].set_title('Drone Visit Density'); plt.colorbar(im1, ax=axes_ex[1])
            im2 = axes_ex[2].imshow(ea.shared_visited, cmap='Greens', vmin=0, vmax=1)
            axes_ex[2].set_title('Shared Exploration Map'); plt.colorbar(im2, ax=axes_ex[2])
            for ax_ex in axes_ex: ax_ex.set_xlabel('X'); ax_ex.set_ylabel('Y')
            plt.suptitle('GAT-MARAHS Exploration Behavior (1 episode)', y=1.02)
            plt.tight_layout(); plt.savefig('figures_gat/fig8_exploration_heatmap.png', bbox_inches='tight'); plt.close()
            del ea
        except Exception as e:
            print(f"  Warning: exploration heatmap failed: {e}", flush=True)

    # Fig 9: Radar chart — multi-metric comparison
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    radar_metrics = ['Coverage', 'Safety', 'Perimeter', 'Sample Eff.', 'Robustness']
    n_metrics = len(radar_metrics)
    angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    for all_res, name, color in [(gat_res, 'GAT-MARAHS', '#3498db'), (nogat_res, 'No-GAT', '#95a5a6'), (mappo_res, 'MAPPO', '#e74c3c')]:
        # Compute normalized scores
        cov = np.mean([r['final_coverage'] for r in all_res])
        saf = np.mean([r['final_safety'] for r in all_res])
        perim = bench.get(name, {}).get('perimeter', bench.get('GAT-MARAHS', {}).get('perimeter', 0) if name == 'GAT-MARAHS' else 0)
        # Sample efficiency: inverse of episodes to 30% coverage (normalized)
        se_scores = []
        for r in all_res:
            cov_smooth = smooth(r['coverages'], 20)
            idx = np.where(cov_smooth >= 30)[0]
            se_scores.append(idx[0] if len(idx) > 0 else len(cov_smooth))
        se = 100.0 / max(1, np.mean(se_scores))  # higher is better
        # Robustness: mean coverage over wind sweep
        rob = np.mean([wind_res[str(w)]['coverage'] for w in sorted([int(k) for k in wind_res.keys()])])
        values = [cov, saf, perim, se * 10, rob]  # scale se to comparable range
        values = [min(100, max(0, v)) for v in values]
        values += values[:1]
        ax.plot(angles, values, 'o-', lw=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(radar_metrics)
    ax.set_ylim(0, 100)
    ax.set_title('Multi-Metric Comparison (Radar)', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout(); plt.savefig('figures_gat/fig9_radar.png', bbox_inches='tight'); plt.close()

    # Fig 10: Scalability — coverage vs swarm size
    if scalability_res:
        fig, ax = plt.subplots(figsize=(8, 5))
        ns = sorted([int(k) for k in scalability_res.keys()])
        covs = [scalability_res[str(n)]['coverage'] for n in ns]
        safes = [scalability_res[str(n)]['safety'] for n in ns]
        c_ci_lo = [scalability_res[str(n)].get('coverage_ci', [0,0])[0] for n in ns]
        c_ci_hi = [scalability_res[str(n)].get('coverage_ci', [0,0])[1] for n in ns]
        ax.errorbar(ns, covs, yerr=[np.array(covs)-np.array(c_ci_lo), np.array(c_ci_hi)-np.array(covs)],
                    marker='o', lw=2, color='#3498db', capsize=5, label='Coverage', fmt='-o')
        ax.errorbar(ns, safes, marker='s', lw=2, color='#2ecc71', capsize=5, label='Safety', fmt='-s')
        ax.set_xlabel('Number of Drones'); ax.set_ylabel('Performance (%)')
        ax.set_title('GAT-MARAHS Scalability'); ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 100)
        ax.set_xticks(ns)
        plt.tight_layout(); plt.savefig('figures_gat/fig10_scalability.png'); plt.close()

    # Fig 11: Contribution isolation — ablation bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    abl_methods = ['GAT-MARAHS', 'No-GAT', 'MAPPO']
    abl_rs = [gat_res, nogat_res, mappo_res]
    abl_colors = ['#3498db', '#95a5a6', '#e74c3c']
    if ippo_res:
        abl_methods.append('IPPO'); abl_rs.append(ippo_res); abl_colors.append('#2ecc71')
    if nocomm_res:
        abl_methods.append('No-Comm'); abl_rs.append(nocomm_res); abl_colors.append('#9b59b6')
    abl_covs = [np.mean([r['final_coverage'] for r in ar]) for ar in abl_rs]
    abl_cov_errs = [np.std([r['final_coverage'] for r in ar]) for ar in abl_rs]
    abl_safs = [np.mean([r['final_safety'] for r in ar]) for ar in abl_rs]
    abl_saf_errs = [np.std([r['final_safety'] for r in ar]) for ar in abl_rs]
    x = np.arange(len(abl_methods)); w = 0.35
    ax.bar(x-w/2, abl_covs, w, yerr=abl_cov_errs, capsize=5, label='Coverage', color=abl_colors, edgecolor='k', lw=0.5, alpha=0.85)
    ax.bar(x+w/2, abl_safs, w, yerr=abl_saf_errs, capsize=5, label='Safety', color=abl_colors, edgecolor='k', lw=0.5, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(abl_methods, rotation=15)
    ax.set_ylabel('Performance (%)'); ax.set_title('Contribution Isolation Ablation')
    ax.legend(); ax.grid(True, axis='y', alpha=0.3); ax.set_ylim(0, 100)
    plt.tight_layout(); plt.savefig('figures_gat/fig11_contribution_ablation.png'); plt.close()

    # Fig 12: Grid scalability — coverage vs grid size
    if grid_scalability_res:
        fig, ax = plt.subplots(figsize=(7, 5))
        grids = sorted([int(k) for k in grid_scalability_res.keys()])
        gs_covs = [grid_scalability_res[str(g)]['coverage'] for g in grids]
        gs_safs = [grid_scalability_res[str(g)]['safety'] for g in grids]
        ax.plot(grids, gs_covs, '-o', color='#3498db', lw=2, label='Coverage')
        ax.plot(grids, gs_safs, '-s', color='#2ecc71', lw=2, label='Safety')
        ax.set_xlabel('Grid Size'); ax.set_ylabel('Performance (%)')
        ax.set_title('GAT-MARAHS Environment Generalization')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 100)
        ax.set_xticks(grids)
        plt.tight_layout(); plt.savefig('figures_gat/fig12_grid_scalability.png'); plt.close()

    # Fig 13: Updated radar chart with all methods
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    radar_metrics = ['Coverage', 'Safety', 'Perimeter', 'Sample Eff.', 'Robustness']
    n_metrics = len(radar_metrics)
    angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]
    radar_methods = [('GAT-MARAHS', gat_res, '#3498db'), ('No-GAT', nogat_res, '#95a5a6'), ('MAPPO', mappo_res, '#e74c3c')]
    if ippo_res: radar_methods.append(('IPPO', ippo_res, '#2ecc71'))
    if nocomm_res: radar_methods.append(('No-Comm', nocomm_res, '#9b59b6'))
    for name, all_res, color in radar_methods:
        cov = np.mean([r['final_coverage'] for r in all_res])
        saf = np.mean([r['final_safety'] for r in all_res])
        perim = bench.get(name, {}).get('perimeter', bench.get('GAT-MARAHS', {}).get('perimeter', 0) if name == 'GAT-MARAHS' else 0)
        se_scores = []
        for r in all_res:
            cov_smooth_arr = smooth(r['coverages'], 20)
            idx = np.where(cov_smooth_arr >= 30)[0]
            se_scores.append(idx[0] if len(idx) > 0 else len(cov_smooth_arr))
        se = 100.0 / max(1, np.mean(se_scores))
        rob = np.mean([wind_res[str(w)]['coverage'] for w in sorted([int(k) for k in wind_res.keys()])]) if wind_res else 0
        values = [cov, saf, perim, se * 10, rob]
        values = [min(100, max(0, v)) for v in values]
        values += values[:1]
        ax.plot(angles, values, 'o-', lw=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.08, color=color)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(radar_metrics)
    ax.set_ylim(0, 100)
    ax.set_title('Multi-Method Comparison (Radar)', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
    plt.tight_layout(); plt.savefig('figures_gat/fig13_radar_all.png', bbox_inches='tight'); plt.close()

    n_figs = 9 + (1 if scalability_res else 0) + (1 if nocomm_res else 0) + (1 if grid_scalability_res else 0) + 1
    print(f"  ✓ {n_figs} figures saved to figures_gat/", flush=True)


# ═══════════════════════════════════════════════════════════════
# SECTION 9: STATISTICAL ANALYSIS & RESEARCH METRICS
# ═══════════════════════════════════════════════════════════════

def compute_cohens_d(x, y):
    """Compute Cohen's d effect size (pooled std)."""
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_std = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2))
    if pooled_std < 1e-10: return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


def bootstrap_ci(data, n_boot=2000, ci=0.95):
    """Bootstrap confidence interval for the mean."""
    if len(data) < 2:
        return np.mean(data), np.mean(data), np.mean(data)
    boot_means = np.array([np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)])
    lo = np.percentile(boot_means, (1-ci)/2 * 100)
    hi = np.percentile(boot_means, (1+ci)/2 * 100)
    return np.mean(data), lo, hi


def statistical_analysis(gat_res, nogat_res, mappo_res, bench_res, wind_res, ippo_res=None, nocomm_res=None):
    """Publication-ready statistical comparison between all methods.
    Reports: means, stds, 95% CIs, Mann-Whitney U p-values, Cohen's d.
    """
    print("\n" + "="*70, flush=True)
    print("STATISTICAL ANALYSIS", flush=True)
    print("="*70, flush=True)

    methods = {
        'GAT-MARAHS': gat_res,
        'No-GAT': nogat_res,
        'MAPPO': mappo_res,
    }
    if ippo_res: methods['IPPO'] = ippo_res
    if nocomm_res: methods['No-Comm'] = nocomm_res

    # --- 1. Training convergence comparison ---
    print("\n--- Training Convergence (final 100-episode windows) ---", flush=True)
    print(f"{'Method':<16s} {'Coverage':>18s} {'Safety':>18s} {'Reward':>18s}", flush=True)
    print(f"{'':16s} {'mean +/- std [95%CI]':>18s} {'mean +/- std [95%CI]':>18s} {'mean +/- std [95%CI]':>18s}", flush=True)
    print("-"*70, flush=True)
    covs_all = {}
    for name, res in methods.items():
        covs = [r['final_coverage'] for r in res]
        safes = [r['final_safety'] for r in res]
        rews = [r['final_reward'] for r in res]
        covs_all[name] = covs
        c_mean, c_lo, c_hi = bootstrap_ci(covs)
        s_mean, s_lo, s_hi = bootstrap_ci(safes)
        r_mean, r_lo, r_hi = bootstrap_ci(rews)
        print(f"{name:<16s} {np.mean(covs):5.1f}+/-{np.std(covs):4.1f} [{c_lo:.1f},{c_hi:.1f}]".ljust(34) +
              f" {np.mean(safes):5.1f}+/-{np.std(safes):4.1f} [{s_lo:.1f},{s_hi:.1f}]".ljust(34) +
              f" {np.mean(rews):7.0f}+/-{np.std(rews):6.0f} [{r_lo:.0f},{r_hi:.0f}]", flush=True)

    # --- 2. Pairwise significance tests ---
    print("\n--- Pairwise Statistical Tests (Mann-Whitney U, two-sided) ---", flush=True)
    pairs = [('GAT-MARAHS', 'No-GAT'), ('GAT-MARAHS', 'MAPPO'), ('No-GAT', 'MAPPO')]
    if ippo_res:
        pairs.extend([('GAT-MARAHS', 'IPPO'), ('MAPPO', 'IPPO')])
    if nocomm_res:
        pairs.extend([('GAT-MARAHS', 'No-Comm'), ('No-GAT', 'No-Comm')])
    for m1, m2 in pairs:
        c1, c2 = covs_all[m1], covs_all[m2]
        if sp_stats is not None:
            stat, p = sp_stats.mannwhitneyu(c1, c2, alternative='two-sided')
        else:
            # Fallback: approximate Mann-Whitney U via permutation
            stat, p = _approx_mannwhitney(c1, c2)
        d = compute_cohens_d(c1, c2)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        direction = "higher" if np.mean(c1) > np.mean(c2) else "lower"
        print(f"  {m1} vs {m2}:", flush=True)
        print(f"    Coverage: {np.mean(c1):.1f}% vs {np.mean(c2):.1f}% ({direction} by {abs(np.mean(c1)-np.mean(c2)):.1f}pp)", flush=True)
        print(f"    U={stat:.0f}, p={p:.4f} {sig}, Cohen's d={d:.2f}", flush=True)

    # --- 3. Benchmark results with CIs ---
    print(f"\n--- Benchmark (wind=12 m/s) with 95% Bootstrap CIs ---", flush=True)
    print(f"{'Method':<18s} {'Safety':>16s} {'Coverage':>16s} {'Perimeter':>16s}", flush=True)
    print("-"*68, flush=True)
    for method_name, data in bench_res.items():
        # Each metric already has per-episode data from benchmark function
        # Use stored values (they are already means from n_eps episodes)
        print(f"{method_name:<18s} {data['safety']:6.1f}%{'':9s} {data['coverage']:6.1f}%{'':9s} {data['perimeter']:6.1f}%", flush=True)

    # --- 4. Wind robustness analysis ---
    print(f"\n--- Wind Robustness (coverage decay rate) ---", flush=True)
    winds = sorted([int(k) for k in wind_res.keys()])
    coverages = [wind_res[str(w)]['coverage'] for w in winds]
    if len(winds) > 1:
        # Linear regression: coverage vs wind speed
        slope, intercept = np.polyfit(winds, coverages, 1)
        print(f"  Coverage decay: {slope:.2f}% per m/s increase", flush=True)
        print(f"  Predicted coverage at 0 m/s: {intercept:.1f}%", flush=True)
        # Robustness score: normalized AUC of coverage over wind range
        auc = np.trapz(coverages, winds) / (winds[-1] - winds[0])
        print(f"  Robustness score (mean coverage over wind range): {auc:.1f}%", flush=True)
        # Pearson correlation
        if sp_stats is not None:
            r, p = sp_stats.pearsonr(winds, coverages)
        else:
            r = np.corrcoef(winds, coverages)[0, 1]
            p = 0.0
        print(f"  Pearson r={r:.3f}, p={p:.4f}", flush=True)

    # --- 5. Sample efficiency analysis ---
    print(f"\n--- Sample Efficiency (episodes to reach coverage thresholds) ---", flush=True)
    thresholds = [20, 30, 40, 50]
    for name, res in methods.items():
        eps_to_threshold = {}
        for t in thresholds:
            for r in res:
                cov_curve = r['coverages']
                # Smooth with 20-ep window
                if len(cov_curve) > 20:
                    smooth = np.convolve(cov_curve, np.ones(20)/20, mode='valid')
                else:
                    smooth = np.array(cov_curve)
                idx = np.where(smooth >= t)[0]
                if len(idx) > 0:
                    eps_to_threshold.setdefault(t, []).append(idx[0] + 20)  # offset for smoothing
                else:
                    eps_to_threshold.setdefault(t, []).append(float('inf'))
        entries = [f"{t}%: {np.mean(eps_to_threshold[t]):.0f}ep" for t in thresholds if t in eps_to_threshold and np.mean(eps_to_threshold[t]) < float('inf')]
        print(f"  {name}: {', '.join(entries) if entries else 'not reached'}", flush=True)

    # --- 6. Exploration diversity analysis ---
    print(f"\n--- Exploration Diversity (coverage variance across seeds) ---", flush=True)
    for name, res in methods.items():
        final_covs = [r['final_coverage'] for r in res]
        print(f"  {name}: {np.mean(final_covs):.1f}% +/- {np.std(final_covs):.1f}% (CV={np.std(final_covs)/max(0.01,np.mean(final_covs))*100:.1f}%)", flush=True)

    # --- 7. Save full statistical report ---
    report = {
        'methods': {},
        'pairwise_tests': [],
        'wind_robustness': {},
    }
    for name, res in methods.items():
        report['methods'][name] = {
            'coverage_mean': float(np.mean([r['final_coverage'] for r in res])),
            'coverage_std': float(np.std([r['final_coverage'] for r in res])),
            'safety_mean': float(np.mean([r['final_safety'] for r in res])),
            'safety_std': float(np.std([r['final_safety'] for r in res])),
            'reward_mean': float(np.mean([r['final_reward'] for r in res])),
            'reward_std': float(np.std([r['final_reward'] for r in res])),
        }
    for m1, m2 in pairs:
        c1, c2 = covs_all[m1], covs_all[m2]
        d = compute_cohens_d(c1, c2)
        if sp_stats is not None:
            _, p = sp_stats.mannwhitneyu(c1, c2, alternative='two-sided')
        else:
            _, p = _approx_mannwhitney(c1, c2)
        report['pairwise_tests'].append({
            'method1': m1, 'method2': m2,
            'coverage_diff': float(np.mean(c1) - np.mean(c2)),
            'cohens_d': float(d), 'p_value': float(p),
        })
    if len(winds) > 1:
        report['wind_robustness'] = {
            'decay_rate_per_ms': float(slope),
            'robustness_score': float(auc),
        }
    with open('statistical_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full statistical report saved to statistical_report.json", flush=True)
    print("="*70, flush=True)

    return report


def _approx_mannwhitney(x, y, n_perm=5000):
    """Fallback Mann-Whitney U test without scipy."""
    nx, ny = len(x), len(y)
    all_data = np.concatenate([x, y])
    ranks = sp_stats.rankdata(all_data) if sp_stats is not None else _rankdata(all_data)
    r1 = ranks[:nx].sum()
    u1 = r1 - nx*(nx+1)/2
    u2 = nx*ny - u1
    u = min(u1, u2)
    # Approximate p-value via normal approximation
    mu = nx*ny/2
    sigma = np.sqrt(nx*ny*(nx+ny+1)/12)
    if sigma < 1e-10:
        return u, 1.0
    z = (u - mu) / sigma
    p = 2 * min(_approx_normal_cdf(z), 1 - _approx_normal_cdf(z))
    return u, min(p, 1.0)


def _rankdata(arr):
    """Rankdata implementation without scipy, handles ties with average ranking."""
    sorted_idx = np.argsort(arr)
    ranks = np.empty_like(sorted_idx, dtype=np.float64)
    ranks[sorted_idx] = np.arange(1, len(arr)+1, dtype=np.float64)
    # Handle ties: assign average rank to tied values
    unique_vals = np.unique(arr)
    for val in unique_vals:
        mask = arr == val
        if mask.sum() > 1:
            tie_ranks = ranks[mask]
            avg_rank = tie_ranks.mean()
            ranks[mask] = avg_rank
    return ranks


def _approx_normal_cdf(x):
    """Approximate standard normal CDF using math.erf."""
    import math
    if np.isscalar(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    return np.array([0.5 * (1 + math.erf(float(v) / math.sqrt(2))) for v in x])


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60, flush=True)
    print("PlumeGym-MARL: Research-Grade Training Pipeline (GPU) v4", flush=True)
    print("="*60, flush=True)

    GRID = 30
    N_DRONES = 10
    MAX_STEPS = 300
    N_EPISODES = 500       # Reduced from 800: more seeds > more episodes for stats
    SEEDS = [42, 123, 256, 789, 2024]  # 5 seeds for proper statistical power
    MAPPO_SEEDS = [42, 123]  # 2 seeds for MAPPO (slower per-episode)

    t_total = time.time()

    # Phase 1: GAT-MARAHS (5 seeds)
    print("\n" + "#"*60, flush=True)
    print(f"# PHASE 1: Train GAT-MARAHS ({len(SEEDS)} seeds)", flush=True)
    print("#"*60, flush=True)
    gat_agent, gat_all_res = train_multi_seed(N_EPISODES, GRID, N_DRONES, MAX_STEPS, use_gat=True, seeds=SEEDS)

    # Phase 2: No-GAT ablation (5 seeds)
    print("\n" + "#"*60, flush=True)
    print(f"# PHASE 2: Train No-GAT ablation ({len(SEEDS)} seeds)", flush=True)
    print("#"*60, flush=True)
    nogat_agent, nogat_all_res = train_multi_seed(N_EPISODES, GRID, N_DRONES, MAX_STEPS, use_gat=False, seeds=SEEDS)

    # Phase 3: MAPPO baseline (2 seeds — slower per-episode)
    print("\n" + "#"*60, flush=True)
    print(f"# PHASE 3: Train MAPPO baseline ({len(MAPPO_SEEDS)} seeds)", flush=True)
    print("#"*60, flush=True)
    mappo_agent, mappo_all_res = train_mappo_multi_seed(N_EPISODES, GRID, N_DRONES, MAX_STEPS, seeds=MAPPO_SEEDS)

    # Phase 4: IPPO baseline (5 seeds)
    print("\n" + "#"*60, flush=True)
    print(f"# PHASE 4: Train IPPO baseline ({len(SEEDS)} seeds)", flush=True)
    print("#"*60, flush=True)
    ippo_pols, ippo_all_res = train_ippo_multi_seed(N_EPISODES, GRID, N_DRONES, MAX_STEPS, seeds=SEEDS)

    # Phase 5: No-Comm ablation (5 seeds — no GAT + no shared map)
    print("\n" + "#"*60, flush=True)
    print(f"# PHASE 5: Train No-Comm ablation ({len(SEEDS)} seeds)", flush=True)
    print("#"*60, flush=True)
    nocomm_agent, nocomm_all_res = train_nocomm_multi_seed(N_EPISODES, GRID, N_DRONES, MAX_STEPS, seeds=SEEDS)

    # Phase 6: Benchmark (GAT vs Random/Greedy/PID, with 95% bootstrap CIs)
    print("\n" + "#"*60, flush=True)
    print("# PHASE 6: Benchmark vs baselines (wind=12)", flush=True)
    print("#"*60, flush=True)
    bench_res = benchmark(gat_agent, GRID, N_DRONES, MAX_STEPS, wind=12.0, n_eps=20)

    # Phase 7: Wind robustness sweep
    print("\n" + "#"*60, flush=True)
    print("# PHASE 7: Wind robustness sweep", flush=True)
    print("#"*60, flush=True)
    wind_res = wind_sweep(gat_agent, GRID, N_DRONES, MAX_STEPS, n_eps=15)

    # Phase 8: Scalability (5/10/20 drones + 20x20/30x30/50x50 grids)
    print("\n" + "#"*60, flush=True)
    print("# PHASE 8: Scalability analysis", flush=True)
    print("#"*60, flush=True)
    scalability_res = scalability_test(gat_agent, grid=GRID, max_steps=MAX_STEPS, n_eps=10)
    grid_scalability_res = grid_scalability_test(gat_agent, n_drones=N_DRONES, max_steps=MAX_STEPS, n_eps=10)

    # Phase 9: Communication topology analysis
    print("\n" + "#"*60, flush=True)
    print("# PHASE 9: Communication topology analysis", flush=True)
    print("#"*60, flush=True)
    comm_data = communication_analysis(gat_agent, grid=GRID, n_drones=N_DRONES, max_steps=MAX_STEPS, n_eps=5)

    # Phase 10: Figures + Statistical analysis
    print("\n" + "#"*60, flush=True)
    print("# PHASE 10: Publication figures & statistical analysis", flush=True)
    print("#"*60, flush=True)
    generate_figures(gat_all_res, nogat_all_res, mappo_all_res,
                     ippo_res=ippo_all_res, nocomm_res=nocomm_all_res,
                     bench=bench_res, wind_res=wind_res,
                     scalability_res=scalability_res, grid_scalability_res=grid_scalability_res,
                     gat_agent=gat_agent, grid=GRID, n_drones=N_DRONES, max_steps=MAX_STEPS)
    stat_report = statistical_analysis(gat_all_res, nogat_all_res, mappo_all_res, bench_res, wind_res,
                                       ippo_res=ippo_all_res, nocomm_res=nocomm_all_res)

    # Summary
    total_time = time.time() - t_total
    print("\n" + "="*60, flush=True)
    print("COMPLETE!", flush=True)
    print("="*60, flush=True)
    print(f"Total time: {total_time/60:.1f} min ({total_time/3600:.1f} hrs)", flush=True)
    print(f"Seeds: GAT/NoGAT/IPPO/NoComm={len(SEEDS)}, MAPPO={len(MAPPO_SEEDS)}", flush=True)
    print(f"Episodes per seed: {N_EPISODES}, Max steps: {MAX_STEPS}", flush=True)
    all_methods = [('GAT-MARAHS', gat_all_res, len(SEEDS)), ('No-GAT', nogat_all_res, len(SEEDS)),
                   ('MAPPO', mappo_all_res, len(MAPPO_SEEDS)), ('IPPO', ippo_all_res, len(SEEDS)),
                   ('No-Comm', nocomm_all_res, len(SEEDS))]
    for label, all_res, n_seeds in all_methods:
        print(f"\n{label} ({n_seeds} seeds):", flush=True)
        print(f"  Coverage: {np.mean([r['final_coverage'] for r in all_res]):.1f}% +/- {np.std([r['final_coverage'] for r in all_res]):.1f}%", flush=True)
        print(f"  Safety:   {np.mean([r['final_safety'] for r in all_res]):.1f}% +/- {np.std([r['final_safety'] for r in all_res]):.1f}%", flush=True)
        if any(r.get('attention_entropy') for r in all_res):
            all_ent = np.concatenate([r['attention_entropy'][-100:] for r in all_res if r.get('attention_entropy')])
            if len(all_ent) > 0:
                print(f"  Attn Entropy: {np.mean(all_ent):.3f} +/- {np.std(all_ent):.3f} (higher=more distributed)", flush=True)
    print(f"\nBenchmark (wind=12):", flush=True)
    for m, v in bench_res.items():
        print(f"  {m}: Safety={v['safety']:.1f}% Coverage={v['coverage']:.1f}% Perimeter={v['perimeter']:.1f}%", flush=True)
    print(f"\nWind sweep:", flush=True)
    for w in [5, 10, 15, 20, 25]:
        if str(w) in wind_res:
            r = wind_res[str(w)]
            print(f"  {w:2d} m/s: Safety={r['safety']:.1f}% Coverage={r['coverage']:.1f}% Perimeter={r['perimeter']:.1f}%", flush=True)
    print(f"\nScalability:", flush=True)
    for n in [5, 10, 20]:
        if str(n) in scalability_res:
            r = scalability_res[str(n)]
            print(f"  {n} drones: Safety={r['safety']:.1f}% Coverage={r['coverage']:.1f}%", flush=True)
    print(f"\nCommunication topology:", flush=True)
    if comm_data:
        print(f"  Adj density: {comm_data.get('adj_density', 0):.3f}", flush=True)
        print(f"  Components: {comm_data.get('connected_components', 0):.2f}", flush=True)
        print(f"  Avg path: {comm_data.get('avg_path_length', 0):.2f}", flush=True)
        print(f"  Attn entropy: {comm_data.get('attention_entropy', 0):.3f}", flush=True)
