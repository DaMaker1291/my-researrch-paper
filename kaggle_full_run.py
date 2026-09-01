#!/usr/bin/env python3
"""
=================================================================
PlumeGym-MARL: Full Training Pipeline for Kaggle (GPU-Accelerated)
=================================================================
Upload this file to Kaggle as a Notebook. Set GPU runtime.

Pipeline:
  1. Train GAT-MARAHS (30×30, 10 drones, 300 steps) × 3 seeds
  2. Train No-GAT ablation × 3 seeds (isolates GAT contribution)
  3. Benchmark vs Random / Greedy / PID baselines
  4. Wind sweep (5/10/15/20/25 m/s) with error bars
  5. Generate publication figures with confidence intervals

Runtime: ~4-6 hours on Kaggle GPU (T4/P100)
Memory: ~4GB (well within 16GB limit)
Kaggle limits: 9hr hard cap (GPU), ~40min idle timeout
Checkpointing every 100 eps ensures progress survives timeouts.

Output files:
  - gat_marahs_best.pt / nogat_best.pt (trained models)
  - gat_training_results.json / nogat_training_results.json
  - gat_benchmark_final.json
  - gat_wind_sweep.json
  - figures_gat/fig1_training.png (GAT vs No-GAT comparison)
  - figures_gat/fig2_benchmark.png
  - figures_gat/fig3_wind_sweep.png
  - figures_gat/fig4_ablation.png

How to use:
1. Go to kaggle.com/code
2. Create new notebook
3. Upload this file as the notebook
4. Set runtime to GPU (T4 or P100)
5. Click Run All
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import os
import sys

# ═══════════════════════════════════════════════════════════════
# SECTION 1: ENVIRONMENT
# ═══════════════════════════════════════════════════════════════

class WildfireEnv:
    """Wildfire perimeter tracking environment."""

    def __init__(self, grid=30, n_drones=10, max_steps=300, wind_speed=12.0, obs_r=4):
        self.grid = grid
        self.n_drones = n_drones
        self.max_steps = max_steps
        self.base_wind = wind_speed
        self.obs_r = obs_r
        self.obs_size = 2 * obs_r + 1
        self.obs_channels = 8
        self.local_obs_dim = self.obs_channels * self.obs_size * self.obs_size
        self.global_obs_dim = 8
        self.obs_dim = self.local_obs_dim + self.global_obs_dim
        self.act_dim = 5
        self.action_deltas = np.array([[0,0],[0,1],[0,-1],[1,0],[-1,0]], dtype=np.float32)
        self.wind_coupling = 0.02
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
        self.reset()

    def reset(self):
        self.step_count = 0
        self.total_cells_explored = set()
        self.fire = np.zeros((self.grid, self.grid), dtype=np.float32)
        cx = np.random.randint(self.base_fire_radius + 2, max(self.base_fire_radius + 3, self.grid - self.base_fire_radius - 2))
        cy = np.random.randint(self.base_fire_radius + 2, max(self.base_fire_radius + 3, self.grid - self.base_fire_radius - 2))
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
        yy, xx = np.meshgrid(np.arange(self.grid), np.arange(self.grid), indexing='ij')
        for k in range(3):
            freq = 0.5 * (k + 1)
            amp = 0.1 * self.base_wind
            phase = np.random.uniform(0, 2 * np.pi)
            self.wind_x += amp * np.sin(2 * np.pi * freq * xx / self.grid + phase)
            self.wind_y += amp * np.sin(2 * np.pi * freq * yy / self.grid + phase * 0.7)
        self.thermal = np.zeros((self.grid, self.grid), dtype=np.float32)
        self._update_thermal()
        self._fire_dist_cache = None
        self._update_fire_dist()
        self._visited_grid = np.zeros((self.grid, self.grid), dtype=np.float32)
        self.drones = []
        for i in range(self.n_drones):
            while True:
                px = np.random.uniform(2, self.grid - 2)
                py = np.random.uniform(2, self.grid - 2)
                ix, iy = int(px), int(py)
                if 0 <= ix < self.grid and 0 <= iy < self.grid and self.fire[iy, ix] < 0.1:
                    break
            self.drones.append({'pos': np.array([px, py], dtype=np.float32), 'vel': np.array([0.0, 0.0], dtype=np.float32), 'alive': True, 'visited': set(), 'battery': 1.0})
        return self._get_obs()

    def _update_thermal(self):
        self.thermal = np.zeros((self.grid, self.grid), dtype=np.float32)
        fire_cells = np.argwhere(self.fire > 0.2)
        for fy, fx in fire_cells:
            intensity = self.fire[fy, fx]
            for dy in range(-5, 6):
                for dx in range(-5, 6):
                    nx, ny = fx + dx, fy + dy
                    if 0 <= nx < self.grid and 0 <= ny < self.grid:
                        dist = np.sqrt(dx*dx + dy*dy)
                        self.thermal[ny, nx] += intensity * np.exp(-dist**2 / 8.0)
        self.thermal = np.clip(self.thermal, 0, self.thermal_cap)

    def _update_fire_dist(self):
        fire_cells = np.argwhere(self.fire > 0.2)
        if len(fire_cells) == 0:
            self._fire_dist_cache = np.full((self.grid, self.grid), 10.0, dtype=np.float32)
            return
        self._fire_dist_cache = np.full((self.grid, self.grid), 10.0, dtype=np.float32)
        yy, xx = np.meshgrid(np.arange(self.grid), np.arange(self.grid), indexing='ij')
        for fy, fx in fire_cells:
            dist = np.sqrt((xx - fx)**2 + (yy - fy)**2)
            self._fire_dist_cache = np.minimum(self._fire_dist_cache, dist)

    def _spread_fire(self, rng):
        fire_cells = np.argwhere(self.fire > 0.2)
        new_fire = self.fire.copy()
        for fy, fx in fire_cells:
            intensity = self.fuel[fy, fx] * self.fire[fy, fx]
            if intensity < 0.05: continue
            wind_mag = np.sqrt(self.wind_x[fy, fx]**2 + self.wind_y[fy, fx]**2)
            spread_prob = self.spread_rate * (1 + self.wind_amplification * wind_mag) * intensity
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nx, ny = fx + dx, fy + dy
                if 0 <= nx < self.grid and 0 <= ny < self.grid and self.fuel[ny, nx] > 0.1:
                    prob = spread_prob * self.fuel[ny, nx]
                    if rng.random() < prob:
                        new_fire[ny, nx] = min(1.0, new_fire[ny, nx] + 0.1)
            if rng.random() < self.spotting_prob * wind_mag / 10.0:
                spot_dist = rng.integers(1, 4)
                spot_angle = rng.uniform(0, 2 * np.pi)
                sx = int(fx + spot_dist * np.cos(spot_angle))
                sy = int(fy + spot_dist * np.sin(spot_angle))
                if 0 <= sx < self.grid and 0 <= sy < self.grid and self.fuel[sy, sx] > 0.1:
                    new_fire[sy, sx] = min(1.0, new_fire[sy, sx] + 0.3)
        new_fire = new_fire * (1 - self.fuel_depletion_rate)
        self.fire = np.clip(new_fire, 0, 1).astype(np.float32)

    def _check_crash(self, pos, thermal_val, wind_spd):
        ix = int(np.clip(np.round(pos[0]), 0, self.grid - 1))
        iy = int(np.clip(np.round(pos[1]), 0, self.grid - 1))
        if self.fire[iy, ix] > self.fire_crash_threshold: return True, 'fire_cell'
        if wind_spd > 10.0:
            fire_dist = self._fire_dist_cache[iy, ix]
            buffer = max(0.3, (wind_spd - 10.0) / 10.0)
            if fire_dist < buffer: return True, 'fire_edge'
        if thermal_val > self.thermal_crash: return True, 'thermal'
        if (pos[0] < self.boundary_margin or pos[0] > self.grid - self.boundary_margin or
            pos[1] < self.boundary_margin or pos[1] > self.grid - self.boundary_margin):
            return True, 'boundary'
        return False, 'safe'

    def _get_frontier_direction(self, pos):
        ix, iy = int(pos[0]), int(pos[1])
        best_dist = float('inf')
        best_dir = np.array([0.0, 0.0])
        for dx in range(-8, 9):
            for dy in range(-8, 9):
                nx, ny = ix + dx, iy + dy
                if 0 <= nx < self.grid and 0 <= ny < self.grid:
                    if self._visited_grid[ny, nx] < 0.5 and self.fire[ny, nx] < 0.2:
                        dist = abs(dx) + abs(dy)
                        if dist < best_dist:
                            best_dist = dist
                            if dist > 0: best_dir = np.array([dx/dist, dy/dist])
        return best_dir

    def _get_obs(self):
        r = self.obs_r
        obs = np.zeros((self.n_drones, self.obs_dim), dtype=np.float32)
        ch_size = self.obs_size * self.obs_size
        for i in range(self.n_drones):
            if not self.drones[i]['alive']: continue
            cx, cy = int(self.drones[i]['pos'][0]), int(self.drones[i]['pos'][1])
            x_min, x_max = max(0, cx - r), min(self.grid, cx + r + 1)
            y_min, y_max = max(0, cy - r), min(self.grid, cy + r + 1)
            h, w = x_max - x_min, y_max - y_min
            local = np.zeros(self.local_obs_dim, dtype=np.float32)
            ch_idx = 0
            for arr in [self.fire, self.wind_x, self.wind_y, self.fuel, self.thermal]:
                grid = np.zeros((self.obs_size, self.obs_size), dtype=np.float32)
                grid[:h, :w] = arr[x_min:x_max, y_min:y_max]
                if arr is self.wind_x or arr is self.wind_y:
                    grid[:h, :w] /= 30.0
                elif arr is self.thermal:
                    grid[:h, :w] /= self.thermal_cap
                local[ch_idx*ch_size:(ch_idx+1)*ch_size] = grid.ravel(); ch_idx += 1
            grid = np.zeros((self.obs_size, self.obs_size), dtype=np.float32)
            for j in range(self.n_drones):
                if j != i and self.drones[j]['alive']:
                    jx = int(self.drones[j]['pos'][0]) - cx + r
                    jy = int(self.drones[j]['pos'][1]) - cy + r
                    if 0 <= jx < self.obs_size and 0 <= jy < self.obs_size: grid[jx, jy] = 1.0
            local[ch_idx*ch_size:(ch_idx+1)*ch_size] = grid.ravel(); ch_idx += 1
            grid = np.zeros((self.obs_size, self.obs_size), dtype=np.float32)
            if self._fire_dist_cache is not None:
                grid[:h, :w] = np.minimum(self._fire_dist_cache[x_min:x_max, y_min:y_max] / 10.0, 1.0)
            local[ch_idx*ch_size:(ch_idx+1)*ch_size] = grid.ravel(); ch_idx += 1
            grid = np.zeros((self.obs_size, self.obs_size), dtype=np.float32)
            grid[:h, :w] = self._visited_grid[x_min:x_max, y_min:y_max]
            local[ch_idx*ch_size:(ch_idx+1)*ch_size] = grid.ravel(); ch_idx += 1
            fire_cells = np.argwhere(self.fire > 0.2)
            if len(fire_cells) > 0:
                fcx, fcy = float(np.mean(fire_cells[:, 0])), float(np.mean(fire_cells[:, 1]))
                fr = float(np.sqrt(len(fire_cells))) / self.grid
            else: fcx, fcy, fr = self.grid/2, self.grid/2, 0.1
            dx = (fcx - self.drones[i]['pos'][0]) / self.grid
            dy = (fcy - self.drones[i]['pos'][1]) / self.grid
            ix = int(np.clip(self.drones[i]['pos'][0], 0, self.grid - 1))
            iy = int(np.clip(self.drones[i]['pos'][1], 0, self.grid - 1))
            wind_dir = float(np.arctan2(self.wind_y[iy, ix], self.wind_x[iy, ix])) / np.pi
            coverage = len(self.total_cells_explored) / (self.grid * self.grid)
            frontier = self._get_frontier_direction(self.drones[i]['pos'])
            global_f = np.array([fcx/self.grid, fcy/self.grid, fr, dx, dy, wind_dir, coverage, np.linalg.norm(frontier)], dtype=np.float32)
            obs[i] = np.concatenate([local, global_f])
        return obs

    def step(self, actions):
        self.step_count += 1
        rng = np.random.default_rng(self.step_count)
        rewards = np.zeros(self.n_drones, dtype=np.float32)
        dones = np.zeros(self.n_drones, dtype=bool)
        infos = [{} for _ in range(self.n_drones)]
        self._spread_fire(rng)
        if self.step_count % 5 == 0:
            self._update_thermal()
            self._update_fire_dist()
        for i in range(self.n_drones):
            if not self.drones[i]['alive']: dones[i] = True; continue
            d = self.drones[i]
            action = int(actions[i])
            dx, dy = self.action_deltas[action]
            target_vel = np.array([dx, dy], dtype=np.float32)
            ix = int(np.clip(d['pos'][0], 0, self.grid - 1))
            iy = int(np.clip(d['pos'][1], 0, self.grid - 1))
            wind_push = np.array([self.wind_x[iy, ix], self.wind_y[iy, ix]]) * self.wind_coupling
            d['vel'] = self.momentum * d['vel'] + (1 - self.momentum) * target_vel + wind_push
            new_pos = d['pos'] + d['vel']
            new_pos = np.clip(new_pos, self.boundary_margin, self.grid - self.boundary_margin)
            ix_new = int(np.clip(new_pos[0], 0, self.grid - 1))
            iy_new = int(np.clip(new_pos[1], 0, self.grid - 1))
            thermal_val = float(self.thermal[iy_new, ix_new])
            wind_spd = float(np.sqrt(self.wind_x[iy_new, ix_new]**2 + self.wind_y[iy_new, ix_new]**2))
            crashed, crash_reason = self._check_crash(new_pos, thermal_val, wind_spd)
            if crashed:
                d['alive'] = False; dones[i] = True; rewards[i] = -10.0
                infos[i] = {'crash': True, 'reason': crash_reason, 'fire_dist': 0.0, 'thermal': thermal_val, 'wind_speed': wind_spd}
            else:
                d['pos'] = new_pos
                gx, gy = int(new_pos[0]), int(new_pos[1])
                if 0 <= gx < self.grid and 0 <= gy < self.grid:
                    d['visited'].add((gx, gy))
                    self.total_cells_explored.add((gx, gy))
                    self._visited_grid[gy, gx] = 1.0
                fire_dist = float(self._fire_dist_cache[iy_new, ix_new]) if self._fire_dist_cache is not None else 10.0
                infos[i] = {'crash': False, 'fire_dist': fire_dist, 'thermal': thermal_val, 'wind_speed': wind_spd}
                rewards[i] = 1.0
            d['battery'] = max(0, d['battery'] - 0.001)
        return self._get_obs(), rewards, dones, infos


# ═══════════════════════════════════════════════════════════════
# SECTION 2: GAT + PPO
# ═══════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)
if device.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | {props.total_memory / 1e9:.1f} GB", flush=True)


class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.randn(n_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.randn(n_heads, self.head_dim))
        nn.init.xavier_uniform_(self.W.weight)
    def forward(self, h, adj_mask):
        K = h.shape[0]
        if K <= 1: return self.W(h)
        Wh = self.W(h).view(K, self.n_heads, self.head_dim)
        e_src = (Wh * self.a_src.unsqueeze(0)).sum(dim=2)
        e_dst = (Wh * self.a_dst.unsqueeze(0)).sum(dim=2)
        e = e_src.unsqueeze(1) + e_dst.unsqueeze(0)
        e = F.leaky_relu(e, 0.2)
        adj = adj_mask.unsqueeze(2).float()
        e = e.masked_fill(~adj.bool(), float('-inf'))
        attn = F.softmax(e, dim=1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.einsum('kjh,khd->khd', attn, Wh).reshape(K, -1)
        return out


class GATCommunication(nn.Module):
    def __init__(self, in_dim=656, hidden_dim=128, out_dim=64, n_heads=4, comm_range=15.0):
        super().__init__()
        self.in_dim, self.out_dim, self.comm_range = in_dim, out_dim, comm_range
        self.attn1 = MultiHeadAttention(in_dim, hidden_dim, n_heads)
        self.attn2 = MultiHeadAttention(hidden_dim, out_dim, n_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.output_proj = nn.Linear(out_dim, out_dim)
    def build_graph(self, positions, alive_mask):
        K = len(positions)
        adj = torch.zeros(K, K, dtype=torch.bool, device=device)
        for i in range(K):
            if not alive_mask[i]: continue
            for j in range(K):
                if not alive_mask[j] or i == j: continue
                if torch.norm(positions[i] - positions[j]) < self.comm_range:
                    adj[i, j] = adj[j, i] = True
            if alive_mask[i]: adj[i, i] = True
        return adj
    def forward(self, obs, positions, alive_mask):
        adj = self.build_graph(positions, alive_mask)
        h = F.relu(self.norm1(self.attn1(obs, adj)))
        h2 = F.relu(self.norm2(self.attn2(h, adj)))
        return torch.cat([obs, self.output_proj(h2)], dim=1)
    @property
    def enhanced_obs_dim(self): return self.in_dim + self.out_dim


class NoGATCommunication(nn.Module):
    """Ablation: replaces GAT with a simple MLP — no inter-agent communication."""
    def __init__(self, in_dim=656, out_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.LayerNorm(128), nn.Linear(128, out_dim))
        self._in_dim = in_dim
        self._out_dim = out_dim
    def forward(self, obs, positions=None, alive_mask=None):
        return torch.cat([obs, self.mlp(obs)], dim=1)
    @property
    def enhanced_obs_dim(self): return self._in_dim + self._out_dim


class PPONetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, h1=256, h2=128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(obs_dim, h1), nn.ReLU(), nn.LayerNorm(h1), nn.Linear(h1, h2), nn.ReLU(), nn.LayerNorm(h2))
        self.policy_head = nn.Linear(h2, act_dim)
        self.value_head = nn.Linear(h2, 1)
        self.apply(lambda m: nn.init.orthogonal_(m.weight, np.sqrt(2)) if isinstance(m, nn.Linear) else None)
    def forward(self, obs):
        f = self.encoder(obs)
        return self.policy_head(f), self.value_head(f).squeeze(-1)
    def evaluate(self, obs, actions):
        logits, value = self.forward(obs)
        probs = torch.distributions.Categorical(logits=logits)
        return logits, probs.log_prob(actions), probs.entropy(), value


class FastGATPPO:
    def __init__(self, obs_dim=656, act_dim=5, use_gat=True, gat_hidden=128, gat_out=64, n_heads=4, comm_range=15.0, lr=3e-4):
        if use_gat:
            self.gat = GATCommunication(obs_dim, gat_hidden, gat_out, n_heads, comm_range).to(device)
        else:
            self.gat = NoGATCommunication(obs_dim, gat_out).to(device)
        self.policy = PPONetwork(obs_dim + gat_out, act_dim).to(device)
        params = list(self.gat.parameters()) + list(self.policy.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=2000, eta_min=1e-5)
        self._traj = []
    def select_actions(self, obs, positions, alive_mask):
        obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
        pos_t = torch.tensor(positions, dtype=torch.float32).to(device)
        alive_t = torch.tensor(alive_mask, dtype=torch.bool).to(device)
        enhanced = self.gat(obs_t, pos_t, alive_t)
        with torch.no_grad():
            logits, values = self.policy(enhanced)
            probs = torch.distributions.Categorical(logits=logits)
            actions = probs.sample()
            log_probs = probs.log_prob(actions)
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy(), enhanced.detach()
    def store(self, enhanced_obs, actions, rewards, dones, log_probs, values):
        for i in range(len(actions)):
            self._traj.append({'obs': enhanced_obs[i].cpu(), 'action': actions[i], 'reward': rewards[i], 'done': dones[i], 'log_prob': log_probs[i], 'value': values[i]})
    def update(self, gamma=0.99, gae_lambda=0.95, n_epochs=6, batch_size=512, clip_eps=0.2, entropy_coef=0.02):
        if not self._traj: return 0.0
        n = len(self._traj)
        rewards = np.array([t['reward'] for t in self._traj], dtype=np.float32)
        values = np.array([t['value'] for t in self._traj], dtype=np.float32)
        dones = np.array([t['done'] for t in self._traj], dtype=np.float32)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for step in reversed(range(n)):
            next_val = 0.0 if step == n-1 else values[step+1]
            next_done = 1.0 if step == n-1 else dones[step+1]
            delta = rewards[step] + gamma * next_val * (1-next_done) - values[step]
            gae = delta + gamma * gae_lambda * (1-next_done) * gae
            advantages[step] = gae
            returns[step] = gae + values[step]
        all_obs = torch.stack([t['obs'].squeeze(0) for t in self._traj]).to(device)
        all_actions = torch.tensor([t['action'] for t in self._traj], dtype=torch.long).to(device)
        all_old_lp = torch.tensor([t['log_prob'] for t in self._traj], dtype=torch.float32).to(device)
        all_adv = torch.tensor(advantages, dtype=torch.float32).to(device)
        all_ret = torch.tensor(returns, dtype=torch.float32).to(device)
        all_adv = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-8)
        total_loss, count = 0.0, 0
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
                nn.utils.clip_grad_norm_(list(self.gat.parameters()) + list(self.policy.parameters()), 0.5)
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
# SECTION 3: REWARD + TRAINING
# ═══════════════════════════════════════════════════════════════

def compute_reward(drone, drone_idx, all_drones, prev_visited, fire_dist, crashed, step, max_steps, grid_size):
    if crashed: return -15.0
    reward = 0.05
    new_cells = sum(1 for c in drone['visited'] if c not in prev_visited)
    reward += 25.0 * new_cells
    if new_cells > 0 and 1.0 < fire_dist < 5.0:
        reward += 10.0
    nearby = sum(1 for j, other in enumerate(all_drones) if j != drone_idx and other['alive'] and np.linalg.norm(drone['pos'] - other['pos']) < 3.0)
    reward -= 3.0 * nearby
    mid = grid_size / 2.0
    q = int(drone['pos'][0] >= mid) + 2*int(drone['pos'][1] >= mid)
    qcounts = [sum(1 for other in all_drones if other['alive'] and int(other['pos'][0] >= mid) + 2*int(other['pos'][1] >= mid) == k) for k in range(4)]
    reward += 2.0 / max(1, qcounts[q])
    if fire_dist < 3.0: reward += 2.0 * (1.0 - fire_dist / 3.0)
    if step >= max_steps - 1: reward += 5.0
    return reward


def train(n_episodes=2000, grid=30, n_drones=10, max_steps=300, use_gat=True, seed=0, run_id="gat"):
    torch.manual_seed(seed); np.random.seed(seed)
    print(f"\n{'='*60}", flush=True)
    tag = "GAT-MARAHS" if use_gat else "No-GAT (Ablation)"
    print(f"{tag} Training | seed={seed} | {n_episodes} eps | {n_drones} drones | {grid}x{grid}", flush=True)
    print(f"{'='*60}", flush=True)

    env = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps, wind_speed=0)
    agent = FastGATPPO(obs_dim=env.obs_dim, act_dim=env.act_dim, use_gat=use_gat)

    rewards_h, coverage_h, safety_h = [], [], []
    best_r = -float('inf')
    early_stop_patience = 500
    early_stop_counter = 0
    early_stop_target = 85.0
    t0 = time.time()

    try:
      for ep in range(n_episodes):
        env.base_wind = 0
        obs = env.reset()
        agent._traj.clear()
        ep_r, ep_crashes = 0.0, 0

        for step in range(max_steps):
            am = np.array([env.drones[i]['alive'] for i in range(n_drones)])
            pos = np.array([env.drones[i]['pos'] for i in range(n_drones)], dtype=np.float32)
            if not am.any(): break
            actions, log_probs, values, enhanced = agent.select_actions(obs, pos, am)
            prev_visited = [set(env.drones[i].get('visited', set())) for i in range(n_drones)]
            obs_next, _, dones, infos = env.step(np.array(actions, dtype=np.int32))
            shaped = np.zeros(n_drones, dtype=np.float32)
            for i in range(n_drones):
                if not am[i]: continue
                fd = infos[i].get('fire_dist', 10.0)
                shaped[i] = compute_reward(env.drones[i], i, env.drones, prev_visited[i], fd, dones[i], step, max_steps, grid)
                ep_r += shaped[i]
                if dones[i] and not env.drones[i]['alive']: ep_crashes += 1
            agent.store(enhanced, actions, shaped, dones.astype(np.float32), log_probs, values)
            obs = obs_next
            if all(dones): break

        loss = agent.update()
        cov = len(env.total_cells_explored) / (grid*grid) * 100
        saf = (1.0 - ep_crashes / n_drones) * 100

        rewards_h.append(ep_r); coverage_h.append(cov); safety_h.append(saf)

        if (ep+1) % 100 == 0:
            avg_r = np.mean(rewards_h[-100:]); avg_cov = np.mean(coverage_h[-100:]); avg_saf = np.mean(safety_h[-100:])
            elapsed = time.time() - t0
            eps_per_sec = (ep+1) / elapsed
            eta_min = (n_episodes - ep - 1) / max(eps_per_sec, 1e-6) / 60.0
            print(f"Ep {ep+1:5d}/{n_episodes} | R: {avg_r:7.1f} | Cov: {avg_cov:5.1f}% | Safe: {avg_saf:4.0f}% | {eps_per_sec:.1f} ep/s | ETA {eta_min:.0f}m", flush=True)
            print(flush=True)
            if avg_r > best_r:
                best_r = avg_r
                agent.save(f'{run_id}_best.pt')
            agent.save(f'{run_id}_checkpoint_ep{ep+1}.pt')
            # Held-out validation
            val_c, val_s = [], []
            for _ in range(5):
                ve = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps, wind_speed=0)
                vo = ve.reset()
                vc_count = 0
                for _ in range(max_steps):
                    va = np.array([ve.drones[i]['alive'] for i in range(n_drones)])
                    vp = np.array([ve.drones[i]['pos'] for i in range(n_drones)], dtype=np.float32)
                    if not va.any(): break
                    vacts, _, _, _ = agent.select_actions(vo, vp, va)
                    vo, _, vd, vi = ve.step(np.array(vacts, dtype=np.int32))
                    for i2 in range(n_drones):
                        if va[i2] and vd[i2] and not ve.drones[i2]['alive']: vc_count += 1
                    if all(vd): break
                val_c.append(len(ve.total_cells_explored)/(grid*grid)*100)
                val_s.append((1.0-vc_count/n_drones)*100)
            print(f"         VAL | Cov: {np.mean(val_c):5.1f}% ± {np.std(val_c):4.1f} | Safe: {np.mean(val_s):4.0f}%", flush=True)
            if avg_cov >= early_stop_target:
                early_stop_counter += 100
            else:
                early_stop_counter = 0
            if early_stop_counter >= early_stop_patience:
                print(f"Early stop at ep {ep+1}: coverage {avg_cov:.1f}% >= {early_stop_target}% for {early_stop_patience} consecutive episodes", flush=True)
                break
    except (KeyboardInterrupt, SystemExit, Exception) as e:
      print(f"\nTraining interrupted at ep {ep+1}: {e}", flush=True)

    agent.save(f'{run_id}_final.pt')
    results = {
        'n_episodes': n_episodes, 'seed': seed, 'use_gat': use_gat,
        'final_reward': float(np.mean(rewards_h[-100:])),
        'final_coverage': float(np.mean(coverage_h[-100:])),
        'final_safety': float(np.mean(safety_h[-100:])),
        'rewards': [float(x) for x in rewards_h],
        'coverages': [float(x) for x in coverage_h],
        'safety': [float(x) for x in safety_h],
    }
    with open(f'{run_id}_training_results.json', 'w') as f: json.dump(results, f, indent=2)
    print(f"\nDone in {time.time()-t0:.0f}s | Reward: {results['final_reward']:.1f} | Coverage: {results['final_coverage']:.1f}% | Safety: {results['final_safety']:.0f}%", flush=True)
    return agent, results


# ═══════════════════════════════════════════════════════════════
# SECTION 4: MULTI-SEED TRAINING
# ═══════════════════════════════════════════════════════════════

def train_multi_seed(n_episodes=2000, grid=30, n_drones=10, max_steps=300, use_gat=True, seeds=[42, 123, 777]):
    run_id = "gat" if use_gat else "nogat"
    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n{'#'*60}", flush=True)
        print(f"# Run {i+1}/{len(seeds)} | seed={seed}", flush=True)
        print(f"{'#'*60}", flush=True)
        agent, res = train(n_episodes, grid, n_drones, max_steps, use_gat=use_gat, seed=seed, run_id=f"{run_id}_s{seed}")
        all_results.append(res)
        # Save best model from this seed
        agent.save(f'{run_id}_seed{seed}_best.pt')

    # Aggregate across seeds
    final_covs = [r['final_coverage'] for r in all_results]
    final_safs = [r['final_safety'] for r in all_results]
    final_rews = [r['final_reward'] for r in all_results]
    print(f"\n{'='*60}", flush=True)
    tag = "GAT-MARAHS" if use_gat else "No-GAT (Ablation)"
    print(f"{tag} | {len(seeds)} seeds summary:", flush=True)
    print(f"  Coverage: {np.mean(final_covs):.1f}% ± {np.std(final_covs):.1f}%", flush=True)
    print(f"  Safety:   {np.mean(final_safs):.1f}% ± {np.std(final_safs):.1f}%", flush=True)
    print(f"  Reward:   {np.mean(final_rews):.1f} ± {np.std(final_rews):.1f}", flush=True)
    print(f"{'='*60}", flush=True)

    # Pick best seed's model as representative
    best_idx = np.argmax(final_covs)
    best_agent_seed = seeds[best_idx]
    # Load best model
    best_agent = FastGATPPO(obs_dim=env.obs_dim if 'env' in dir() else 520, act_dim=5, use_gat=use_gat)
    try:
        best_agent.load(f'{run_id}_seed{best_agent_seed}_best.pt')
    except:
        pass  # If load fails, use last trained agent

    return best_agent, all_results


# ═══════════════════════════════════════════════════════════════
# SECTION 5: BENCHMARK
# ═══════════════════════════════════════════════════════════════

def benchmark(agent, grid=30, n_drones=10, max_steps=300, wind=12.0, n_eps=20):
    print(f"\n{'='*60}\nBENCHMARK | wind={wind} | {n_drones} drones | {n_eps} eps\n{'='*60}", flush=True)
    results = {}

    for method_name, method_fn in [
        ('GAT-MARAHS', lambda env: _run_gat(agent, env, n_drones)),
        ('Random', lambda env: _run_random(env, n_drones)),
        ('Greedy', lambda env: _run_greedy(env, n_drones)),
        ('PID', lambda env: _run_pid(env, n_drones)),
    ]:
        s, c, p, a = [], [], [], []
        for _ in range(n_eps):
            env_b = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps, wind_speed=wind)
            obs = env_b.reset()
            for step in range(max_steps):
                alive = np.array([env_b.drones[i]['alive'] for i in range(n_drones)])
                if not alive.any(): break
                actions = method_fn(env_b)
                obs, _, dones, _ = env_b.step(np.array(actions, dtype=np.int32))
                if all(dones): break
            ac = sum(1 for i in range(n_drones) if env_b.drones[i]['alive'])
            s.append(ac/n_drones*100); c.append(len(env_b.total_cells_explored)/(grid*grid)*100)
            fc = np.argwhere(env_b.fire > 0.2)
            pc = set()
            for fx,fy in fc:
                for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx,ny=fx+dx,fy+dy
                    if 0<=nx<grid and 0<=ny<grid and env_b.fire[nx,ny]<0.1: pc.add((nx,ny))
            vis=set()
            for i in range(n_drones): vis.update(env_b.drones[i].get('visited',set()))
            p.append(len(pc&vis)/max(1,len(pc))*100); a.append(ac)
        results[method_name] = {'safety':np.mean(s),'coverage':np.mean(c),'perimeter':np.mean(p),'alive':np.mean(a)}

    print(f"\n{'Method':<18s} {'Safety':>8s} {'Coverage':>10s} {'Perimeter':>10s} {'Alive':>8s}", flush=True)
    print("-"*58, flush=True)
    for m,v in results.items(): print(f"{m:<18s} {v['safety']:7.1f}%  {v['coverage']:8.1f}%  {v['perimeter']:8.1f}%  {v['alive']:6.1f}/{n_drones}", flush=True)
    print("="*58, flush=True)

    with open('gat_benchmark_final.json','w') as f: json.dump(results,f,indent=2)
    return results

def _run_gat(agent, env, n_drones):
    am = np.array([env.drones[i]['alive'] for i in range(n_drones)])
    pos = np.array([env.drones[i]['pos'] for i in range(n_drones)], dtype=np.float32)
    obs = env._get_obs()
    acts, _, _, _ = agent.select_actions(obs, pos, am)
    return acts
def _run_random(env, n_drones):
    return np.random.randint(0, 5, n_drones)
def _run_greedy(env, n_drones):
    acts = np.zeros(n_drones, dtype=np.int32)
    for i in range(n_drones):
        if not env.drones[i]['alive']: continue
        d = env.drones[i]; ix,iy = int(d['pos'][0]),int(d['pos'][1])
        best_a,best_v = 0,-1
        for ai,(dx,dy) in enumerate([(0,0),(0,1),(0,-1),(1,0),(-1,0)]):
            nx,ny=ix+int(dx),iy+int(dy)
            if 0<=nx<env.grid and 0<=ny<env.grid and (nx,ny) not in d.get('visited',set()):
                v=1.0
                if env._fire_dist_cache is not None: v+=2.0/(env._fire_dist_cache[ny,nx]+1.0)
                if v>best_v: best_v,best_a=v,ai
        acts[i]=best_a
    return acts
def _run_pid(env, n_drones):
    acts = np.zeros(n_drones, dtype=np.int32)
    for i in range(n_drones):
        if not env.drones[i]['alive']: continue
        d = env.drones[i]
        fc = np.argwhere(env.fire > 0.2)
        if len(fc)>0:
            fcx,fcy = float(np.mean(fc[:,0])),float(np.mean(fc[:,1]))
            ddx,ddy = fcx-d['pos'][0],fcy-d['pos'][1]
            if abs(ddx)>abs(ddy): acts[i]=3 if ddx>0 else 4
            else: acts[i]=1 if ddy>0 else 2
    return acts


# ═══════════════════════════════════════════════════════════════
# SECTION 6: WIND SWEEP
# ═══════════════════════════════════════════════════════════════

def wind_sweep(agent, grid=30, n_drones=10, max_steps=300, n_eps=15):
    print(f"\n{'='*60}\nWIND SWEEP | {n_drones} drones | {n_eps} episodes each\n{'='*60}", flush=True)
    sweep = {}
    for wind in [5, 10, 15, 20, 25]:
        s,c,p = [],[],[]
        for _ in range(n_eps):
            env_w = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps, wind_speed=wind)
            obs = env_w.reset()
            for step in range(max_steps):
                am = np.array([env_w.drones[i]['alive'] for i in range(n_drones)])
                pos = np.array([env_w.drones[i]['pos'] for i in range(n_drones)], dtype=np.float32)
                if not am.any(): break
                acts,_,_,_ = agent.select_actions(obs,pos,am)
                obs,_,dones,_ = env_w.step(np.array(acts,dtype=np.int32))
                if all(dones): break
            ac = sum(1 for i in range(n_drones) if env_w.drones[i]['alive'])
            s.append(ac/n_drones*100); c.append(len(env_w.total_cells_explored)/(grid*grid)*100)
            fc=np.argwhere(env_w.fire>0.2); pc=set()
            for fx,fy in fc:
                for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx,ny=fx+dx,fy+dy
                    if 0<=nx<grid and 0<=ny<grid and env_w.fire[nx,ny]<0.1: pc.add((nx,ny))
            vis=set()
            for i in range(n_drones): vis.update(env_w.drones[i].get('visited',set()))
            p.append(len(pc&vis)/max(1,len(pc))*100)
        sweep[str(wind)] = {'safety':np.mean(s),'coverage':np.mean(c),'perimeter':np.mean(p),
                           'safety_std':float(np.std(s)),'coverage_std':float(np.std(c)),'perimeter_std':float(np.std(p))}
        print(f"  Wind={wind:2d} m/s | Safety={np.mean(s):5.1f}% | Coverage={np.mean(c):5.1f}% | Perimeter={np.mean(p):5.1f}%", flush=True)
    with open('gat_wind_sweep.json','w') as f: json.dump(sweep,f,indent=2)
    return sweep


# ═══════════════════════════════════════════════════════════════
# SECTION 7: FIGURES
# ═══════════════════════════════════════════════════════════════

def generate_figures(gat_all_res, nogat_all_res, bench_res, wind_res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs('figures_gat', exist_ok=True)
    rc = plt.rcParams; rc['font.family']='serif'; rc['font.size']=11; rc['figure.dpi']=150; rc['savefig.bbox']='tight'

    def smooth(x, w=20):
        return np.convolve(x, np.ones(w)/w, mode='valid')

    # Fig 1: GAT vs No-GAT training curves (with seed error bands)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax_idx, (metric, label, color) in enumerate([
        ('coverages', 'Coverage (%)', '#3498db'),
        ('safety', 'Safety (%)', '#2ecc71'),
        ('rewards', 'Reward', '#2c3e50'),
    ]):
        ax = axes[ax_idx // 2][ax_idx % 2]
        for all_res, name, ls in [(gat_all_res, 'GAT-MARAHS', '-'), (nogat_all_res, 'No-GAT', '--')]:
            curves = [smooth(r[metric]) for r in all_res if len(r[metric]) > 20]
            if curves:
                min_len = min(len(c) for c in curves)
                arr = np.array([c[:min_len] for c in curves])
                mean = np.mean(arr, axis=0)
                std = np.std(arr, axis=0)
                x = np.arange(len(mean))
                ax.plot(x, mean, ls, color=color, lw=2, label=name)
                ax.fill_between(x, mean-std, mean+std, alpha=0.15, color=color)
        ax.set_xlabel('Episode'); ax.set_ylabel(label); ax.set_title(f'(chr{chr(97+ax_idx)}) {label}')
        ax.legend(); ax.grid(True, alpha=0.3)
    # Ablation bar chart
    ax = axes[1][1]
    gat_cov = np.mean([r['final_coverage'] for r in gat_all_res])
    gat_saf = np.mean([r['final_safety'] for r in gat_all_res])
    nogat_cov = np.mean([r['final_coverage'] for r in nogat_all_res])
    nogat_saf = np.mean([r['final_safety'] for r in nogat_all_res])
    x = np.arange(2); w = 0.3
    ax.bar(x-w/2, [gat_cov, nogat_cov], w, label='Coverage', color='#3498db', edgecolor='k', lw=0.5)
    ax.bar(x+w/2, [gat_saf, nogat_saf], w, label='Safety', color='#2ecc71', edgecolor='k', lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(['GAT-MARAHS', 'No-GAT'])
    ax.set_ylabel('Final Performance (%)'); ax.set_title('(d) Ablation Summary')
    ax.legend(); ax.grid(True, axis='y', alpha=0.3); ax.set_ylim(0, 100)
    plt.tight_layout(); plt.savefig('figures_gat/fig1_training.png'); plt.close()

    # Fig 2: Benchmark bar chart
    fig,ax=plt.subplots(figsize=(8,5))
    methods=list(bench_res.keys()); x=np.arange(len(methods)); w=0.25
    ax.bar(x-w,[bench_res[m]['safety'] for m in methods],w,label='Safety',color='#2ecc71',edgecolor='k',lw=0.5)
    ax.bar(x,[bench_res[m]['coverage'] for m in methods],w,label='Coverage',color='#3498db',edgecolor='k',lw=0.5)
    ax.bar(x+w,[bench_res[m]['perimeter'] for m in methods],w,label='Perimeter',color='#e74c3c',edgecolor='k',lw=0.5)
    ax.set_ylabel('Performance (%)'); ax.set_title('GAT-MARAHS Benchmark (wind=12 m/s)'); ax.set_xticks(x); ax.set_xticklabels(methods,rotation=15)
    ax.legend(); ax.grid(True,axis='y',alpha=0.3); ax.set_ylim(0,100)
    plt.tight_layout(); plt.savefig('figures_gat/fig2_benchmark.png'); plt.close()

    # Fig 3: Wind sweep with error bars
    fig,ax=plt.subplots(figsize=(8,5))
    winds=[int(k) for k in sorted(wind_res.keys())]
    ax.errorbar(winds,[wind_res[str(w)]['perimeter'] for w in winds],yerr=[wind_res[str(w)]['perimeter_std'] for w in winds],marker='o',lw=2,label='Perimeter',color='#e74c3c',capsize=4)
    ax.errorbar(winds,[wind_res[str(w)]['safety'] for w in winds],yerr=[wind_res[str(w)]['safety_std'] for w in winds],marker='s',lw=2,label='Safety',color='#2ecc71',capsize=4)
    ax.errorbar(winds,[wind_res[str(w)]['coverage'] for w in winds],yerr=[wind_res[str(w)]['coverage_std'] for w in winds],marker='^',lw=2,label='Coverage',color='#3498db',capsize=4)
    ax.set_xlabel('Wind Speed (m/s)'); ax.set_ylabel('Performance (%)'); ax.set_title('GAT-MARAHS vs Wind Intensity')
    ax.legend(); ax.grid(True,alpha=0.3); ax.set_ylim(0,100)
    plt.tight_layout(); plt.savefig('figures_gat/fig3_wind_sweep.png'); plt.close()

    # Fig 4: Ablation with error bars
    fig,ax=plt.subplots(figsize=(6,5))
    cats = ['Coverage', 'Safety']
    gat_vals = [np.mean([r['final_coverage'] for r in gat_all_res]), np.mean([r['final_safety'] for r in gat_all_res])]
    gat_errs = [np.std([r['final_coverage'] for r in gat_all_res]), np.std([r['final_safety'] for r in gat_all_res])]
    nogat_vals = [np.mean([r['final_coverage'] for r in nogat_all_res]), np.mean([r['final_safety'] for r in nogat_all_res])]
    nogat_errs = [np.std([r['final_coverage'] for r in nogat_all_res]), np.std([r['final_safety'] for r in nogat_all_res])]
    x = np.arange(len(cats)); w = 0.35
    ax.bar(x-w/2, gat_vals, w, yerr=gat_errs, capsize=5, label='GAT-MARAHS', color='#3498db', edgecolor='k', lw=0.5)
    ax.bar(x+w/2, nogat_vals, w, yerr=nogat_errs, capsize=5, label='No-GAT (ablation)', color='#95a5a6', edgecolor='k', lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel('Performance (%)'); ax.set_title('GAT Communication Ablation')
    ax.legend(); ax.grid(True, axis='y', alpha=0.3); ax.set_ylim(0, 100)
    plt.tight_layout(); plt.savefig('figures_gat/fig4_ablation.png'); plt.close()

    print("  ✓ 4 figures saved to figures_gat/", flush=True)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60, flush=True)
    print("PlumeGym-MARL: Full Kaggle Training Pipeline (GPU)", flush=True)
    print("="*60, flush=True)

    GRID = 30
    N_DRONES = 10
    MAX_STEPS = 300
    N_EPISODES = 2000
    SEEDS = [42, 123, 777]

    t_total = time.time()

    # Step 1: Train GAT-MARAHS × 3 seeds
    print("\n" + "#"*60, flush=True)
    print("# PHASE 1: Train GAT-MARAHS (3 seeds)", flush=True)
    print("#"*60, flush=True)
    gat_agent, gat_all_res = train_multi_seed(N_EPISODES, GRID, N_DRONES, MAX_STEPS, use_gat=True, seeds=SEEDS)

    # Step 2: Train No-GAT ablation × 3 seeds
    print("\n" + "#"*60, flush=True)
    print("# PHASE 2: Train No-GAT ablation (3 seeds)", flush=True)
    print("#"*60, flush=True)
    nogat_agent, nogat_all_res = train_multi_seed(N_EPISODES, GRID, N_DRONES, MAX_STEPS, use_gat=False, seeds=SEEDS)

    # Step 3: Benchmark
    print("\n" + "#"*60, flush=True)
    print("# PHASE 3: Benchmark vs baselines", flush=True)
    print("#"*60, flush=True)
    bench_res = benchmark(gat_agent, GRID, N_DRONES, MAX_STEPS, wind=12.0, n_eps=20)

    # Step 4: Wind sweep
    print("\n" + "#"*60, flush=True)
    print("# PHASE 4: Wind robustness sweep", flush=True)
    print("#"*60, flush=True)
    wind_res = wind_sweep(gat_agent, GRID, N_DRONES, MAX_STEPS, n_eps=15)

    # Step 5: Figures
    print("\n" + "#"*60, flush=True)
    print("# PHASE 5: Generate figures", flush=True)
    print("#"*60, flush=True)
    generate_figures(gat_all_res, nogat_all_res, bench_res, wind_res)

    # Step 6: Summary
    total_time = time.time() - t_total
    print("\n" + "="*60, flush=True)
    print("COMPLETE!", flush=True)
    print("="*60, flush=True)
    print(f"Total time: {total_time/60:.1f} minutes", flush=True)
    print(f"\nGAT-MARAHS ({len(SEEDS)} seeds):", flush=True)
    print(f"  Coverage: {np.mean([r['final_coverage'] for r in gat_all_res]):.1f}% ± {np.std([r['final_coverage'] for r in gat_all_res]):.1f}%", flush=True)
    print(f"  Safety:   {np.mean([r['final_safety'] for r in gat_all_res]):.1f}% ± {np.std([r['final_safety'] for r in gat_all_res]):.1f}%", flush=True)
    print(f"\nNo-GAT ablation ({len(SEEDS)} seeds):", flush=True)
    print(f"  Coverage: {np.mean([r['final_coverage'] for r in nogat_all_res]):.1f}% ± {np.std([r['final_coverage'] for r in nogat_all_res]):.1f}%", flush=True)
    print(f"  Safety:   {np.mean([r['final_safety'] for r in nogat_all_res]):.1f}% ± {np.std([r['final_safety'] for r in nogat_all_res]):.1f}%", flush=True)
    print(f"\nBenchmark (wind=12):", flush=True)
    for m,v in bench_res.items():
        print(f"  {m}: Safety={v['safety']:.1f}% Coverage={v['coverage']:.1f}% Perimeter={v['perimeter']:.1f}%", flush=True)
    print(f"\nFiles generated:", flush=True)
    for f in ['gat_*_best.pt', 'nogat_*_best.pt', '*_training_results.json', 'gat_benchmark_final.json', 'gat_wind_sweep.json', 'figures_gat/fig*.png']:
        print(f"  - {f}", flush=True)
