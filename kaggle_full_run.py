#!/usr/bin/env python3
"""
=================================================================
PlumeGym-MARL: Full Training Pipeline for Kaggle
=================================================================
Upload this file to Kaggle as a Notebook. Set CPU runtime.
It will train GAT-MARAHS for 3000 episodes, run benchmarks,
generate publication figures, and save everything.

Runtime: ~25-35 minutes on Kaggle CPU
Memory: ~2GB (well within 16GB limit)

How to use:
1. Go to kaggle.com/code
2. Create new notebook
3. Upload this file as the notebook
4. Add these files as Dataset or paste the code cells
5. Set runtime to CPU
6. Click Run All

Output files:
- gat_marahs_best.pt (trained model)
- gat_benchmark_final.json (benchmark numbers)
- gat_wind_sweep.json (wind sweep data)
- figures_gat/fig1_training.png
- figures_gat/fig2_benchmark.png
- figures_gat/fig3_wind_sweep.png
- figures_gat/fig4_scaling.png
=================================================================
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import os

# ═══════════════════════════════════════════════════════════════
# SECTION 1: ENVIRONMENT (copy from paper_ready_train.py)
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
            # Other drones
            grid = np.zeros((self.obs_size, self.obs_size), dtype=np.float32)
            for j in range(self.n_drones):
                if j != i and self.drones[j]['alive']:
                    jx = int(self.drones[j]['pos'][0]) - cx + r
                    jy = int(self.drones[j]['pos'][1]) - cy + r
                    if 0 <= jx < self.obs_size and 0 <= jy < self.obs_size: grid[jx, jy] = 1.0
            local[ch_idx*ch_size:(ch_idx+1)*ch_size] = grid.ravel(); ch_idx += 1
            # Fire distance
            grid = np.zeros((self.obs_size, self.obs_size), dtype=np.float32)
            if self._fire_dist_cache is not None:
                grid[:h, :w] = np.minimum(self._fire_dist_cache[x_min:x_max, y_min:y_max] / 10.0, 1.0)
            local[ch_idx*ch_size:(ch_idx+1)*ch_size] = grid.ravel(); ch_idx += 1
            # Visited
            grid = np.zeros((self.obs_size, self.obs_size), dtype=np.float32)
            grid[:h, :w] = self._visited_grid[x_min:x_max, y_min:y_max]
            local[ch_idx*ch_size:(ch_idx+1)*ch_size] = grid.ravel(); ch_idx += 1
            # Global features
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
# SECTION 2: GAT + PPO (copy from gat_communication.py + train_gat_fast.py)
# ═══════════════════════════════════════════════════════════════

device = torch.device("cpu")

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
        adj = torch.zeros(K, K, dtype=torch.bool)
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
    def __init__(self, obs_dim=656, act_dim=5, gat_hidden=128, gat_out=64, n_heads=4, comm_range=15.0, lr=3e-4):
        self.gat = GATCommunication(obs_dim, gat_hidden, gat_out, n_heads, comm_range)
        self.policy = PPONetwork(obs_dim + gat_out, act_dim)
        self.optimizer = torch.optim.Adam(list(self.gat.parameters()) + list(self.policy.parameters()), lr=lr, eps=1e-5)
        self._traj = []
    def select_actions(self, obs, positions, alive_mask):
        obs_t = torch.tensor(obs, dtype=torch.float32)
        pos_t = torch.tensor(positions, dtype=torch.float32)
        enhanced = self.gat(obs_t, pos_t, alive_mask)
        with torch.no_grad():
            logits, values = self.policy(enhanced)
            probs = torch.distributions.Categorical(logits=logits)
            actions = probs.sample()
            log_probs = probs.log_prob(actions)
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy(), enhanced.detach()
    def store(self, enhanced_obs, actions, rewards, dones, log_probs, values):
        for i in range(len(actions)):
            self._traj.append({'obs': enhanced_obs[i], 'action': actions[i], 'reward': rewards[i], 'done': dones[i], 'log_prob': log_probs[i], 'value': values[i]})
    def update(self, gamma=0.99, gae_lambda=0.95, n_epochs=4, batch_size=256, clip_eps=0.2, entropy_coef=0.02):
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
        all_obs = torch.stack([t['obs'].squeeze(0) for t in self._traj])
        all_actions = torch.tensor([t['action'] for t in self._traj], dtype=torch.long)
        all_old_lp = torch.tensor([t['log_prob'] for t in self._traj], dtype=torch.float32)
        all_adv = torch.tensor(advantages, dtype=torch.float32)
        all_ret = torch.tensor(returns, dtype=torch.float32)
        all_adv = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-8)
        total_loss, count = 0.0, 0
        for _ in range(n_epochs):
            perm = torch.randperm(n)
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
        return total_loss / max(1, count)
    def save(self, path):
        torch.save({'gat': self.gat.state_dict(), 'policy': self.policy.state_dict()}, path)
    def load(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.gat.load_state_dict(ckpt['gat'])
        self.policy.load_state_dict(ckpt['policy'])


# ═══════════════════════════════════════════════════════════════
# SECTION 3: TRAINING
# ═══════════════════════════════════════════════════════════════

def compute_reward(drone, prev_visited, fire_dist, crashed, step, max_steps):
    if crashed: return -15.0
    reward = 0.05
    new_cells = sum(1 for c in drone['visited'] if c not in prev_visited)
    reward += 25.0 * new_cells
    if fire_dist < 3.0: reward += 8.0 * (1.0 - fire_dist / 3.0)
    if step >= max_steps - 1: reward += 5.0
    return reward

def train(n_episodes=3000, grid=30, n_drones=10, max_steps=300):
    print("="*60)
    print(f"GAT-MARAHS Training | {n_episodes} eps | {n_drones} drones | {grid}x{grid}")
    print("="*60)
    
    env = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps, wind_speed=0)
    agent = FastGATPPO(obs_dim=env.obs_dim, act_dim=env.act_dim)
    
    stages = [(0,500),(5,1000),(10,1500),(15,2000),(20,2500),(25,3000)]
    rewards_h, coverage_h, safety_h = [], [], []
    best_r = -float('inf')
    t0 = time.time()
    
    for ep in range(n_episodes):
        wind = 0
        for w, end in stages:
            if ep < end: wind = w; break
        env.base_wind = wind
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
                shaped[i] = compute_reward(env.drones[i], prev_visited[i], fd, dones[i], step, max_steps)
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
            print(f"Ep {ep+1:5d}/{n_episodes} | R: {avg_r:7.1f} | Cov: {avg_cov:5.1f}% | Safe: {avg_saf:4.0f}% | wind={wind} | {elapsed:.0f}s")
            if avg_r > best_r: best_r = avg_r; agent.save('gat_marahs_best.pt')
        if (ep+1) % 500 == 0: agent.save(f'gat_ep{ep+1}.pt')
    
    agent.save('gat_marahs_final.pt')
    results = {'n_episodes': n_episodes, 'final_reward': float(np.mean(rewards_h[-100:])), 'final_coverage': float(np.mean(coverage_h[-100:])), 'final_safety': float(np.mean(safety_h[-100:])), 'rewards': [float(x) for x in rewards_h], 'coverages': [float(x) for x in coverage_h], 'safety': [float(x) for x in safety_h]}
    with open('gat_training_results.json', 'w') as f: json.dump(results, f, indent=2)
    print(f"\nDone in {time.time()-t0:.0f}s | Reward: {results['final_reward']:.1f} | Coverage: {results['final_coverage']:.1f}% | Safety: {results['final_safety']:.0f}%")
    return agent, results


# ═══════════════════════════════════════════════════════════════
# SECTION 4: BENCHMARK
# ═══════════════════════════════════════════════════════════════

def benchmark(agent, grid=30, n_drones=10, max_steps=300, wind=12.0, n_eps=20):
    print(f"\n{'='*60}\nBENCHMARK | wind={wind} | {n_drones} drones | {n_eps} eps\n{'='*60}")
    results = {}
    
    for method_name, method_fn in [
        ('GAT-MARAHS', lambda env: _run_gat(agent, env, n_drones)),
        ('Random', lambda env: _run_random(env, n_drones)),
        ('Greedy', lambda env: _run_greedy(env, n_drones)),
        ('PID', lambda env: _run_pid(env, n_drones)),
    ]:
        s, c, p, a = [], [], [], []
        for _ in range(n_eps):
            env = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps, wind_speed=wind)
            obs = env.reset()
            for step in range(max_steps):
                alive = np.array([env.drones[i]['alive'] for i in range(n_drones)])
                if not alive.any(): break
                actions = method_fn(env)
                obs, _, dones, _ = env.step(np.array(actions, dtype=np.int32))
                if all(dones): break
            ac = sum(1 for i in range(n_drones) if env.drones[i]['alive'])
            s.append(ac/n_drones*100); c.append(len(env.total_cells_explored)/(grid*grid)*100)
            fc = np.argwhere(env.fire > 0.2)
            pc = set()
            for fx,fy in fc:
                for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx,ny=fx+dx,fy+dy
                    if 0<=nx<grid and 0<=ny<grid and env.fire[nx,ny]<0.1: pc.add((nx,ny))
            vis=set()
            for i in range(n_drones): vis.update(env.drones[i].get('visited',set()))
            p.append(len(pc&vis)/max(1,len(pc))*100); a.append(ac)
        results[method_name] = {'safety':np.mean(s),'coverage':np.mean(c),'perimeter':np.mean(p),'alive':np.mean(a)}
    
    print(f"\n{'Method':<18s} {'Safety':>8s} {'Coverage':>10s} {'Perimeter':>10s} {'Alive':>8s}")
    print("-"*58)
    for m,v in results.items(): print(f"{m:<18s} {v['safety']:7.1f}%  {v['coverage']:8.1f}%  {v['perimeter']:8.1f}%  {v['alive']:6.1f}/10")
    print("="*58)
    
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
        d = env.drones[i]; ix,iy = int(d['pos'][0]),int(d['pos'][1])
        fc = np.argwhere(env.fire > 0.2)
        if len(fc)>0:
            fcx,fcy = float(np.mean(fc[:,0])),float(np.mean(fc[:,1]))
            ddx,ddy = fcx-d['pos'][0],fcy-d['pos'][1]
            if abs(ddx)>abs(ddy): acts[i]=3 if ddx>0 else 4
            else: acts[i]=1 if ddy>0 else 2
    return acts


# ═══════════════════════════════════════════════════════════════
# SECTION 5: WIND SWEEP
# ═══════════════════════════════════════════════════════════════

def wind_sweep(agent, grid=30, n_drones=10, max_steps=300, n_eps=15):
    print(f"\n{'='*60}\nWIND SWEEP | {n_drones} drones | {n_eps} episodes each\n{'='*60}")
    sweep = {}
    for wind in [5, 10, 15, 20, 25]:
        s,c,p = [],[],[]
        for _ in range(n_eps):
            env = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps, wind_speed=wind)
            obs = env.reset()
            for step in range(max_steps):
                am = np.array([env.drones[i]['alive'] for i in range(n_drones)])
                pos = np.array([env.drones[i]['pos'] for i in range(n_drones)], dtype=np.float32)
                if not am.any(): break
                acts,_,_,_ = agent.select_actions(obs,pos,am)
                obs,_,dones,_ = env.step(np.array(acts,dtype=np.int32))
                if all(dones): break
            ac = sum(1 for i in range(n_drones) if env.drones[i]['alive'])
            s.append(ac/n_drones*100); c.append(len(env.total_cells_explored)/(grid*grid)*100)
            fc=np.argwhere(env.fire>0.2); pc=set()
            for fx,fy in fc:
                for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx,ny=fx+dx,fy+dy
                    if 0<=nx<grid and 0<=ny<grid and env.fire[nx,ny]<0.1: pc.add((nx,ny))
            vis=set()
            for i in range(n_drones): vis.update(env.drones[i].get('visited',set()))
            p.append(len(pc&vis)/max(1,len(pc))*100)
        sweep[str(wind)] = {'safety':np.mean(s),'coverage':np.mean(c),'perimeter':np.mean(p),
                           'safety_std':float(np.std(s)),'coverage_std':float(np.std(c)),'perimeter_std':float(np.std(p))}
        print(f"  Wind={wind:2d} m/s | Safety={np.mean(s):5.1f}% | Coverage={np.mean(c):5.1f}% | Perimeter={np.mean(p):5.1f}%")
    with open('gat_wind_sweep.json','w') as f: json.dump(sweep,f,indent=2)
    return sweep


# ═══════════════════════════════════════════════════════════════
# SECTION 6: FIGURES
# ═══════════════════════════════════════════════════════════════

def generate_figures(train_res, bench_res, wind_res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    os.makedirs('figures_gat', exist_ok=True)
    rc = plt.rcParams; rc['font.family']='serif'; rc['font.size']=11; rc['figure.dpi']=150; rc['savefig.bbox']='tight'
    
    # Fig 1: Training curves
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4))
    cov_smooth = np.convolve(train_res['coverages'], np.ones(20)/20, mode='valid')
    saf_smooth = np.convolve(train_res['safety'], np.ones(20)/20, mode='valid')
    ax1.plot(cov_smooth,'b-',lw=1.5); ax1.set_xlabel('Episode'); ax1.set_ylabel('Coverage (%)'); ax1.set_title('(a) Exploration'); ax1.grid(True,alpha=0.3)
    ax2.plot(saf_smooth,'r-',lw=1.5); ax2.set_xlabel('Episode'); ax2.set_ylabel('Safety (%)'); ax2.set_title('(b) Safety'); ax2.set_ylim(0,105); ax2.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig('figures_gat/fig1_training.png'); plt.close()
    
    # Fig 2: Benchmark
    fig,ax=plt.subplots(figsize=(8,5))
    methods=list(bench_res.keys()); x=np.arange(len(methods)); w=0.25
    ax.bar(x-w,[bench_res[m]['safety'] for m in methods],w,label='Safety',color='#2ecc71',edgecolor='k',lw=0.5)
    ax.bar(x,[bench_res[m]['coverage'] for m in methods],w,label='Coverage',color='#3498db',edgecolor='k',lw=0.5)
    ax.bar(x+w,[bench_res[m]['perimeter'] for m in methods],w,label='Perimeter',color='#e74c3c',edgecolor='k',lw=0.5)
    ax.set_ylabel('Performance (%)'); ax.set_title('GAT-MARAHS Benchmark'); ax.set_xticks(x); ax.set_xticklabels(methods,rotation=15)
    ax.legend(); ax.grid(True,axis='y',alpha=0.3); ax.set_ylim(0,100)
    plt.tight_layout(); plt.savefig('figures_gat/fig2_benchmark.png'); plt.close()
    
    # Fig 3: Wind sweep
    fig,ax=plt.subplots(figsize=(8,5))
    winds=[int(k) for k in sorted(wind_res.keys())]
    ax.errorbar(winds,[wind_res[str(w)]['perimeter'] for w in winds],yerr=[wind_res[str(w)]['perimeter_std'] for w in winds],marker='o',lw=2,label='Perimeter',color='#e74c3c',capsize=4)
    ax.errorbar(winds,[wind_res[str(w)]['safety'] for w in winds],yerr=[wind_res[str(w)]['safety_std'] for w in winds],marker='s',lw=2,label='Safety',color='#2ecc71',capsize=4)
    ax.errorbar(winds,[wind_res[str(w)]['coverage'] for w in winds],yerr=[wind_res[str(w)]['coverage_std'] for w in winds],marker='^',lw=2,label='Coverage',color='#3498db',capsize=4)
    ax.set_xlabel('Wind Speed (m/s)'); ax.set_ylabel('Performance (%)'); ax.set_title('GAT-MARAHS vs Wind Intensity')
    ax.legend(); ax.grid(True,alpha=0.3); ax.set_ylim(0,100)
    plt.tight_layout(); plt.savefig('figures_gat/fig3_wind_sweep.png'); plt.close()
    
    print("  ✓ 3 figures saved to figures_gat/")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("PlumeGym-MARL: Full Kaggle Training Pipeline")
    print("="*60)
    
    GRID = 30
    N_DRONES = 10
    MAX_STEPS = 300
    N_EPISODES = 3000
    
    # Step 1: Train
    agent, train_res = train(N_EPISODES, GRID, N_DRONES, MAX_STEPS)
    
    # Step 2: Benchmark
    bench_res = benchmark(agent, GRID, N_DRONES, MAX_STEPS, wind=12.0, n_eps=20)
    
    # Step 3: Wind sweep
    wind_res = wind_sweep(agent, GRID, N_DRONES, MAX_STEPS, n_eps=15)
    
    # Step 4: Figures
    generate_figures(train_res, bench_res, wind_res)
    
    # Step 5: Summary
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"Files generated:")
    print(f"  - gat_marahs_best.pt (trained model)")
    print(f"  - gat_training_results.json")
    print(f"  - gat_benchmark_final.json")
    print(f"  - gat_wind_sweep.json")
    print(f"  - figures_gat/fig1_training.png")
    print(f"  - figures_gat/fig2_benchmark.png")
    print(f"  - figures_gat/fig3_wind_sweep.png")
    print(f"\nKey results:")
    print(f"  Training reward: {train_res['final_reward']:.1f}")
    print(f"  Training coverage: {train_res['final_coverage']:.1f}%")
    print(f"  Benchmark (wind=12):")
    for m,v in bench_res.items():
        print(f"    {m}: Safety={v['safety']:.1f}% Coverage={v['coverage']:.1f}% Perimeter={v['perimeter']:.1f}%")
