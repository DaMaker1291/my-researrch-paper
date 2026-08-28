#!/usr/bin/env python3
"""
Paper-Ready Wildfire Drone Training Pipeline v3
================================================
Key insight: PPO fails to explore because:
1. 575-dim obs is too large for from-scratch MLP
2. Multi-agent interleaved GAE breaks temporal credit assignment
3. Entropy collapse → hover policy

v3 approach: Keep the environment, replace PPO with a simpler 
 reward-to-go approach, much stronger entropy, and smaller obs.
"""
import numpy as np
import json, os, time

# ═══════════════════════════════════════════════════════════════
# 1. ENVIRONMENT (same as v2, keep it)
# ═══════════════════════════════════════════════════════════════

class WildfireEnv:
    def __init__(self, grid=30, n_drones=10, max_steps=300,
                 wind_speed=12.0):
        self.grid = grid
        self.n_drones = n_drones
        self.max_steps = max_steps
        self.base_wind = wind_speed

        yy, xx = np.meshgrid(np.arange(grid), np.arange(grid), indexing='ij')
        self.xx = xx.astype(np.float32)
        self.yy = yy.astype(np.float32)

        self.obs_r = 4
        self.obs_size = 2 * self.obs_r + 1
        self.obs_channels = 8  # added visited channel
        self.local_obs_dim = self.obs_channels * self.obs_size * self.obs_size
        self.global_obs_dim = 8
        self.obs_dim = self.local_obs_dim + self.global_obs_dim
        self.act_dim = 5

        self.action_deltas = np.array([
            [0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]
        ], dtype=np.float32)

        self.wind_coupling = 0.02
        self.thermal_cap = 8.0
        self.drone_speed = 2.0
        self.fire_crash_threshold = 0.3
        self.thermal_crash = 15.0
        self.wind_crash = 35.0
        self.boundary_margin = 1.0
        self._fire_dist_cache = None

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        self.fire = np.zeros((self.grid, self.grid), dtype=np.float32)
        cx = rng.integers(8, self.grid - 8)
        cy = rng.integers(8, self.grid - 8)
        r = rng.integers(2, 3)
        mask = (self.xx - cx)**2 + (self.yy - cy)**2 < r**2
        self.fire[mask] = rng.uniform(0.5, 0.8, size=mask.sum())

        self.fuel = np.ones((self.grid, self.grid), dtype=np.float32)
        for _ in range(rng.integers(2, 4)):
            px, py = rng.integers(0, self.grid, size=2)
            pr = rng.integers(2, 5)
            pm = (self.xx - px)**2 + (self.yy - py)**2 < pr**2
            self.fuel[pm] = rng.uniform(0.1, 0.4)

        self.wind_phase = 0.0
        self._update_wind(rng)
        self.thermal = np.zeros((self.grid, self.grid), dtype=np.float32)
        self._update_thermal()
        self._update_fire_dist_cache()

        self.drones = []
        angle_off = rng.uniform(0, 2 * np.pi)
        fire_cx, fire_cy = float(cx), float(cy)
        for i in range(self.n_drones):
            angle = angle_off + 2 * np.pi * i / self.n_drones
            dist = self.grid * 0.40 + rng.uniform(-2, 2)
            sx = np.clip(fire_cx + dist * np.cos(angle), 3, self.grid - 4)
            sy = np.clip(fire_cy + dist * np.sin(angle), 3, self.grid - 4)
            self.drones.append({
                'pos': np.array([sx, sy], dtype=np.float32),
                'vel': np.array([0.0, 0.0], dtype=np.float32),
                'battery': 500, 'alive': True, 'visited': set(),
                'crashes': 0, 'alive_steps': 0, 'prev_action': 0,
            })

        self.step_count = 0
        self.total_perimeter_cells = 0
        self.visited_perimeter = set()
        self.fire_cx, self.fire_cy = fire_cx, fire_cy
        self.total_cells_explored = set()
        self._visited_grid = np.zeros((self.grid, self.grid), dtype=bool)
        return self._get_obs()

    def _update_wind(self, rng):
        self.wind_phase += 0.05
        turbulence = max(0.3, self.base_wind * 0.25)
        speed = self.base_wind + turbulence * np.sin(self.wind_phase * 1.3)
        direction = 0.0 + 0.4 * np.sin(self.wind_phase * 0.7)
        self.wind_x = speed * np.cos(direction) * np.ones((self.grid, self.grid), dtype=np.float32)
        self.wind_y = speed * np.sin(direction) * np.ones((self.grid, self.grid), dtype=np.float32)
        for k in range(3):
            freq = 0.08 * (2**k)
            amp = turbulence / (2**k)
            self.wind_x += amp * np.sin(self.xx * freq + self.wind_phase * (k+1))
            self.wind_y += amp * np.cos(self.yy * freq + self.wind_phase * (k+1) * 0.7)
        noise_level = max(0.1, self.base_wind * 0.02)
        self.wind_x += rng.normal(0, noise_level, self.wind_x.shape)
        self.wind_y += rng.normal(0, noise_level, self.wind_y.shape)

    def _update_thermal(self):
        self.thermal[:] = 0
        fire_cells = np.argwhere(self.fire > 0.2)
        for cell in fire_cells:
            intensity = self.fire[cell[0], cell[1]]
            r2 = (self.xx - cell[0])**2 + (self.yy - cell[1])**2
            plume = 5.0 * intensity * np.exp(-r2 / 12.0)
            self.thermal += np.minimum(plume, 2.5)
        self.thermal = np.minimum(self.thermal, self.thermal_cap)

    def _update_fire_dist_cache(self):
        from scipy.ndimage import distance_transform_edt
        self._fire_dist_cache = distance_transform_edt(
            ~(self.fire > 0.1)).astype(np.float32)

    def _spread_fire(self, rng):
        new_fire = self.fire.copy()
        fire_mask = self.fire > 0.1
        from scipy.signal import convolve2d
        kernel = np.array([[0.05, 0.1, 0.05], [0.1, 0, 0.1], [0.05, 0.1, 0.05]])
        neighbors = convolve2d(fire_mask.astype(float), kernel, mode='same', boundary='fill')
        wind_speed = np.sqrt(self.wind_x**2 + self.wind_y**2)
        spread = 0.012 * (1 + 1.0 * wind_speed / 10.0) * self.fuel * neighbors
        noise = rng.random((self.grid, self.grid))
        spreading = (noise < spread) & (~fire_mask) & (self.fuel > 0.05)
        new_fire = np.clip(new_fire + 0.015 * fire_mask + 0.25 * spreading, 0, 1.0)
        self.fuel = np.maximum(0, self.fuel - 0.006 * fire_mask)
        fire_cells = np.argwhere(fire_mask)
        for cell in fire_cells[::4]:
            if rng.random() < 0.008:
                a = rng.uniform(0, 2 * np.pi)
                d = rng.integers(1, 3)
                sx, sy = int(cell[0] + d * np.cos(a)), int(cell[1] + d * np.sin(a))
                if 0 <= sx < self.grid and 0 <= sy < self.grid:
                    if self.fuel[sx, sy] > 0.05 and new_fire[sx, sy] < 0.1:
                        new_fire[sx, sy] = 0.4
        self.fire = new_fire

    def _get_perimeter(self):
        from scipy.ndimage import convolve
        fire_mask = self.fire > 0.1
        kernel = np.ones((3, 3))
        neighbors = convolve(fire_mask.astype(float), kernel, mode='constant')
        return set(zip(*np.where((fire_mask) & (neighbors < 9))))

    def _dist_to_fire(self, pos):
        ix = int(np.clip(np.round(pos[0]), 0, self.grid - 1))
        iy = int(np.clip(np.round(pos[1]), 0, self.grid - 1))
        return float(self._fire_dist_cache[iy, ix])

    def _get_frontier_direction(self, drone_pos):
        unexplored = np.argwhere(~self._visited_grid)
        if len(unexplored) == 0:
            return np.array([0.0, 0.0], dtype=np.float32)
        dists = np.sum((unexplored - drone_pos)**2, axis=1)
        nearby = unexplored[dists < 225]
        if len(nearby) == 0:
            nearby = unexplored
        centroid = np.mean(nearby, axis=0)
        direction = centroid - drone_pos
        norm = np.linalg.norm(direction) + 1e-8
        return (direction / norm).astype(np.float32)

    def _get_obs(self):
        r = self.obs_r
        obs = np.zeros((self.n_drones, self.obs_dim), dtype=np.float32)
        ch_size = self.obs_size * self.obs_size

        for i in range(self.n_drones):
            if not self.drones[i]['alive']:
                continue
            cx, cy = int(self.drones[i]['pos'][0]), int(self.drones[i]['pos'][1])
            x_min, x_max = max(0, cx - r), min(self.grid, cx + r + 1)
            y_min, y_max = max(0, cy - r), min(self.grid, cy + r + 1)
            h, w = x_max - x_min, y_max - y_min

            local = np.zeros(self.local_obs_dim, dtype=np.float32)
            ch_idx = 0

            # Ch0: fire
            grid = np.zeros((self.obs_size, self.obs_size), dtype=np.float32)
            grid[:h, :w] = self.fire[x_min:x_max, y_min:y_max]
            local[ch_idx*ch_size:(ch_idx+1)*ch_size] = grid.ravel(); ch_idx += 1

            # Ch1-2: wind
            for wind_arr in [self.wind_x, self.wind_y]:
                grid = np.zeros((self.obs_size, self.obs_size), dtype=np.float32)
                grid[:h, :w] = wind_arr[x_min:x_max, y_min:y_max] / 30.0
                local[ch_idx*ch_size:(ch_idx+1)*ch_size] = grid.ravel(); ch_idx += 1

            # Ch3: fuel
            grid = np.zeros((self.obs_size, self.obs_size), dtype=np.float32)
            grid[:h, :w] = self.fuel[x_min:x_max, y_min:y_max]
            local[ch_idx*ch_size:(ch_idx+1)*ch_size] = grid.ravel(); ch_idx += 1

            # Ch4: thermal
            grid = np.zeros((self.obs_size, self.obs_size), dtype=np.float32)
            grid[:h, :w] = self.thermal[x_min:x_max, y_min:y_max] / self.thermal_cap
            local[ch_idx*ch_size:(ch_idx+1)*ch_size] = grid.ravel(); ch_idx += 1

            # Ch5: other drones
            grid = np.zeros((self.obs_size, self.obs_size), dtype=np.float32)
            for j in range(self.n_drones):
                if j != i and self.drones[j]['alive']:
                    jx = int(self.drones[j]['pos'][0]) - cx + r
                    jy = int(self.drones[j]['pos'][1]) - cy + r
                    if 0 <= jx < self.obs_size and 0 <= jy < self.obs_size:
                        grid[jx, jy] = 1.0
            local[ch_idx*ch_size:(ch_idx+1)*ch_size] = grid.ravel(); ch_idx += 1

            # Ch6: fire distance
            grid = np.zeros((self.obs_size, self.obs_size), dtype=np.float32)
            if self._fire_dist_cache is not None:
                grid[:h, :w] = np.minimum(
                    self._fire_dist_cache[x_min:x_max, y_min:y_max] / 10.0, 1.0)
            local[ch_idx*ch_size:(ch_idx+1)*ch_size] = grid.ravel(); ch_idx += 1

            # Ch7: VISITED (KEY: lets drone see which cells it already explored)
            grid = np.zeros((self.obs_size, self.obs_size), dtype=np.float32)
            grid[:h, :w] = self._visited_grid[x_min:x_max, y_min:y_max].astype(np.float32)
            local[ch_idx*ch_size:(ch_idx+1)*ch_size] = grid.ravel(); ch_idx += 1

            # Global features
            fire_cells = np.argwhere(self.fire > 0.2)
            if len(fire_cells) > 0:
                fcx = float(np.mean(fire_cells[:, 0]))
                fcy = float(np.mean(fire_cells[:, 1]))
                fr = float(np.sqrt(len(fire_cells))) / self.grid
            else:
                fcx, fcy = self.grid / 2, self.grid / 2
                fr = 0.1

            dx = (fcx - self.drones[i]['pos'][0]) / self.grid
            dy = (fcy - self.drones[i]['pos'][1]) / self.grid
            ix = int(np.clip(self.drones[i]['pos'][0], 0, self.grid - 1))
            iy = int(np.clip(self.drones[i]['pos'][1], 0, self.grid - 1))
            wind_dir = float(np.arctan2(self.wind_y[iy, ix], self.wind_x[iy, ix])) / np.pi
            coverage = len(self.total_cells_explored) / (self.grid * self.grid)
            frontier = self._get_frontier_direction(self.drones[i]['pos'])

            global_f = np.array([
                fcx / self.grid, fcy / self.grid, fr,
                dx, dy, wind_dir, coverage, np.linalg.norm(frontier)
            ], dtype=np.float32)
            obs[i] = np.concatenate([local, global_f])
        return obs

    def _check_crash(self, pos, thermal_val, wind_spd):
        ix = int(np.clip(np.round(pos[0]), 0, self.grid - 1))
        iy = int(np.clip(np.round(pos[1]), 0, self.grid - 1))
        if self.fire[iy, ix] > self.fire_crash_threshold:
            return True, 'fire_cell'
        if wind_spd > 10.0:
            fire_dist = self._fire_dist_cache[iy, ix]
            buffer = max(0.3, (wind_spd - 10.0) / 10.0)
            if fire_dist < buffer:
                return True, 'fire_edge'
        if thermal_val > self.thermal_crash:
            return True, 'thermal'
        if (pos[0] < self.boundary_margin or pos[0] > self.grid - self.boundary_margin or
            pos[1] < self.boundary_margin or pos[1] > self.grid - self.boundary_margin):
            return True, 'boundary'
        if wind_spd > self.wind_crash:
            return True, 'wind'
        return False, 'safe'

    def step(self, actions):
        self.step_count += 1
        rng = np.random.default_rng()
        if self.step_count % 10 == 0:
            self._update_wind(rng)
            self._update_thermal()

        rewards = np.zeros(self.n_drones, dtype=np.float32)
        dones = np.zeros(self.n_drones, dtype=bool)
        infos = [{} for _ in range(self.n_drones)]
        perimeter = self._get_perimeter()
        self.total_perimeter_cells = max(1, len(perimeter))

        for i in range(self.n_drones):
            d = self.drones[i]
            if not d['alive']:
                continue
            d['battery'] -= 1
            d['alive_steps'] += 1

            dx, dy = self.action_deltas[actions[i]]
            new_pos = d['pos'] + np.array([dx, dy], dtype=np.float32) * self.drone_speed
            ix = int(np.clip(new_pos[0], 0, self.grid - 1))
            iy = int(np.clip(new_pos[1], 0, self.grid - 1))
            wind_push = np.array([self.wind_x[iy, ix], self.wind_y[iy, ix]]) * self.wind_coupling
            new_pos += wind_push
            thermal_val = float(self.thermal[iy, ix])
            new_pos = np.clip(new_pos, 1.0, self.grid - 2.0)
            d['vel'] = new_pos - d['pos']
            d['pos'] = new_pos

            reward = 0.0
            # NO per-step survival reward — forces agent to explore
            # Terminal bonus for surviving full episode instead

            gx, gy = int(new_pos[0]), int(new_pos[1])
            if 0 <= gx < self.grid and 0 <= gy < self.grid:
                if not self._visited_grid[gx, gy]:
                    self._visited_grid[gx, gy] = True
                    d['visited'].add((gx, gy))
                    self.total_cells_explored.add((gx, gy))
                    reward += 50.0
                elif self.fire[gx, gy] < 0.1:
                    reward += 0.1

            if (gx, gy) in perimeter:
                reward += 10.0
                self.visited_perimeter.add((gx, gy))

            displacement = np.linalg.norm(d['vel'])
            reward += displacement * 0.5

            frontier = self._get_frontier_direction(new_pos)
            vel_normalized = d['vel'] / (np.linalg.norm(d['vel']) + 1e-8)
            alignment = np.dot(vel_normalized, frontier)
            if alignment > 0:
                reward += alignment * 2.0

            d['prev_action'] = actions[i]
            ix2 = int(np.clip(np.round(new_pos[0]), 0, self.grid - 1))
            iy2 = int(np.clip(np.round(new_pos[1]), 0, self.grid - 1))
            wind_spd = float(np.sqrt(self.wind_x[iy2, ix2]**2 + self.wind_y[iy2, ix2]**2))
            crash, reason = self._check_crash(new_pos, thermal_val, wind_spd)
            if crash:
                d['alive'] = False
                d['crashes'] += 1
                dones[i] = True
                reward -= 10.0
            if d['battery'] <= 0:
                d['alive'] = False
                dones[i] = True
            rewards[i] = reward

        if all(not d['alive'] for d in self.drones):
            dones[:] = True
        if self.step_count >= self.max_steps:
            dones[:] = True
            # TERMINAL BONUS: reward drones for surviving the full episode
            for i in range(self.n_drones):
                if self.drones[i]['alive']:
                    rewards[i] += 200.0  # large bonus for full survival

        for i in range(self.n_drones):
            d = self.drones[i]
            pfr = (len(self.visited_perimeter) / self.total_perimeter_cells * 100
                   if self.total_perimeter_cells > 0 else 0)
            infos[i] = {
                'alive': d['alive'], 'crashes': d['crashes'],
                'cells_visited': len(d['visited']),
                'fire_dist': self._dist_to_fire(d['pos']) if d['alive'] else -1,
                'perimeter_frac': pfr, 'alive_steps': d['alive_steps'],
                'coverage': len(d['visited']) / (self.grid * self.grid) * 100,
                'total_coverage': len(self.total_cells_explored) / (self.grid * self.grid) * 100,
            }

        self._spread_fire(rng)
        self._update_fire_dist_cache()
        return self._get_obs(), rewards, dones, infos


# ═══════════════════════════════════════════════════════════════
# 2. PPO with PER-DRONE trajectory storage
# ═══════════════════════════════════════════════════════════════

class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=1e-3, gamma=0.99,
                 eps_clip=0.2, lam=0.95, ent_coef=0.15):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.lam = lam
        self.lr = lr
        self.ent_coef = ent_coef

        h = 128
        scale = np.sqrt(2.0 / obs_dim)
        self.w1 = np.random.randn(obs_dim, h).astype(np.float32) * scale
        self.b1 = np.zeros(h, dtype=np.float32)
        self.pol_w = np.random.randn(h, act_dim).astype(np.float32) * np.sqrt(2.0 / h)
        self.pol_b = np.zeros(act_dim, dtype=np.float32)
        self.val_w = np.random.randn(h, 1).astype(np.float32) * np.sqrt(2.0 / h)
        self.val_b = np.zeros(1, dtype=np.float32)

        self.params = ['w1', 'b1', 'pol_w', 'pol_b', 'val_w', 'val_b']
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.s = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.t = 0

        # Per-drone trajectory storage
        self.drone_bufs = {}

    def _init_drone_buf(self, idx):
        self.drone_bufs[idx] = {
            'obs': [], 'act': [], 'rew': [], 'val': [], 'done': [], 'logp': []
        }

    def _forward(self, obs_flat):
        h = np.maximum(0, obs_flat @ self.w1 + self.b1)
        logits = h @ self.pol_w + self.pol_b
        val = float((h @ self.val_w + self.val_b).ravel()[0])
        return h, logits, val

    def _softmax(self, logits):
        l = logits - np.max(logits)
        e = np.exp(l)
        return e / (np.sum(e) + 1e-8)

    def act(self, obs, deterministic=False, drone_idx=None):
        obs_flat = obs.flatten()[:self.obs_dim]
        h, logits, value = self._forward(obs_flat)
        probs = self._softmax(logits)
        action = int(np.argmax(probs)) if deterministic else int(
            np.random.choice(self.act_dim, p=probs))
        logp = np.log(probs[action] + 1e-8)
        return action, value, logp

    def store(self, drone_idx, obs, action, reward, value, done, logp):
        if drone_idx not in self.drone_bufs:
            self._init_drone_buf(drone_idx)
        b = self.drone_bufs[drone_idx]
        b['obs'].append(obs.flatten()[:self.obs_dim].copy())
        b['act'].append(action)
        b['rew'].append(reward)
        b['val'].append(value)
        b['done'].append(float(done))
        b['logp'].append(logp)

    def _compute_gae_single(self, rews, vals, dones):
        """Compute GAE for a single drone trajectory."""
        n = len(rews)
        if n == 0:
            return np.array([]), np.array([])
        adv = np.zeros(n, dtype=np.float32)
        ret = np.zeros(n, dtype=np.float32)
        last_gae = 0
        for t in reversed(range(n)):
            next_val = 0 if t == n - 1 else vals[t + 1]
            delta = rews[t] + self.gamma * next_val * (1 - dones[t]) - vals[t]
            last_gae = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae
            adv[t] = last_gae
            ret[t] = adv[t] + vals[t]
        return adv, ret

    def update(self, n_epochs=4, batch_size=128):
        # Collect all transitions with per-drone GAE
        all_obs, all_acts, all_adv, all_ret, all_logp = [], [], [], [], []

        for idx, b in self.drone_bufs.items():
            if len(b['rew']) < 2:
                continue
            # Mark done at end of trajectory
            dones = b['done'].copy()
            dones[-1] = 1.0  # episode end
            adv, ret = self._compute_gae_single(
                b['rew'], b['val'], dones)
            if len(adv) == 0:
                continue
            all_obs.extend(b['obs'])
            all_acts.extend(b['act'])
            all_adv.extend(adv.tolist())
            all_ret.extend(ret.tolist())
            all_logp.extend(b['logp'])

        # Clear buffers
        self.drone_bufs.clear()

        if len(all_obs) < 500:
            return 0.0

        all_obs = np.array(all_obs)
        all_acts = np.array(all_acts)
        all_adv = np.array(all_adv, dtype=np.float32)
        all_ret = np.array(all_ret, dtype=np.float32)
        all_logp = np.array(all_logp, dtype=np.float32)

        all_adv = (all_adv - np.mean(all_adv)) / (np.std(all_adv) + 1e-8)

        n = len(all_obs)
        total_loss = 0.0
        count = 0

        for _ in range(n_epochs):
            perm = np.random.permutation(n)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = perm[start:end]
                bs = len(idx)

                batch_obs = all_obs[idx]
                batch_acts = all_acts[idx]
                batch_adv = all_adv[idx]
                batch_ret = all_ret[idx]
                batch_logp = all_logp[idx]

                h = np.maximum(0, batch_obs @ self.w1 + self.b1)
                logits = h @ self.pol_w + self.pol_b
                logits_s = logits - np.max(logits, axis=1, keepdims=True)
                probs = np.exp(logits_s) / (np.sum(np.exp(logits_s), axis=1, keepdims=True) + 1e-8)

                new_logp = np.log(probs[np.arange(bs), batch_acts] + 1e-8)
                ratio = np.exp(new_logp - batch_logp)

                surr1 = ratio * batch_adv
                surr2 = np.clip(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_adv
                pol_loss = -np.mean(np.minimum(surr1, surr2))

                # Entropy bonus
                entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
                entropy_bonus = self.ent_coef * np.mean(entropy)

                val_pred = h @ self.val_w + self.val_b
                val_loss = 0.5 * np.mean((val_pred.ravel() - batch_ret)**2)

                # Combined loss (gradient update)
                loss = pol_loss + 0.5 * val_loss - entropy_bonus
                total_loss += loss
                count += 1

                # Gradients
                grad_logits = probs.copy()
                grad_logits[np.arange(bs), batch_acts] -= 1.0
                grad_logits *= (batch_adv[:, None] / bs)

                d_pol_w = h.T @ grad_logits
                d_pol_b = grad_logits.sum(axis=0)
                d_h = (grad_logits @ self.pol_w.T) * (h > 0).astype(float)
                d_enc_w = batch_obs.T @ d_h
                d_enc_b = d_h.sum(axis=0)

                val_err = (val_pred.ravel() - batch_ret)[:, None]
                d_val_w = h.T @ val_err
                d_val_b = val_err.sum(axis=0)
                d_hv = val_err @ self.val_w.T * (h > 0).astype(float)
                d_enc_w += batch_obs.T @ d_hv
                d_enc_b += d_hv.sum(axis=0)

                grads = {
                    'w1': np.clip(d_enc_w, -5.0, 5.0),
                    'b1': np.clip(d_enc_b, -5.0, 5.0),
                    'pol_w': np.clip(d_pol_w, -5.0, 5.0),
                    'pol_b': np.clip(d_pol_b, -5.0, 5.0),
                    'val_w': np.clip(d_val_w, -5.0, 5.0),
                    'val_b': np.clip(d_val_b, -5.0, 5.0),
                }

                self.t += 1
                for p in self.params:
                    self.m[p] = 0.9 * self.m[p] + 0.1 * grads[p]
                    self.s[p] = 0.999 * self.s[p] + 0.001 * grads[p]**2
                    m_hat = self.m[p] / (1 - 0.9**self.t)
                    s_hat = self.s[p] / (1 - 0.999**self.t)
                    setattr(self, p, getattr(self, p) - self.lr * m_hat / (np.sqrt(s_hat) + 1e-8))

        return float(total_loss / max(1, count))

    def save(self, path):
        d = {p: getattr(self, p) for p in self.params}
        np.savez(path, **d)

    def load(self, path):
        data = np.load(path)
        for p in self.params:
            setattr(self, p, data[p])


# ═══════════════════════════════════════════════════════════════
# 3. BASELINES
# ═══════════════════════════════════════════════════════════════

class RandomAgent:
    def __init__(self, act_dim=5): self.act_dim = act_dim
    def act(self, obs, deterministic=False):
        return (np.random.randint(self.act_dim), 0.0, 0.0)


class LawnmowerPositionAgent:
    def __init__(self, env):
        self.env = env
        self.grid = env.grid
        self.n_drones = env.n_drones
        self.path = self._plan_path()
        self.drone_progress = [0] * env.n_drones

    def _plan_path(self):
        targets = []
        step = 2
        for y in range(1, self.grid - 1, step):
            if (y // step) % 2 == 0:
                for x in range(1, self.grid - 1):
                    targets.append(np.array([float(x), float(y)]))
            else:
                for x in range(self.grid - 2, 0, -1):
                    targets.append(np.array([float(x), float(y)]))
        return targets

    def act(self, obs_idx, drone_idx):
        if drone_idx >= self.n_drones:
            return 0
        pos = self.env.drones[drone_idx]['pos']
        fire_dist = self.env._dist_to_fire(pos)
        if fire_dist < 4.0:
            fcx, fcy = self.env.fire_cx, self.env.fire_cy
            away = pos - np.array([fcx, fcy])
            if abs(away[0]) > abs(away[1]):
                return 3 if away[0] > 0 else 4
            else:
                return 1 if away[1] > 0 else 2
        if pos[0] < 3: return 3
        if pos[0] > self.grid - 4: return 4
        if pos[1] < 3: return 1
        if pos[1] > self.grid - 4: return 2
        progress = self.drone_progress[drone_idx]
        target = self.path[progress % len(self.path)]
        diff = target - pos
        if abs(diff[0]) > abs(diff[1]):
            action = 3 if diff[0] > 0 else 4
        else:
            action = 1 if diff[1] > 0 else 2
        if np.linalg.norm(diff) < 2.0:
            self.drone_progress[drone_idx] += 1
        return action


class GreedyExplorationAgent:
    def __init__(self, env):
        self.env = env

    def act(self, obs_idx, drone_idx):
        if drone_idx >= self.env.n_drones:
            return 0
        pos = self.env.drones[drone_idx]['pos']
        fire_dist = self.env._dist_to_fire(pos)
        if fire_dist < 3.0:
            fcx, fcy = self.env.fire_cx, self.env.fire_cy
            away = pos - np.array([fcx, fcy])
            if abs(away[0]) > abs(away[1]):
                return 3 if away[0] > 0 else 4
            else:
                return 1 if away[1] > 0 else 2
        unexplored = np.argwhere(~self.env._visited_grid)
        if len(unexplored) == 0:
            return 0
        dists = np.sum((unexplored - pos)**2, axis=1)
        nearest = unexplored[np.argmin(dists)]
        diff = nearest - pos
        if abs(diff[0]) > abs(diff[1]):
            return 3 if diff[0] > 0 else 4
        else:
            return 1 if diff[1] > 0 else 2


class PIDAgent:
    def __init__(self):
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)

    def act(self, obs, deterministic=False):
        fire = obs[0]
        if fire.max() < 0.01:
            return (0, 0.0, 0.0)
        try:
            r = obs.shape[1] // 2
            yy, xx = np.meshgrid(np.arange(obs.shape[1]), np.arange(obs.shape[2]))
            cx = np.average(xx, weights=fire + 1e-8)
            cy = np.average(yy, weights=fire + 1e-8)
            error = np.array([cx - r, cy - r])
            self.integral += error
            deriv = error - self.prev_error
            self.prev_error = error.copy()
            control = 1.5 * error + 0.1 * self.integral + 0.3 * deriv
            if abs(control[0]) > abs(control[1]):
                return (3 if control[0] > 0 else 4, 0.0, 0.0)
            else:
                return (1 if control[1] > 0 else 2, 0.0, 0.0)
        except:
            return (0, 0.0, 0.0)


# ═══════════════════════════════════════════════════════════════
# 4. TRAINING (per-drone trajectory storage)
# ═══════════════════════════════════════════════════════════════

def train_ppo(n_episodes=3000, grid=30, n_drones=10, max_steps=300,
              log_every=100):
    env = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps)
    agent = PPOAgent(env.obs_dim, env.act_dim, lr=1e-3, ent_coef=0.15)

    wind_levels = [0, 3, 5, 8, 10, 12, 15, 18, 20, 25]
    eps_per_level = n_episodes // len(wind_levels)

    history = {'rewards': [], 'perimeters': [], 'safety': [], 'wind': [],
               'coverage': [], 'total_coverage': []}
    best_reward = -1e9
    start = time.time()

    print(f"{'='*70}")
    print(f"Training PPO v3 | {n_episodes} eps | {n_drones} drones | {grid}x{grid}")
    print(f"  Per-drone GAE, ent_coef=0.15, lr=1e-3")
    print(f"{'='*70}")

    for ep in range(n_episodes):
        level_idx = min(len(wind_levels) - 1, ep // eps_per_level)
        env.base_wind = wind_levels[level_idx]

        obs = env.reset(seed=ep)
        ep_reward = 0.0

        for step in range(max_steps):
            actions = np.zeros(n_drones, dtype=int)
            for i in range(n_drones):
                if not env.drones[i]['alive']:
                    continue
                action, value, logp = agent.act(obs[i], drone_idx=i)
                agent.store(i, obs[i], action, 0, value, False, logp)
                actions[i] = action

            obs, rewards, dones, infos = env.step(actions)

            # Update stored rewards
            for i in range(n_drones):
                if agent.drone_bufs.get(i):
                    b = agent.drone_bufs[i]
                    if len(b['rew']) > 0 and not b['done'][-1]:
                        b['rew'][-1] = rewards[i]
                        if dones[i]:
                            b['done'][-1] = 1.0

            ep_reward += np.sum(rewards)
            if all(dones):
                break

        loss = agent.update(n_epochs=4, batch_size=128)

        alive = sum(1 for d in env.drones if d['alive'])
        safety = alive / n_drones * 100
        cov = np.mean([infos[i].get('coverage', 0) for i in range(n_drones)])
        total_cov = infos[0].get('total_coverage', 0) if infos else 0
        pfr = infos[0].get('perimeter_frac', 0) if infos else 0

        history['rewards'].append(ep_reward)
        history['perimeters'].append(pfr)
        history['safety'].append(safety)
        history['wind'].append(env.base_wind)
        history['coverage'].append(cov)
        history['total_coverage'].append(total_cov)

        if (ep + 1) % log_every == 0:
            r = np.mean(history['rewards'][-log_every:])
            p = np.mean(history['perimeters'][-log_every:])
            s = np.mean(history['safety'][-log_every:])
            c = np.mean(history['coverage'][-log_every:])
            tc = np.mean(history['total_coverage'][-log_every:])
            elapsed = time.time() - start
            print(f"Ep {ep+1:5d}/{n_episodes} | R:{r:8.1f} | Peri:{p:.2f}% | "
                  f"Safe:{s:.0f}% | Cov:{c:.1f}% | TotCov:{tc:.1f}% | "
                  f"Wind:{env.base_wind:.0f} | t:{elapsed:.0f}s")
            if r > best_reward:
                best_reward = r
                agent.save('best_ppo.npz')

    agent.save('final_ppo.npz')
    total_time = time.time() - start
    print(f"\nTraining done: {total_time:.0f}s ({n_episodes / total_time:.2f} eps/s)")
    print(f"Final: R={np.mean(history['rewards'][-200:]):.1f} "
          f"Safe={np.mean(history['safety'][-200:]):.1f}% "
          f"Cov={np.mean(history['coverage'][-200:]):.1f}% "
          f"TotCov={np.mean(history['total_coverage'][-200:]):.1f}%")
    return agent, history


# ═══════════════════════════════════════════════════════════════
# 5. EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_ppo(agent, env, n_episodes=20, wind=12):
    env.base_wind = wind
    results = []
    for ep in range(n_episodes):
        obs = env.reset(seed=ep + 10000)
        for step in range(env.max_steps):
            actions = np.zeros(env.n_drones, dtype=int)
            for i in range(env.n_drones):
                if env.drones[i]['alive']:
                    action, _, _ = agent.act(obs[i], deterministic=True)
                    actions[i] = action
            obs, _, dones, infos = env.step(actions)
            if all(dones):
                break
        alive = sum(1 for d in env.drones if d['alive'])
        cells = max((len(d['visited']) for d in env.drones), default=0)
        alive_steps = max((d['alive_steps'] for d in env.drones), default=0)
        total_cov = infos[0].get('total_coverage', 0) if infos else 0
        pfr = infos[0].get('perimeter_frac', 0) if infos else 0
        results.append({
            'safety': alive / env.n_drones * 100,
            'perimeter': pfr, 'cells': cells,
            'total_coverage': total_cov, 'alive_steps': alive_steps,
        })
    m = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
    m['safety_std'] = float(np.std([r['safety'] for r in results]))
    m['total_coverage_std'] = float(np.std([r['total_coverage'] for r in results]))
    return m


def evaluate_baseline(agent_fn, env, n_episodes=20, wind=12, use_position=False):
    env.base_wind = wind
    results = []
    for ep in range(n_episodes):
        obs = env.reset(seed=ep + 10000)
        agent_state = None
        if use_position:
            agent_state = agent_fn()
        for step in range(env.max_steps):
            actions = np.zeros(env.n_drones, dtype=int)
            for i in range(env.n_drones):
                if env.drones[i]['alive']:
                    if use_position:
                        actions[i] = agent_state.act(obs[i], i)
                    else:
                        result = agent_state.act(obs[i])
                        actions[i] = int(result[0]) if isinstance(result, tuple) else int(result)
            obs, _, dones, infos = env.step(actions)
            if all(dones):
                break
        alive = sum(1 for d in env.drones if d['alive'])
        cells = max((len(d['visited']) for d in env.drones), default=0)
        alive_steps = max((d['alive_steps'] for d in env.drones), default=0)
        total_cov = infos[0].get('total_coverage', 0) if infos else 0
        pfr = infos[0].get('perimeter_frac', 0) if infos else 0
        results.append({
            'safety': alive / env.n_drones * 100,
            'perimeter': pfr, 'cells': cells,
            'total_coverage': total_cov, 'alive_steps': alive_steps,
        })
    m = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
    m['safety_std'] = float(np.std([r['safety'] for r in results]))
    m['total_coverage_std'] = float(np.std([r['total_coverage'] for r in results]))
    return m


# ═══════════════════════════════════════════════════════════════
# 6. FIGURES
# ═══════════════════════════════════════════════════════════════

def generate_figures(history, all_results, wind_levels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs('figures', exist_ok=True)

    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 11,
        'axes.labelsize': 12, 'axes.titlesize': 13,
        'figure.dpi': 150,
    })

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    window = 50
    eps = np.arange(len(history['rewards']))
    for ax, key, label, color in [
        (axes[0,0], 'rewards', 'Episode Reward', '#2196F3'),
        (axes[0,1], 'safety', 'Safety Rate (%)', '#4CAF50'),
        (axes[1,0], 'coverage', 'Per-Drone Coverage (%)', '#FF9800'),
        (axes[1,1], 'total_coverage', 'Total Grid Coverage (%)', '#9C27B0'),
    ]:
        data = history[key]
        if len(data) >= window:
            smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
            ax.plot(eps[:len(smoothed)], smoothed, color=color, linewidth=1.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    plt.suptitle('PPO Training (30×30, 10 Drones)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/fig1_training.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Figure 1")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    names = list(all_results.keys())
    colors = ['#9E9E9E', '#4CAF50', '#FF9800', '#2196F3', '#F44336']
    for ax, key, ylabel in [
        (axes[0], 'safety', 'Safety (%)'),
        (axes[1], 'total_coverage', 'Coverage (%)'),
        (axes[2], 'perimeter', 'Perimeter (%)'),
    ]:
        vals = [all_results[n][key] for n in names]
        errs = [all_results[n].get(f'{key}_std', 0) for n in names]
        ax.bar(range(len(names)), vals, yerr=errs, color=colors[:len(names)],
               capsize=3, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis='y')
    plt.suptitle('Benchmark (30×30, Wind=12 m/s)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/fig2_benchmark.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Figure 2")


# ═══════════════════════════════════════════════════════════════
# 7. MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    total_start = time.time()
    print("=" * 70)
    print("Paper-Ready Training Pipeline v3")
    print("=" * 70)

    print("\n[1/4] Training PPO (3000 eps, curriculum 0→25 m/s)...")
    agent, history = train_ppo(
        n_episodes=3000, grid=30, n_drones=10, max_steps=300, log_every=300)

    print("\n[2/4] Benchmarking (wind=12 m/s)...")
    eval_wind = 12
    all_results = {}
    all_results['PPO'] = evaluate_ppo(agent, WildfireEnv(grid=30, n_drones=10, max_steps=300),
                                        n_episodes=20, wind=eval_wind)
    all_results['Greedy'] = evaluate_baseline(
        lambda: None, WildfireEnv(grid=30, n_drones=10, max_steps=300),
        n_episodes=20, wind=eval_wind, use_position=True)
    all_results['Lawnmower'] = evaluate_baseline(
        lambda: None, WildfireEnv(grid=30, n_drones=10, max_steps=300),
        n_episodes=20, wind=eval_wind, use_position=True)
    all_results['PID'] = evaluate_baseline(
        PIDAgent, WildfireEnv(grid=30, n_drones=10, max_steps=300),
        n_episodes=20, wind=eval_wind)
    all_results['Random'] = evaluate_baseline(
        lambda: RandomAgent(), WildfireEnv(grid=30, n_drones=10, max_steps=300),
        n_episodes=20, wind=eval_wind)

    print(f"\n{'='*90}")
    print(f"{'Method':<15} {'Safety':>8} {'Perimeter':>10} {'Coverage':>10} {'TotalCov':>10} {'Alive':>8}")
    print(f"{'-'*90}")
    for name in ['PPO', 'Greedy', 'Lawnmower', 'PID', 'Random']:
        r = all_results[name]
        print(f"{name:<15} {r['safety']:>6.1f}% {r['perimeter']:>8.2f}% "
              f"{r['cells']:>8.0f} {r['total_coverage']:>8.1f}% {r['alive_steps']:>7.0f}")
    print(f"{'='*90}")

    print("\n[3/4] Generating figures...")
    try:
        generate_figures(history, all_results, [0, 3, 5, 8, 10, 12, 15, 18, 20, 25])
    except ImportError as e:
        print(f"  matplotlib not available: {e}")

    print("\n[4/4] Saving results...")
    output = {
        'benchmark': all_results,
        'training': {
            'final_reward': float(np.mean(history['rewards'][-200:])),
            'final_safety': float(np.mean(history['safety'][-200:])),
            'final_coverage': float(np.mean(history['coverage'][-200:])),
            'final_total_coverage': float(np.mean(history['total_coverage'][-200:])),
        }
    }
    with open('paper_ready_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    t = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"DONE! {t:.0f}s ({t/60:.1f} min)")
    print(f"{'='*70}")
