"""
WildfirePerimeterTracking Environment — PlumeGym-MARL

Physics-informed multi-agent simulation with:
- Rothermel fire spread (wind-driven, fuel moisture, spotting)
- Thermal plume model (Gaussian updrafts)
- Multi-drone dynamics (momentum, wind coupling, crash conditions)
- Decentralized observations (local 9x9 grid + global features)
"""
import numpy as np

class WildfireEnv:
    """
    Multi-agent wildfire perimeter tracking environment.
    
    Args:
        grid: Grid size (N x N cells)
        n_drones: Number of drones
        max_steps: Maximum steps per episode
        wind_speed: Base wind speed (m/s)
        obs_r: Local observation radius
    """
    
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
        
        self.action_deltas = np.array([
            [0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]
        ], dtype=np.float32)
        
        self.wind_coupling = 0.02
        self.momentum = 0.7
        self.fire_crash_threshold = 0.3
        self.thermal_crash = 15.0
        self.boundary_margin = 1.0
        self.thermal_cap = 25.0
        
        # Fire spread params
        self.spread_rate = 0.1
        self.wind_amplification = 0.05
        self.fuel_depletion_rate = 0.002
        self.base_fire_radius = 3
        self.fire_intensity_init = 0.8
        self.spotting_prob = 0.02
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state."""
        self.step_count = 0
        self.total_cells_explored = set()
        
        # Initialize fire
        self.fire = np.zeros((self.grid, self.grid), dtype=np.float32)
        margin = min(self.base_fire_radius + 2, self.grid // 4)
        cx = np.random.randint(margin, max(margin + 1, self.grid - margin))
        cy = np.random.randint(margin, max(margin + 1, self.grid - margin))
        for dx in range(-self.base_fire_radius, self.base_fire_radius + 1):
            for dy in range(-self.base_fire_radius, self.base_fire_radius + 1):
                if dx*dx + dy*dy <= self.base_fire_radius**2:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.grid and 0 <= ny < self.grid:
                        self.fire[ny, nx] = self.fire_intensity_init
        
        self.fire_center = np.array([cx, cy], dtype=np.float32)
        
        # Fuel
        self.fuel = np.clip(0.8 - 0.3 * np.random.randn(self.grid, self.grid), 0.3, 1.0).astype(np.float32)
        
        # Wind field
        angle = np.random.uniform(0, 2 * np.pi)
        self.wind_x = np.full((self.grid, self.grid), self.base_wind * np.cos(angle), dtype=np.float32)
        self.wind_y = np.full((self.grid, self.grid), self.base_wind * np.sin(angle), dtype=np.float32)
        
        # Add gusts
        yy, xx = np.meshgrid(np.arange(self.grid), np.arange(self.grid), indexing='ij')
        for k in range(3):
            freq = 0.5 * (k + 1)
            amp = 0.1 * self.base_wind
            phase = np.random.uniform(0, 2 * np.pi)
            self.wind_x += amp * np.sin(2 * np.pi * freq * xx / self.grid + phase)
            self.wind_y += amp * np.sin(2 * np.pi * freq * yy / self.grid + phase * 0.7)
        
        # Thermal plume
        self.thermal = np.zeros((self.grid, self.grid), dtype=np.float32)
        self._update_thermal()
        
        # Fire distance cache
        self._fire_dist_cache = None
        self._update_fire_dist()
        
        # Visited grid (shared)
        self._visited_grid = np.zeros((self.grid, self.grid), dtype=np.float32)
        
        # Initialize drones
        self.drones = []
        for i in range(self.n_drones):
            while True:
                px = np.random.uniform(2, self.grid - 2)
                py = np.random.uniform(2, self.grid - 2)
                # Not on fire
                ix, iy = int(px), int(py)
                if 0 <= ix < self.grid and 0 <= iy < self.grid and self.fire[iy, ix] < 0.1:
                    break
            self.drones.append({
                'pos': np.array([px, py], dtype=np.float32),
                'vel': np.array([0.0, 0.0], dtype=np.float32),
                'alive': True,
                'visited': set(),
                'battery': 1.0,
            })
        
        return self._get_obs()
    
    def _update_thermal(self):
        """Compute thermal updrafts from fire."""
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
        """Compute distance to nearest fire cell for each grid cell."""
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
        """Spread fire using simplified Rothermel model."""
        fire_cells = np.argwhere(self.fire > 0.2)
        new_fire = self.fire.copy()
        
        for fy, fx in fire_cells:
            intensity = self.fuel[fy, fx] * self.fire[fy, fx]
            if intensity < 0.05:
                continue
            
            # Wind-driven spread
            wind_mag = np.sqrt(self.wind_x[fy, fx]**2 + self.wind_y[fy, fx]**2)
            spread_prob = self.spread_rate * (1 + self.wind_amplification * wind_mag) * intensity
            
            # Neighbor spread
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nx, ny = fx + dx, fy + dy
                if 0 <= nx < self.grid and 0 <= ny < self.grid:
                    if self.fuel[ny, nx] > 0.1:
                        # Wind direction bonus
                        wind_dir = np.array([self.wind_x[fy, fx], self.wind_y[fy, fx]])
                        neighbor_dir = np.array([dx, dy], dtype=np.float32)
                        wind_factor = 1.0
                        if np.linalg.norm(wind_dir) > 0.1 and np.linalg.norm(neighbor_dir) > 0:
                            cos_angle = np.dot(wind_dir, neighbor_dir) / (np.linalg.norm(wind_dir) + 1e-8)
                            wind_factor = 1.0 + 0.5 * max(0, cos_angle)
                        
                        prob = spread_prob * wind_factor * self.fuel[ny, nx]
                        if rng.random() < prob:
                            new_fire[ny, nx] = min(1.0, new_fire[ny, nx] + 0.1)
            
            # Spotting (ember transport)
            if rng.random() < self.spotting_prob * wind_mag / 10.0:
                spot_dist = rng.integers(1, 4)
                spot_angle = rng.uniform(0, 2 * np.pi)
                sx = int(fx + spot_dist * np.cos(spot_angle))
                sy = int(fy + spot_dist * np.sin(spot_angle))
                if 0 <= sx < self.grid and 0 <= sy < self.grid:
                    if self.fuel[sy, sx] > 0.1:
                        new_fire[sy, sx] = min(1.0, new_fire[sy, sx] + 0.3)
        
        # Decay
        new_fire = new_fire * (1 - self.fuel_depletion_rate)
        self.fire = np.clip(new_fire, 0, 1).astype(np.float32)
    
    def _check_crash(self, pos, thermal_val, wind_spd):
        """Check if drone has crashed."""
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
        
        return False, 'safe'
    
    def _get_frontier_direction(self, pos):
        """Get direction toward nearest unexplored frontier."""
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
                            if dist > 0:
                                best_dir = np.array([dx / dist, dy / dist])
        
        return best_dir
    
    def _get_obs(self):
        """Get observations for all drones."""
        r = self.obs_r
        obs = np.zeros((self.n_drones, self.obs_dim), dtype=np.float32)
        ch_size = self.obs_size * self.obs_size
        
        for i in range(self.n_drones):
            if not self.drones[i]['alive']:
                continue
            cx, cy = int(self.drones[i]['pos'][0]), int(self.drones[i]['pos'][1])
            x_min, x_max = max(0, cx - r), min(self.grid, cx + r + 1)
            y_min, y_max = max(0, cy - r), min(self.grid, cy + r + 1)
            h = y_max - y_min
            w = x_max - x_min
            
            local = np.zeros(self.local_obs_dim, dtype=np.float32)
            ch_idx = 0
            
            # Ch0: fire
            grid = np.zeros((self.obs_size, self.obs_size), dtype=np.float32)
            grid[:h, :w] = self.fire[y_min:y_max, x_min:x_max]
            local[ch_idx*ch_size:(ch_idx+1)*ch_size] = grid.ravel(); ch_idx += 1
            
            # Ch1-2: wind
            for wind_arr in [self.wind_x, self.wind_y]:
                grid = np.zeros((self.obs_size, self.obs_size), dtype=np.float32)
                grid[:h, :w] = wind_arr[y_min:y_max, x_min:x_max] / 30.0
                local[ch_idx*ch_size:(ch_idx+1)*ch_size] = grid.ravel(); ch_idx += 1
            
            # Ch3: fuel
            grid = np.zeros((self.obs_size, self.obs_size), dtype=np.float32)
            grid[:h, :w] = self.fuel[y_min:y_max, x_min:x_max]
            local[ch_idx*ch_size:(ch_idx+1)*ch_size] = grid.ravel(); ch_idx += 1
            
            # Ch4: thermal
            grid = np.zeros((self.obs_size, self.obs_size), dtype=np.float32)
            grid[:h, :w] = self.thermal[y_min:y_max, x_min:x_max] / self.thermal_cap
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
                grid[:h, :w] = np.minimum(self._fire_dist_cache[y_min:y_max, x_min:x_max] / 10.0, 1.0)
            local[ch_idx*ch_size:(ch_idx+1)*ch_size] = grid.ravel(); ch_idx += 1
            
            # Ch7: visited
            grid = np.zeros((self.obs_size, self.obs_size), dtype=np.float32)
            grid[:h, :w] = self._visited_grid[y_min:y_max, x_min:x_max]
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
    
    def step(self, actions):
        """
        Step the environment.
        
        Args:
            actions: (n_drones,) array of action indices
        Returns:
            obs, rewards, dones, infos
        """
        self.step_count += 1
        rng = np.random.default_rng(self.step_count)
        
        rewards = np.zeros(self.n_drones, dtype=np.float32)
        dones = np.zeros(self.n_drones, dtype=bool)
        infos = [{} for _ in range(self.n_drones)]
        
        # Update fire
        self._spread_fire(rng)
        if self.step_count % 5 == 0:
            self._update_thermal()
            self._update_fire_dist()
        
        # Move drones
        for i in range(self.n_drones):
            if not self.drones[i]['alive']:
                dones[i] = True
                continue
            
            d = self.drones[i]
            action = int(actions[i]) if hasattr(actions[i], 'item') else int(actions[i])
            dx, dy = self.action_deltas[action]
            
            # New velocity with momentum and wind
            target_vel = np.array([dx, dy], dtype=np.float32)
            ix = int(np.clip(d['pos'][0], 0, self.grid - 1))
            iy = int(np.clip(d['pos'][1], 0, self.grid - 1))
            wind_push = np.array([self.wind_x[iy, ix], self.wind_y[iy, ix]]) * self.wind_coupling
            
            d['vel'] = self.momentum * d['vel'] + (1 - self.momentum) * target_vel + wind_push
            
            # New position
            new_pos = d['pos'] + d['vel']
            new_pos = np.clip(new_pos, self.boundary_margin, self.grid - self.boundary_margin)
            
            # Check crash
            ix_new = int(np.clip(new_pos[0], 0, self.grid - 1))
            iy_new = int(np.clip(new_pos[1], 0, self.grid - 1))
            thermal_val = float(self.thermal[iy_new, ix_new])
            wind_spd = float(np.sqrt(self.wind_x[iy_new, ix_new]**2 + self.wind_y[iy_new, ix_new]**2))
            
            crashed, crash_reason = self._check_crash(new_pos, thermal_val, wind_spd)
            
            if crashed:
                d['alive'] = False
                dones[i] = True
                rewards[i] = -10.0
                infos[i] = {'crash': True, 'reason': crash_reason, 'fire_dist': 0.0, 'thermal': thermal_val, 'wind_speed': wind_spd}
            else:
                d['pos'] = new_pos
                # Mark visited
                gx, gy = int(new_pos[0]), int(new_pos[1])
                if 0 <= gx < self.grid and 0 <= gy < self.grid:
                    d['visited'].add((gx, gy))
                    self.total_cells_explored.add((gx, gy))
                    self._visited_grid[gy, gx] = 1.0
                
                fire_dist = float(self._fire_dist_cache[iy_new, ix_new]) if self._fire_dist_cache is not None else 10.0
                infos[i] = {'crash': False, 'fire_dist': fire_dist, 'thermal': thermal_val, 'wind_speed': wind_spd}
                rewards[i] = 1.0  # survival reward
            
            # Battery
            d['battery'] = max(0, d['battery'] - 0.001)
        
        return self._get_obs(), rewards, dones, infos
