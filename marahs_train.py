#!/usr/bin/env python3
"""
MARAHS Training Pipeline — All 4 Innovations (Optimized)
==========================================================
Key optimizations:
- GP grid predictions cached (only recomputed when dirty)
- Info gain map cached per step  
- GAT cached per step
- Neural-CBF warm-started from rule-based safety labels
- Neural-CBF only overrides when h < -0.5 (not h < 0)
"""
import numpy as np
import json, os, time, sys
from kaggle_wildfire_train import (
    WildfireEnv, PPOAgent, RandomAgent, GreedyFireAgent, PIDAgent,
    NeuralCBFSafetyFilter
)
from marahs_core import GPFireFront, InformationTheoreticPlanner, NeuralCBFNetwork, GATCommunicationModule


class MARAHSAgent:
    """
    Complete MARAHS agent integrating all 4 innovations.
    
    Observation: raw env obs (7x9x9=567) + GP(1) + info_gain(1) + GAT(12) = 581
    """
    
    def __init__(self, env, hidden=128, lr=3e-4):
        self.env = env
        self.grid = env.grid
        self.n_drones = env.n_drones
        
        # Innovation 1: GP Fire Front (cached grid predictions)
        self.gp = GPFireFront(grid_size=env.grid, length_scale=3.0, max_points=150)
        
        # Innovation 2: Info-Theoretic Planner (cached info gain map)
        self.info_planner = InformationTheoreticPlanner(self.gp, grid_size=env.grid)
        
        # Innovation 3: Neural-CBF (learned, not rule-based)
        self.neural_cbf = NeuralCBFNetwork(input_dim=8, hidden_dim=32, lr=1e-3)
        
        # Innovation 4: GAT Communication (cached per step)
        self.gat = GATCommunicationModule(node_dim=12, n_heads=3, comm_range=8.0)
        
        # Observation dimensions
        flat_obs = env.obs_channels * env.obs_size * env.obs_size  # 567
        self.extended_obs_dim = flat_obs + 1 + 1 + 12  # 581
        
        # PPO agent
        self.ppo = PPOAgent(self.extended_obs_dim, env.act_dim, hidden=hidden, lr=lr)
        
        # Metrics
        self.gp_observations = 0
        self.cbf_interventions = 0
        self.cbf_total_checks = 0
        self._current_step = 0
        self._warmed_up = False
    
    def warmstart_cbf(self, n_samples=500):
        """
        Warm-start Neural-CBF with rule-based safety labels.
        
        Generates safe/unsafe state pairs from the environment's deterministic
        crash conditions so the Neural-CBF learns the safety boundary BEFORE
        training begins. This prevents the 99% override problem.
        """
        rng = np.random.default_rng(42)
        grid = self.grid
        
        for _ in range(n_samples):
            # Random position on grid
            px = rng.uniform(1.0, grid - 2.0)
            py = rng.uniform(1.0, grid - 2.0)
            pos = np.array([px, py], dtype=np.float32)
            
            # Random velocity
            vx = rng.uniform(-1.5, 1.5)
            vy = rng.uniform(-1.5, 1.5)
            vel = np.array([vx, vy], dtype=np.float32)
            
            # Compute environment features
            fire_dist = self.env._dist_to_fire(pos)
            ix = int(np.clip(px, 0, grid - 1))
            iy = int(np.clip(py, 0, grid - 1))
            thermal = float(self.env.thermal[iy, ix]) if hasattr(self.env, 'thermal') else 0.0
            wind = np.array([
                float(self.env.wind_x[iy, ix]) if hasattr(self.env, 'wind_x') else 0.0,
                float(self.env.wind_y[iy, ix]) if hasattr(self.env, 'wind_y') else 0.0
            ])
            
            # Compute CBF feature vector
            state = self.neural_cbf.compute_features(pos, vel, fire_dist, thermal, wind)
            
            # Rule-based safety label (matching WildfireEnv._check_crash)
            is_safe = True
            if fire_dist < 0.5:
                is_safe = False
            if thermal > 13.0:
                is_safe = False
            if px < 1.0 or px > grid - 2.0 or py < 1.0 or py > grid - 2.0:
                is_safe = False
            wind_spd = np.sqrt(wind[0]**2 + wind[1]**2)
            if wind_spd > 35.0:
                is_safe = False
            
            # Store as transition: next state same as current (single-step label)
            self.neural_cbf.store_transition(state, is_safe, state)
        
        # Train CBF on warm-start data (many steps to converge)
        for _ in range(20):
            self.neural_cbf.train_step(n_samples=128)
        
        self._warmed_up = True
        print(f"  Neural-CBF warm-started with {n_samples} samples, "
              f"buffer size: {len(self.neural_cbf.buffer)}")
    
    def update_gp(self):
        """Update GP with current drone observations."""
        for i in range(self.n_drones):
            d = self.env.drones[i]
            if d['alive']:
                pos = d['pos']
                ix = int(np.clip(pos[0], 0, self.env.grid - 1))
                iy = int(np.clip(pos[1], 0, self.env.grid - 1))
                fire_val = float(self.env.fire[iy, ix])
                self.gp.add_observation(pos, fire_val)
                self.gp_observations += 1
    
    def _build_gat_states(self):
        """Build GAT input states from all alive drones."""
        gat_states = []
        gat_positions = []
        alive_indices = []
        for j in range(self.n_drones):
            d = self.env.drones[j]
            if d['alive']:
                fire_dist = self.env._dist_to_fire(d['pos'])
                ix = int(np.clip(d['pos'][0], 0, self.env.grid - 1))
                iy = int(np.clip(d['pos'][1], 0, self.env.grid - 1))
                thermal = float(self.env.thermal[iy, ix])
                gat_states.append(np.array([
                    d['pos'][0], d['pos'][1],
                    d['vel'][0], d['vel'][1],
                    fire_dist, thermal
                ], dtype=np.float32))
                gat_positions.append(d['pos'].copy())
                alive_indices.append(j)
        return gat_states, gat_positions, alive_indices
    
    def get_all_actions(self, obs, deterministic=False):
        """
        Get actions for ALL alive drones in one pass.
        - Builds GAT features once per step (cached)
        - Computes GP+info gain once per step (cached)
        - Returns per-drone actions, values, logps, and CBF flags
        """
        self._current_step += 1
        
        # --- Innovation 1 & 2: GP + Info gain (cached, computed once) ---
        gp_mean_grid, gp_var_grid = self.gp.predict_cached()
        info_gain_map = self.info_planner.compute_info_gain_map()
        
        # --- Innovation 4: GAT (computed once per step) ---
        gat_states, gat_positions, alive_indices = self._build_gat_states()
        gat_features = self.gat.communicate(gat_states, np.array(gat_positions),
                                             step_id=self._current_step)
        
        gat_lookup = {alive_indices[k]: k for k in range(len(alive_indices))}
        
        results = {}
        for i in range(self.n_drones):
            d = self.env.drones[i]
            if not d['alive']:
                continue
            
            obs_flat = obs[i].flatten()
            
            ix = int(np.clip(d['pos'][0], 0, self.grid - 1))
            iy = int(np.clip(d['pos'][1], 0, self.grid - 1))
            gp_feat = float(gp_mean_grid[ix, iy])
            info_feat = float(info_gain_map[ix, iy])
            
            if i in gat_lookup:
                gat_feat = gat_features[gat_lookup[i]]
            else:
                gat_feat = np.zeros(12, dtype=np.float32)
            
            extended = np.concatenate([obs_flat, [gp_feat, info_feat], gat_feat])
            action, value, logp = self.ppo.act(extended, deterministic=deterministic)
            
            # --- Innovation 3: Neural-CBF safety filter ---
            fire_dist = self.env._dist_to_fire(d['pos'])
            thermal = float(self.env.thermal[iy, ix])
            wind = np.array([float(self.env.wind_x[iy, ix]), float(self.env.wind_y[iy, ix])])
            
            cbf_state = self.neural_cbf.compute_features(d['pos'], d['vel'], fire_dist, thermal, wind)
            h = self.neural_cbf.safety_margin(cbf_state)
            
            self.cbf_total_checks += 1
            cbf_override = False
            
            # Only override if Neural-CBF is CERTAIN state is unsafe (h < -0.5)
            # This prevents the untrained CBF from overriding everything
            if h < -0.5:
                best_h = -float('inf')
                best_action = action
                for a in range(5):
                    dx, dy = self.env.action_deltas[a]
                    new_pos = d['pos'] + np.array([dx, dy], dtype=np.float32) * self.env.drone_speed
                    new_pos = np.clip(new_pos, 1.0, self.env.grid - 2.0)
                    
                    nix = int(np.clip(new_pos[0], 0, self.env.grid - 1))
                    niy = int(np.clip(new_pos[1], 0, self.env.grid - 1))
                    nfire = self.env._dist_to_fire(new_pos)
                    ntherm = float(self.env.thermal[niy, nix])
                    nwind = np.array([float(self.env.wind_x[niy, nix]), float(self.env.wind_y[niy, nix])])
                    
                    ns = self.neural_cbf.compute_features(new_pos, d['vel'], nfire, ntherm, nwind)
                    h_a = self.neural_cbf.safety_margin(ns)
                    
                    if h_a > best_h:
                        best_h = h_a
                        best_action = a
                
                action = best_action
                cbf_override = True
                self.cbf_interventions += 1
                self.neural_cbf.store_transition(cbf_state, False, cbf_state)
            else:
                # State is safe or uncertain — record as safe for CBF training
                self.neural_cbf.store_transition(cbf_state, True, cbf_state)
            
            results[i] = {
                'action': action, 'value': value, 'logp': logp,
                'extended': extended, 'cbf_state': cbf_state,
                'cbf_override': cbf_override, 'h': h,
            }
        
        return results
    
    def store_transitions(self, results, obs, rewards, dones):
        """Store all transitions into PPO buffer."""
        for i, r in results.items():
            self.ppo.store(r['extended'], r['action'], rewards[i], r['value'], dones[i], r['logp'])
    
    def train_ppo(self):
        return self.ppo.update(n_epochs=3, batch_size=128)
    
    def train_neural_cbf(self):
        return self.neural_cbf.train_step()
    
    def save(self, prefix='marahs'):
        np.savez(f'{prefix}_ppo.npz',
                 enc_w=self.ppo.enc_w, enc_b=self.ppo.enc_b,
                 pol_w1=self.ppo.pol_w1, pol_b1=self.ppo.pol_b1,
                 pol_w2=self.ppo.pol_w2, pol_b2=self.ppo.pol_b2,
                 val_w1=self.ppo.val_w1, val_b1=self.ppo.val_b1,
                 val_w2=self.ppo.val_w2, val_b2=self.ppo.val_b2)
        np.savez(f'{prefix}_cbf.npz',
                 W1=self.neural_cbf.W1, b1=self.neural_cbf.b1,
                 W2=self.neural_cbf.W2, b2=self.neural_cbf.b2,
                 W3=self.neural_cbf.W3, b3=self.neural_cbf.b3)


# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def train_marahs(n_episodes=1000, grid=30, n_drones=10, max_steps=150,
                 wind_curriculum=True, log_every=100):
    env = WildfireEnv(grid=grid, n_drones=n_drones, max_steps=max_steps)
    agent = MARAHSAgent(env, hidden=128, lr=3e-4)
    
    wind_levels = [5.0, 7.0, 10.0, 12.0, 15.0, 15.0, 18.0, 18.0, 20.0, 25.0]
    
    history = {'rewards': [], 'perimeters': [], 'safety': [], 'wind': [],
               'cbf_rate': [], 'info_bonus': []}
    best_reward = -1e9
    start = time.time()
    
    print(f"{'='*70}")
    print(f"MARAHS Training | {n_episodes} eps | {n_drones} drones | {grid}x{grid}")
    print(f"  GP Fire Front + Info-Theoretic + Neural-CBF + GAT")
    print(f"{'='*70}")
    
    # Warm-start Neural-CBF before training (need env reset for fire dist cache)
    env.reset(seed=0)
    print("\n  Warming up Neural-CBF from rule-based safety labels...")
    agent.warmstart_cbf(n_samples=500)
    
    for ep in range(n_episodes):
        # Wind curriculum
        if wind_curriculum:
            idx = min(len(wind_levels) - 1, ep // (n_episodes // len(wind_levels)))
            env.base_wind = wind_levels[idx]
        
        obs = env.reset(seed=ep)
        ep_reward = 0.0
        cbf_overrides_ep = 0
        info_bonus_sum = 0.0
        
        # Update GP at episode start
        agent.update_gp()
        agent.info_planner.invalidate_cache()
        
        for step in range(max_steps):
            # Update GP every 10 steps
            if step % 10 == 0 and step > 0:
                agent.update_gp()
                agent.info_planner.invalidate_cache()
            
            # Get ALL actions in one batched pass
            results = agent.get_all_actions(obs)
            
            # Build action array
            actions = np.zeros(n_drones, dtype=int)
            for i in range(n_drones):
                if i in results:
                    actions[i] = results[i]['action']
            
            # Step environment
            obs_new, rewards, dones, infos = env.step(actions)
            
            # Add info-theoretic bonus to rewards
            for i in results:
                if i < len(rewards) and env.drones[i]['alive']:
                    pos = env.drones[i]['pos']
                    ig = agent.info_planner.compute_reward_bonus(pos)
                    rewards[i] += ig * 2.0
                    info_bonus_sum += ig
            
            # Store all transitions
            agent.store_transitions(results, obs, rewards, dones)
            
            ep_reward += float(np.sum(rewards))
            cbf_overrides_ep += sum(1 for r in results.values() if r['cbf_override'])
            
            obs = obs_new
            if all(dones):
                break
        
        # Train PPO every episode
        agent.train_ppo()
        
        # Train Neural-CBF every 3 episodes (keeps it updated)
        if ep % 3 == 0:
            agent.train_neural_cbf()
        
        # Metrics
        pfr = infos[0].get('perimeter_frac', 0) if infos else 0
        alive = sum(1 for d in env.drones if d['alive'])
        safety = alive / n_drones * 100
        total_checks = max(1, sum(1 for r in results.values()))
        cbf_rate = cbf_overrides_ep / total_checks * 100
        
        history['rewards'].append(ep_reward)
        history['perimeters'].append(pfr)
        history['safety'].append(safety)
        history['wind'].append(env.base_wind)
        history['cbf_rate'].append(cbf_rate)
        history['info_bonus'].append(info_bonus_sum)
        
        if (ep + 1) % log_every == 0:
            r = np.mean(history['rewards'][-log_every:])
            p = np.mean(history['perimeters'][-log_every:])
            s = np.mean(history['safety'][-log_every:])
            c = np.mean(history['cbf_rate'][-log_every:])
            elapsed = time.time() - start
            eps_per_sec = (ep + 1) / elapsed
            print(f"Ep {ep+1:5d}/{n_episodes} | R:{r:8.1f} | Peri:{p:.2f}% | "
                  f"Safe:{s:.0f}% | CBF:{c:.0f}% | Wind:{env.base_wind:.0f} | "
                  f"t:{elapsed:.0f}s ({eps_per_sec:.2f} eps/s)")
            
            if r > best_reward:
                best_reward = r
                agent.save('marahs_best')
    
    agent.save('marahs_final')
    
    total_time = time.time() - start
    print(f"\nTraining complete in {total_time:.0f}s ({n_episodes / total_time:.2f} eps/s)")
    print(f"GP observations: {agent.gp_observations}")
    print(f"CBF interventions: {agent.cbf_interventions}/{agent.cbf_total_checks}")
    print(f"Final reward: {np.mean(history['rewards'][-100:]):.1f}")
    print(f"Final safety: {np.mean(history['safety'][-100:]):.1f}%")
    
    return agent, history


# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_marahs(agent, env, n_episodes=20):
    """Evaluate MARAHS agent (with all 4 innovations)."""
    results = []
    for ep in range(n_episodes):
        obs = env.reset(seed=ep + 5000)
        ep_reward = 0.0
        
        agent.update_gp()
        agent.info_planner.invalidate_cache()
        
        for step in range(env.max_steps):
            if step % 10 == 0 and step > 0:
                agent.update_gp()
                agent.info_planner.invalidate_cache()
            
            agent_results = agent.get_all_actions(obs, deterministic=True)
            actions = np.zeros(env.n_drones, dtype=int)
            for i in range(env.n_drones):
                if i in agent_results:
                    actions[i] = agent_results[i]['action']
            
            obs, rewards, dones, infos = env.step(actions)
            ep_reward += float(np.sum(rewards))
            if all(dones):
                break
        
        pfr = infos[0].get('perimeter_frac', 0) if infos else 0
        alive = sum(1 for d in env.drones if d['alive'])
        cells = max((len(d['visited']) for d in env.drones), default=0)
        alive_steps = max((d['alive_steps'] for d in env.drones), default=0)
        results.append({'reward': ep_reward, 'perimeter': pfr,
                        'safety': alive / env.n_drones * 100,
                        'cells': cells, 'alive_steps': alive_steps})
    
    m = {k: float(np.mean([r[k] for r in results])) for k in ['reward', 'perimeter', 'safety', 'cells', 'alive_steps']}
    m['perimeter_std'] = float(np.std([r['perimeter'] for r in results]))
    m['safety_std'] = float(np.std([r['safety'] for r in results]))
    return m


def evaluate_baseline(agent_fn, env, n_episodes=20, use_cbf=False):
    """Evaluate a baseline, optionally with the rule-based CBF."""
    cbf = NeuralCBFSafetyFilter(env) if use_cbf else None
    results = []
    for ep in range(n_episodes):
        obs = env.reset(seed=ep + 5000)
        agent = agent_fn()
        for step in range(env.max_steps):
            actions = np.zeros(env.n_drones, dtype=int)
            for i in range(env.n_drones):
                if env.drones[i]['alive']:
                    result = agent.act(obs[i])
                    action = int(result[0]) if isinstance(result, tuple) else int(result)
                    if cbf:
                        safe_action, _ = cbf.filter_action(i, env.drones[i]['pos'], action)
                        action = safe_action
                    actions[i] = action
            obs, rewards, dones, infos = env.step(actions)
            if all(dones):
                break
        pfr = infos[0].get('perimeter_frac', 0) if infos else 0
        alive = sum(1 for d in env.drones if d['alive'])
        cells = max((len(d['visited']) for d in env.drones), default=0)
        alive_steps = max((d['alive_steps'] for d in env.drones), default=0)
        results.append({'perimeter': pfr, 'safety': alive / env.n_drones * 100,
                        'cells': cells, 'alive_steps': alive_steps})
    m = {k: float(np.mean([r[k] for r in results])) for k in ['perimeter', 'safety', 'cells', 'alive_steps']}
    m['perimeter_std'] = float(np.std([r['perimeter'] for r in results]))
    m['safety_std'] = float(np.std([r['safety'] for r in results]))
    return m


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    total_start = time.time()
    
    print("=" * 70)
    print("MARAHS: Full Innovation Training Pipeline (Optimized)")
    print("=" * 70)
    
    # Quick speed test
    print("\n[0/4] Speed validation...")
    env_test = WildfireEnv(grid=30, n_drones=10, max_steps=50)
    agent_test = MARAHSAgent(env_test, hidden=128, lr=3e-4)
    env_test.reset(seed=0)
    agent_test.warmstart_cbf(n_samples=200)
    t0 = time.time()
    obs = env_test.reset(seed=0)
    for step in range(50):
        agent_test.update_gp() if step % 10 == 0 else None
        results = agent_test.get_all_actions(obs)
        actions = np.zeros(10, dtype=int)
        for i in results:
            actions[i] = results[i]['action']
        obs, _, dones, _ = env_test.step(actions)
        if all(dones):
            break
    dt = time.time() - t0
    print(f"  50 steps with 10 drones: {dt:.1f}s ({50/dt:.1f} steps/s)")
    print(f"  Estimated 1000 eps: {1000*150/50*dt/60:.0f} min")
    
    # Train
    print("\n[1/4] Training MARAHS agent (1000 episodes)...")
    agent, history = train_marahs(
        n_episodes=1000, grid=30, n_drones=10, max_steps=150,
        wind_curriculum=True, log_every=100
    )
    
    # Benchmark
    print("\n[2/4] Running benchmark...")
    env = WildfireEnv(grid=30, n_drones=10, max_steps=150)
    n_eval = 20
    
    all_results = {}
    all_results['MARAHS (PPO+CBF+GP+GAT)'] = evaluate_marahs(agent, env, n_episodes=n_eval)
    all_results['PID+CBF'] = evaluate_baseline(PIDAgent, env, n_episodes=n_eval, use_cbf=True)
    all_results['PID'] = evaluate_baseline(PIDAgent, env, n_episodes=n_eval, use_cbf=False)
    all_results['Greedy'] = evaluate_baseline(GreedyFireAgent, env, n_episodes=n_eval, use_cbf=False)
    all_results['Random'] = evaluate_baseline(RandomAgent, env, n_episodes=n_eval, use_cbf=False)
    
    # Print results
    print(f"\n{'='*90}")
    print(f"{'Method':<35} {'Safety':>8} {'Perimeter':>10} {'Coverage':>10} {'Alive':>8}")
    print(f"{'-'*90}")
    for name in ['MARAHS (PPO+CBF+GP+GAT)', 'PID+CBF', 'PID', 'Greedy', 'Random']:
        r = all_results[name]
        print(f"{name:<35} {r['safety']:>6.1f}% {r['perimeter']:>8.2f}% {r['cells']:>8.0f} {r['alive_steps']:>7.0f}")
    print(f"{'='*90}")
    
    # Generate figures
    print("\n[3/4] Generating figures...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        os.makedirs('figures', exist_ok=True)
        
        # Fig 1: Training curves
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        window = 50
        eps = np.arange(len(history['rewards']))
        
        for ax, key, label, color in [
            (axes[0,0], 'rewards', 'Episode Reward', 'blue'),
            (axes[0,1], 'perimeters', 'Perimeter %', 'green'),
            (axes[1,0], 'safety', 'Safety %', 'red'),
            (axes[1,1], 'cbf_rate', 'CBF Override %', 'orange'),
        ]:
            data = history[key]
            if len(data) >= window:
                smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
                ax.plot(eps[:len(smoothed)], smoothed, color=color, linewidth=1.5)
            else:
                ax.plot(eps, data, color=color, linewidth=1.5)
            ax.set_xlabel('Episode')
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('MARAHS Training Progress', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('figures/fig1_training.pdf', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Figure 1: Training curves saved")
        
        # Fig 2: Benchmark comparison
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        names = list(all_results.keys())
        short_names = ['MARAHS', 'PID+CBF', 'PID', 'Greedy', 'Random']
        colors = ['#2196F3', '#4CAF50', '#FF5722', '#FF9800', '#9E9E9E']
        
        metrics = [('safety', 'Safety Rate (%)', 100), ('perimeter', 'Perimeter Tracking (%)', None), ('cells', 'Cells Covered', None)]
        for ax, (key, ylabel, ylim) in zip(axes, metrics):
            vals = [all_results[n][key] for n in names]
            errs = [all_results[n].get(f'{key}_std', 0) for n in names]
            bars = ax.bar(range(len(names)), vals, yerr=errs, color=colors, capsize=3)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(short_names, rotation=30, ha='right', fontsize=9)
            ax.set_ylabel(ylabel)
            if ylim: ax.set_ylim(0, ylim)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('MARAHS vs Baselines (30x30, 10 drones)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('figures/fig2_benchmark.pdf', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Figure 2: Benchmark comparison saved")
        
        # Fig 3: Innovation ablation
        fig, ax = plt.subplots(figsize=(8, 5))
        ablation_names = ['MARAHS (All)', 'PID+CBF', 'PID only']
        ablation_safety = [
            all_results['MARAHS (PPO+CBF+GP+GAT)']['safety'],
            all_results['PID+CBF']['safety'],
            all_results['PID']['safety'],
        ]
        ablation_perimeter = [
            all_results['MARAHS (PPO+CBF+GP+GAT)']['perimeter'],
            all_results['PID+CBF']['perimeter'],
            all_results['PID']['perimeter'],
        ]
        x = np.arange(len(ablation_names))
        w = 0.35
        ax.bar(x - w/2, ablation_safety, w, label='Safety %', color='#4CAF50')
        ax.bar(x + w/2, ablation_perimeter, w, label='Perimeter %', color='#2196F3')
        ax.set_xticks(x)
        ax.set_xticklabels(ablation_names)
        ax.legend()
        ax.set_title('Ablation: Neural-CBF Impact')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('figures/fig3_cbf.pdf', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Figure 3: CBF ablation saved")
        
        # Fig 4: Information gain over time
        fig, ax = plt.subplots(figsize=(8, 5))
        ig = history['info_bonus']
        if len(ig) >= window:
            smoothed = np.convolve(ig, np.ones(window)/window, mode='valid')
            ax.plot(eps[:len(smoothed)], smoothed, color='purple', linewidth=1.5)
        else:
            ax.plot(eps, ig, color='purple', linewidth=1.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Mean Info Gain')
        ax.set_title('Information-Theoretic Sensing: Mutual Information')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('figures/fig4_info_gain.pdf', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Figure 4: Information gain saved")
        
    except ImportError as e:
        print(f"  matplotlib not available: {e}")
    
    # Save results
    print("\n[4/4] Saving results...")
    output = {
        'benchmark': all_results,
        'training': {
            'final_reward': float(np.mean(history['rewards'][-100:])),
            'final_perimeter': float(np.mean(history['perimeters'][-100:])),
            'final_safety': float(np.mean(history['safety'][-100:])),
            'gp_observations': agent.gp_observations,
            'cbf_interventions': agent.cbf_interventions,
            'cbf_total_checks': agent.cbf_total_checks,
        }
    }
    with open('marahs_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    t = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"DONE! Total time: {t:.0f}s ({t/60:.1f} min)")
    print(f"{'='*70}")
    print(f"Files generated:")
    print(f"  - marahs_best_ppo.npz, marahs_best_cbf.npz")
    print(f"  - marahs_final_ppo.npz, marahs_final_cbf.npz")
    print(f"  - marahs_results.json")
    print(f"  - figures/fig1_training.pdf through fig4_info_gain.pdf")
