#!/usr/bin/env python3
"""Generate publication figures for GAT-MARAHS."""
import numpy as np
import json
import os

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Publication-quality settings
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.linewidth'] = 0.8
rcParams['figure.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.1

os.makedirs('figures_gat', exist_ok=True)


def fig1_training_curves():
    """Figure 1: GAT-MARAHS training curves (coverage + safety)."""
    with open('gat_marahs_results.json') as f:
        d = json.load(f)
    
    rewards = d['rewards']
    coverages = d['coverages']
    safety = d['safety']
    
    # Smooth with moving average
    def smooth(x, w=20):
        return np.convolve(x, np.ones(w)/w, mode='valid')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Coverage
    cov_smooth = smooth(coverages)
    ax1.plot(range(len(cov_smooth)), cov_smooth, 'b-', linewidth=1.5, label='Coverage')
    ax1.fill_between(range(len(cov_smooth)),
                     np.clip(cov_smooth - np.std(coverages[:len(cov_smooth)]), 0, 100),
                     np.clip(cov_smooth + np.std(coverages[:len(cov_smooth)]), 0, 100),
                     alpha=0.2, color='blue')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Grid Coverage (%)')
    ax1.set_title('(a) Exploration Coverage')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Safety
    saf_smooth = smooth(safety)
    ax2.plot(range(len(saf_smooth)), saf_smooth, 'r-', linewidth=1.5, label='Safety')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Safety Rate (%)')
    ax2.set_title('(b) Drone Safety')
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures_gat/fig1_training.pdf')
    plt.savefig('figures_gat/fig1_training.png')
    plt.close()
    print("  ✓ Figure 1: Training curves")


def fig2_benchmark():
    """Figure 2: Benchmark comparison bar chart."""
    with open('gat_benchmark.json') as f:
        data = json.load(f)
    
    methods = list(data.keys())
    safety = [data[m]['safety'] for m in methods]
    coverage = [data[m]['coverage'] for m in methods]
    perimeter = [data[m]['perimeter'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width, safety, width, label='Safety (%)', color='#2ecc71', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, coverage, width, label='Coverage (%)', color='#3498db', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, perimeter, width, label='Perimeter (%)', color='#e74c3c', edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Performance (%)')
    ax.set_title('GAT-MARAHS Benchmark (20×20 grid, 6 drones, wind=12 m/s)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('figures_gat/fig2_benchmark.pdf')
    plt.savefig('figures_gat/fig2_benchmark.png')
    plt.close()
    print("  ✓ Figure 2: Benchmark comparison")


def fig3_scaling():
    """Figure 3: Scalability test (5, 10, 20, 50 drones)."""
    # Run quick scalability test
    from paper_ready_train import WildfireEnv
    from train_gat_fast import FastGATPPO
    
    agent = FastGATPPO(obs_dim=656, act_dim=5)
    if os.path.exists('gat_marahs_best.pt'):
        agent.load('gat_marahs_best.pt')
    
    drone_counts = [4, 6, 10, 15]
    safety_vals = []
    coverage_vals = []
    perimeter_vals = []
    times = []
    
    for n_d in drone_counts:
        s_list, c_list, p_list = [], [], []
        t0 = __import__('time').time()
        for _ in range(10):
            env = WildfireEnv(grid=20, n_drones=n_d, max_steps=100, wind_speed=12.0)
            obs = env.reset()
            for step in range(100):
                am = np.array([env.drones[i]['alive'] for i in range(n_d)])
                pos = np.array([env.drones[i]['pos'] for i in range(n_d)], dtype=np.float32)
                if not am.any(): break
                acts, _, _, _ = agent.select_actions(obs, pos, am)
                obs, _, dones, _ = env.step(np.array(acts, dtype=np.int32))
                if all(dones): break
            ac = sum(1 for i in range(n_d) if env.drones[i]['alive'])
            s_list.append(ac / n_d * 100)
            c_list.append(len(env.total_cells_explored) / 400 * 100)
            fc = np.argwhere(env.fire > 0.2)
            pc = set()
            for fx, fy in fc:
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = fx+dx, fy+dy
                    if 0 <= nx < 20 and 0 <= ny < 20 and env.fire[nx, ny] < 0.1:
                        pc.add((nx, ny))
            vis = set()
            for i in range(n_d): vis.update(env.drones[i].get('visited', set()))
            p_list.append(len(pc & vis) / max(1, len(pc)) * 100)
        
        t_elapsed = __import__('time').time() - t0
        safety_vals.append(np.mean(s_list))
        coverage_vals.append(np.mean(c_list))
        perimeter_vals.append(np.mean(p_list))
        times.append(t_elapsed / 10)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.plot(drone_counts, safety_vals, 'go-', linewidth=2, markersize=8, label='Safety')
    ax1.plot(drone_counts, coverage_vals, 'bs-', linewidth=2, markersize=8, label='Coverage')
    ax1.plot(drone_counts, perimeter_vals, 'r^-', linewidth=2, markersize=8, label='Perimeter')
    ax1.set_xlabel('Number of Drones')
    ax1.set_ylabel('Performance (%)')
    ax1.set_title('(a) Performance vs Swarm Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(drone_counts, times, 'ko-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Drones')
    ax2.set_ylabel('Time per Episode (s)')
    ax2.set_title('(b) Computational Scaling')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures_gat/fig3_scaling.pdf')
    plt.savefig('figures_gat/fig3_scaling.png')
    plt.close()
    print("  ✓ Figure 3: Scalability")


def fig4_gat_attention():
    """Figure 4: GAT attention visualization (what drones communicate about)."""
    import torch
    from gat_communication import GATCommunication
    
    gat = GATCommunication(656, 128, 64, 4, 15.0)
    if os.path.exists('gat_marahs_best.pt'):
        ckpt = torch.load('gat_marahs_best.pt', map_location='cpu')
        if 'gat' in ckpt:
            gat.load_state_dict(ckpt['gat'])
    
    # Simulate 6 drones in two clusters
    positions = torch.tensor([
        [5.0, 5.0], [6.0, 6.0], [7.0, 5.0],  # Cluster A
        [15.0, 15.0], [16.0, 16.0], [17.0, 15.0],  # Cluster B
    ], dtype=torch.float32)
    alive = torch.ones(6, dtype=torch.bool)
    
    obs = torch.randn(6, 656) * 0.1
    obs[:3, 100] = 0.8  # Fire near cluster A
    obs[3:, 200] = 0.3  # Less fire near cluster B
    
    adj = gat.build_graph(positions, alive)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    colors = ['#e74c3c', '#e74c3c', '#e74c3c', '#3498db', '#3498db', '#3498db']
    labels = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3']
    
    for i in range(6):
        for j in range(6):
            if adj[i, j] and i != j:
                ax.plot([positions[i, 0], positions[j, 0]],
                       [positions[i, 1], positions[j, 1]],
                       'gray', linewidth=0.5, alpha=0.5)
    
    for i in range(6):
        ax.scatter(positions[i, 0], positions[i, 1], c=colors[i], s=200, zorder=5,
                  edgecolors='black', linewidth=1)
        ax.annotate(labels[i], (positions[i, 0], positions[i, 1]),
                   textcoords="offset points", xytext=(0, 12), ha='center', fontsize=10)
    
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('GAT Communication Graph (comm_range=15)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Legend
    ax.scatter([], [], c='#e74c3c', s=100, label='Cluster A (fire nearby)')
    ax.scatter([], [], c='#3498db', s=100, label='Cluster B (no fire)')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('figures_gat/fig4_gat_graph.pdf')
    plt.savefig('figures_gat/fig4_gat_graph.png')
    plt.close()
    print("  ✓ Figure 4: GAT communication graph")


if __name__ == "__main__":
    print("=" * 60)
    print("Generating GAT-MARAHS Publication Figures")
    print("=" * 60)
    
    fig1_training_curves()
    fig2_benchmark()
    fig3_scaling()
    fig4_gat_attention()
    
    print("\n" + "=" * 60)
    print(f"All figures saved to figures_gat/")
    print("=" * 60)
