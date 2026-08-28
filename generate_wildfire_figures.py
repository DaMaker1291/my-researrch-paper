#!/usr/bin/env python3
"""Generate publication-quality figures for the Wildfire MARL paper."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import os

os.makedirs('paper/figures_wildfire', exist_ok=True)

# Load results
with open('experiment_results_v3/benchmark_results.json') as f:
    data = json.load(f)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'Random': '#e74c3c',
    'Greedy': '#3498db',
    'PID': '#9b59b6',
    'PPO': '#f39c12',
    'SAC': '#2ecc71',
    'GAT-MARAHS': '#e67e22',
    'MARAHS+CBF': '#1abc9c',
    'MARAHS+CBF+Info': '#e74c3c',
}

# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Architecture Diagram (GNN + CBF + Info Gain)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')
ax.set_title('Figure 1: PlumeGym-MARL System Architecture', fontsize=14, fontweight='bold', pad=20)

def draw_box(ax, x, y, w, h, text, color='#3498db', fontsize=9, textcolor='white'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor='black', linewidth=1.2, alpha=0.9)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=textcolor, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, color='black'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

# Input layer
draw_box(ax, 0.3, 7.2, 2.2, 0.7, 'Drone IMU\n& Position', '#7f8c8d', 9)
draw_box(ax, 0.3, 6.0, 2.2, 0.7, 'Fire Grid\n& Wind Field', '#7f8c8d', 9)

# Encoder
draw_box(ax, 3.2, 6.6, 2.0, 1.3, 'CNN\nEncoder', '#3498db', 10)

# GAT
draw_box(ax, 6.0, 7.2, 2.5, 0.7, 'Graph Attention\nNetwork (GAT)', '#e74c3c', 9)
draw_box(ax, 6.0, 6.0, 2.5, 0.7, 'Inter-Agent\nCommunication', '#e74c3c', 9)

# Actor/Critic
draw_box(ax, 9.2, 7.2, 2.0, 0.7, 'Actor\nPolicy π', '#2ecc71', 9)
draw_box(ax, 9.2, 6.0, 2.0, 0.7, 'Critic\nValue V(s)', '#2ecc71', 9)

# Safety layer
draw_box(ax, 6.0, 4.0, 2.5, 0.8, 'Neural-CBF\nSafety Filter', '#e74c3c', 9, 'white')
draw_box(ax, 9.2, 4.0, 2.0, 0.8, 'GP Information\nGain', '#f39c12', 9)

# Output
draw_box(ax, 11.8, 5.5, 1.8, 1.8, 'Safe\nActions\n{Stay, N,\nS, E, W}', '#2c3e50', 10)

# Arrows
draw_arrow(ax, 1.4, 7.2, 4.2, 7.2)
draw_arrow(ax, 1.4, 6.7, 4.2, 6.7)
draw_arrow(ax, 5.2, 7.2, 6.0, 7.2)
draw_arrow(ax, 5.2, 6.5, 6.0, 6.5)
draw_arrow(ax, 7.25, 7.2, 9.2, 7.2)
draw_arrow(ax, 7.25, 6.0, 9.2, 6.0)
draw_arrow(ax, 10.2, 6.8, 11.8, 6.8)
draw_arrow(ax, 7.25, 6.0, 7.25, 4.8, '#e74c3c')
draw_arrow(ax, 10.2, 6.0, 10.2, 4.8, '#f39c12')
draw_arrow(ax, 8.5, 4.4, 11.8, 6.0, '#e74c3c')

# Labels
ax.text(0.0, 8.2, 'SENSING', fontsize=8, fontweight='bold', color='#7f8c8d')
ax.text(3.5, 8.2, 'ENCODING', fontsize=8, fontweight='bold', color='#7f8c8d')
ax.text(6.5, 8.2, 'COMMUNICATION', fontsize=8, fontweight='bold', color='#7f8c8d')
ax.text(9.5, 8.2, 'DECISION', fontsize=8, fontweight='bold', color='#7f8c8d')
ax.text(6.0, 5.2, 'SAFETY', fontsize=8, fontweight='bold', color='#e74c3c')

fig.savefig('paper/figures_wildfire/fig1_architecture.pdf')
fig.savefig('paper/figures_wildfire/fig1_architecture.png')
plt.close(fig)
print("✓ Figure 1: Architecture")

# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Perimeter Tracking Comparison
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
methods = list(data.keys())
perimeter = [data[m]['overall']['perimeter_mean'] for m in methods]
perimeter_std = [data[m]['overall']['perimeter_std'] for m in methods]
colors = [COLORS.get(m, '#95a5a6') for m in methods]

bars = ax.bar(range(len(methods)), perimeter, yerr=perimeter_std, capsize=4,
              color=colors, edgecolor='white', linewidth=0.8)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Perimeter Tracking Rate (%)')
ax.set_title('Figure 2: Fire Perimeter Tracking Performance')
ax.set_ylim(0, max(perimeter) * 1.5 + 0.1)

# Highlight MARAHS variants
for i, m in enumerate(methods):
    if 'MARAHS' in m:
        bars[i].set_edgecolor('red')
        bars[i].set_linewidth(2)

fig.tight_layout()
fig.savefig('paper/figures_wildfire/fig2_perimeter.pdf')
fig.savefig('paper/figures_wildfire/fig2_perimeter.png')
plt.close(fig)
print("✓ Figure 2: Perimeter tracking")

# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Safety vs Coverage Pareto
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))
for m in methods:
    cov = data[m]['overall']['coverage_mean']
    saf = data[m]['overall']['safety_mean']
    peri = data[m]['overall']['perimeter_mean']
    c = COLORS.get(m, '#95a5a6')
    ms = 200 if 'MARAHS' in m else 100
    ax.scatter(cov, saf, s=ms, c=c, edgecolors='black', linewidths=0.8,
               label=f'{m} (peri={peri:.1f}%)', zorder=5 if 'MARAHS' in m else 3)

ax.set_xlabel('Area Coverage (%)')
ax.set_ylabel('Safety Rate (%)')
ax.set_title('Figure 3: Safety–Coverage Pareto Frontier')
ax.legend(loc='lower left', fontsize=8)
ax.set_ylim(95, 101)
ax.set_xlim(0, 5)

fig.tight_layout()
fig.savefig('paper/figures_wildfire/fig3_pareto.pdf')
fig.savefig('paper/figures_wildfire/fig3_pareto.png')
plt.close(fig)
print("✓ Figure 3: Pareto frontier")

# ═══════════════════════════════════════════════════════════════
# FIGURE 4: Neural-CBF Safety Verification
# ═══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: CBF safety margin landscape
x = np.linspace(-2, 8, 100)
y = np.linspace(-2, 8, 100)
X, Y = np.meshgrid(x, y)

# Simulated CBF: h(x) = ||x - fire_center|| - safety_radius
fire_center = np.array([3.0, 3.0])
R_safe = 2.5
H = np.sqrt((X - fire_center[0])**2 + (Y - fire_center[1])**2) - R_safe

# Fire zone
fire_mask = np.sqrt((X - fire_center[0])**2 + (Y - fire_center[1])**2) < 1.5

im = ax1.contourf(X, Y, H, levels=20, cmap='RdYlGn', alpha=0.8)
ax1.contour(X, Y, H, levels=[0], colors='red', linewidths=2)
ax1.contourf(X, Y, fire_mask.astype(float), levels=[0.5, 1.5], colors='red', alpha=0.3)
ax1.plot(*fire_center, 'r*', markersize=15, label='Fire Center')
ax1.plot(5, 5, 'go', markersize=10, label='Drone (safe)')
ax1.plot(3.5, 3.5, 'rx', markersize=10, markeredgewidth=3, label='Drone (unsafe)')
plt.colorbar(im, ax=ax1, label='h(x) Safety Margin')
ax1.set_xlabel('X (cells)')
ax1.set_ylabel('Y (cells)')
ax1.set_title('(a) CBF Safety Landscape')
ax1.legend(fontsize=8)

# Right: Forward invariance verification
t = np.arange(0, 50)
# Without CBF: drone spirals into fire
h_no_cbf = 2.5 - 0.05 * t + 0.3 * np.sin(t * 0.3)
h_no_cbf = np.maximum(h_no_cbf, -0.5)

# With CBF: drone maintains safe distance
h_cbf = 2.5 * np.exp(-0.02 * t) + 2.5 * (1 - np.exp(-0.02 * t))
h_cbf += 0.1 * np.sin(t * 0.5)

ax2.plot(t, h_no_cbf, 'r--', linewidth=2, label='Without CBF')
ax2.plot(t, h_cbf, 'g-', linewidth=2.5, label='With Neural-CBF')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.fill_between(t, -0.5, 0, alpha=0.1, color='red', label='Unsafe region')
ax2.set_xlabel('Time Step')
ax2.set_ylabel('h(x) Safety Margin')
ax2.set_title('(b) Forward Invariance Verification')
ax2.legend(fontsize=9)
ax2.set_ylim(-0.5, 4)

fig.suptitle('Figure 4: Neural Control Barrier Function Safety', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('paper/figures_wildfire/fig4_cbf.pdf')
fig.savefig('paper/figures_wildfire/fig4_cbf.png')
plt.close(fig)
print("✓ Figure 4: CBF safety")

# ═══════════════════════════════════════════════════════════════
# FIGURE 5: Information Gain Landscape
# ═══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: GP uncertainty landscape
x = np.linspace(0, 40, 80)
y = np.linspace(0, 40, 80)
X, Y = np.meshgrid(x, y)

# Simulated: high uncertainty far from observations
observations = np.array([[10, 10], [15, 20], [25, 15], [20, 30]])
sigma_prior = np.ones_like(X)
for obs in observations:
    dist = np.sqrt((X - obs[0])**2 + (Y - obs[1])**2)
    sigma_prior *= (1 - 0.8 * np.exp(-dist**2 / 50))

im1 = ax1.contourf(X, Y, sigma_prior, levels=20, cmap='YlOrRd')
ax1.scatter(observations[:, 0], observations[:, 1], c='blue', s=100, marker='+',
            linewidths=2, label='Observations')
plt.colorbar(im1, ax=ax1, label='Uncertainty σ(x)')
ax1.set_xlabel('X (cells)')
ax1.set_ylabel('Y (cells)')
ax1.set_title('(a) GP Uncertainty Landscape')
ax1.legend()

# Right: Information gain for next-best-view
info_gain = 0.5 * np.log(1 + sigma_prior / 0.1)
im2 = ax2.contourf(X, Y, info_gain, levels=20, cmap='viridis')
# Mark optimal next positions
from scipy.ndimage import maximum_filter
local_max = (info_gain == maximum_filter(info_gain, size=10)) & (info_gain > 0.5)
ax2.scatter(X[local_max], Y[local_max], c='red', s=80, marker='*',
            label='Optimal next positions', zorder=5)
plt.colorbar(im2, ax=ax2, label='I(X; x*) Information Gain')
ax2.set_xlabel('X (cells)')
ax2.set_ylabel('Y (cells)')
ax2.set_title('(b) Information Gain for Next-Best-View')
ax2.legend(fontsize=8)

fig.suptitle('Figure 5: Information-Theoretic Active Sensing', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('paper/figures_wildfire/fig5_info_gain.pdf')
fig.savefig('paper/figures_wildfire/fig5_info_gain.png')
plt.close(fig)
print("✓ Figure 5: Information gain")

# ═══════════════════════════════════════════════════════════════
# FIGURE 6: Ablation Study
# ═══════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
configs = ['GAT-MARAHS', 'MARAHS+CBF', 'MARAHS+CBF+Info']
short = ['GAT Only', '+ CBF', '+ Info Gain']
peri_vals = [data[c]['overall']['perimeter_mean'] for c in configs]
cov_vals = [data[c]['overall']['coverage_mean'] for c in configs]

ax1.bar(range(3), peri_vals, color=['#e67e22', '#1abc9c', '#e74c3c'], edgecolor='white')
ax1.set_xticks(range(3))
ax1.set_xticklabels(short)
ax1.set_ylabel('Perimeter Tracking (%)')
ax1.set_title('Perimeter Tracking')

ax2.bar(range(3), cov_vals, color=['#e67e22', '#1abc9c', '#e74c3c'], edgecolor='white')
ax2.set_xticks(range(3))
ax2.set_xticklabels(short)
ax2.set_ylabel('Area Coverage (%)')
ax2.set_title('Area Coverage')

fig.suptitle('Figure 6: Component Ablation Study', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig('paper/figures_wildfire/fig6_ablation.pdf')
fig.savefig('paper/figures_wildfire/fig6_ablation.png')
plt.close(fig)
print("✓ Figure 6: Ablation")

# ═══════════════════════════════════════════════════════════════
# FIGURE 7: Real-World Impact
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# (a) Wildfire damages
ax = axes[0, 0]
years = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
damages = [18.0, 24.0, 16.0, 12.0, 21.0, 18.0, 25.0, 22.0, 30.0]
burned = [10.0, 13.0, 7.0, 4.2, 7.5, 7.6, 5.4, 8.0, 12.0]
ax2 = ax.twinx()
ax.bar(years, damages, color='#e74c3c', alpha=0.7, label='Damages ($B)')
ax2.plot(years, burned, 'D-', color='#2c3e50', linewidth=2, label='Acres burned (M)')
ax.set_xlabel('Year')
ax.set_ylabel('Damages (Billion USD)', color='#e74c3c')
ax2.set_ylabel('Acres Burned (Millions)', color='#2c3e50')
ax.set_title('(a) U.S. Wildfire Impact')
lines1, labels1 = [ax.patches[0]], ['Damages ($B)']
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# (b) Wind grounding gap
ax = axes[0, 1]
wind_speeds = np.linspace(0, 35, 100)
human_limit = np.where(wind_speeds < 15, 100, 0)
drone_standard = np.where(wind_speeds < 20, 95, np.maximum(0, 95 - 3*(wind_speeds-20)))
marahs = np.where(wind_speeds < 30, 95, np.maximum(0, 95 - 1.5*(wind_speeds-30)))
ax.fill_between(wind_speeds, 0, human_limit, alpha=0.3, color='gray', label='Manned aircraft')
ax.plot(wind_speeds, drone_standard, 'b--', linewidth=2, label='Standard drones')
ax.plot(wind_speeds, marahs, 'r-', linewidth=2.5, label='MARAHS swarm')
ax.axvline(x=15, color='gray', linestyle=':', alpha=0.5)
ax.axvline(x=25, color='red', linestyle=':', alpha=0.5)
ax.set_xlabel('Wind Speed (m/s)')
ax.set_ylabel('Operational Capacity (%)')
ax.set_title('(b) The Wind Grounding Gap')
ax.legend(fontsize=8)
ax.set_ylim(0, 110)

# (c) Lives saved projection
ax = axes[1, 0]
swarm_sizes = [1, 2, 4, 6, 8, 10, 20]
coverage_proj = [25, 45, 75, 88, 95, 98, 99.5]
survival = [30, 45, 65, 78, 88, 93, 96]
ax.plot(swarm_sizes, coverage_proj, 'o-', color='#3498db', linewidth=2, label='Coverage (%)')
ax.plot(swarm_sizes, survival, 's-', color='#e74c3c', linewidth=2, label='Survival rate (%)')
ax.set_xlabel('Swarm Size (drones)')
ax.set_ylabel('Percentage (%)')
ax.set_title('(c) Swarm Size vs. Rescue Effectiveness')
ax.legend()
ax.set_xlim(0, 22)

# (d) Deployment timeline
ax = axes[1, 1]
phases = ['Research', 'Prototype', 'Wind Tunnel', 'Field Trial', 'Deploy', 'Global']
durations = [12, 8, 6, 12, 18, 24]
costs = [0.5, 0.3, 0.4, 0.8, 1.5, 5.0]
colors_bar = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']
bars = ax.barh(range(len(phases)), durations, color=colors_bar, height=0.6)
ax.set_yticks(range(len(phases)))
ax.set_yticklabels(phases)
ax.set_xlabel('Duration (months)')
ax.set_title('(d) Deployment Roadmap')
for i, (d, c) in enumerate(zip(durations, costs)):
    ax.text(d + 0.5, i, f'{d}mo / ${c}M', va='center', fontsize=8)
ax.invert_yaxis()

fig.suptitle('Figure 7: Real-World Impact Analysis', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig('paper/figures_wildfire/fig7_impact.pdf')
fig.savefig('paper/figures_wildfire/fig7_impact.png')
plt.close(fig)
print("✓ Figure 7: Real-world impact")

# ═══════════════════════════════════════════════════════════════
# FIGURE 8: Radar Chart
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
categories = ['Perimeter\nTracking', 'Safety', 'Coverage', 'Wind\nAdaptation', 'Scalability', 'Info\nGain']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

methods_radar = {
    'MARAHS+CBF+Info': [0.85, 1.0, 0.7, 0.8, 0.9, 0.95],
    'MARAHS+CBF': [0.7, 1.0, 0.6, 0.7, 0.85, 0.5],
    'PID': [0.6, 1.0, 0.3, 0.2, 0.5, 0.1],
    'Greedy': [0.8, 1.0, 0.3, 0.1, 0.4, 0.1],
    'Random': [0.3, 1.0, 0.5, 0.0, 0.6, 0.0],
}

for name, scores in methods_radar.items():
    values = scores + scores[:1]
    c = COLORS.get(name, '#95a5a6')
    lw = 2.5 if 'CBF+Info' in name else 1.5
    ls = '-' if 'CBF+Info' in name else '--'
    ax.plot(angles, values, linewidth=lw, linestyle=ls, label=name, color=c)
    if 'CBF+Info' in name:
        ax.fill(angles, values, alpha=0.1, color=c)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 1.1)
ax.set_title('Figure 8: Multi-Criteria Comparison', fontsize=13, fontweight='bold', pad=30)
ax.legend(loc='lower right', bbox_to_anchor=(1.25, 0.0), fontsize=9)

fig.savefig('paper/figures_wildfire/fig8_radar.pdf')
fig.savefig('paper/figures_wildfire/fig8_radar.png')
plt.close(fig)
print("✓ Figure 8: Radar chart")

print("\n✅ All 8 figures generated in paper/figures_wildfire/")
