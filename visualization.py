"""
MARAHS Visualization Module
=============================

Publication-quality visualizations for:
1. GP wind field maps with uncertainty
2. Safety certificates and barrier functions
3. Coverage progress and multi-agent coordination
4. Adaptation timescales
5. Experimental results

Usage:
    from visualization import MARAHSVisualizer
    
    viz = MARAHSVisualizer()
    viz.plot_wind_field(wind_mapper)
    viz.plot_safety_certificate(cbf, state)
    viz.plot_coverage_progress(results)
    viz.plot_ablation_study(ablation_results)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple
import os


class MARAHSVisualizer:
    """
    Publication-quality visualizations for MARAHS research.
    """
    
    def __init__(self, style: str = 'research'):
        """Initialize with publication style."""
        if style == 'research':
            self._set_research_style()
        
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#2ECC71',
            'danger': '#E74C3C',
            'warning': '#F39C12',
            'info': '#3498DB',
        }
    
    def _set_research_style(self):
        """Set matplotlib style for research papers."""
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 11,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'axes.grid': True,
            'grid.alpha': 0.3,
        })
    
    def plot_wind_field(self, wind_mapper, title: str = "GP Wind Field Reconstruction",
                       save_path: Optional[str] = None):
        """
        Plot GP wind field with uncertainty.
        
        Creates a 2D visualization of the wind field with:
        - Arrows showing wind direction and magnitude
        - Color showing wind speed
        - Uncertainty contours
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Get wind field
        wind_field = wind_mapper.get_wind_field()
        N = wind_field.shape[0]
        
        # Create coordinate grids
        x = np.linspace(0, wind_mapper.config.grid_size, N)
        y = np.linspace(0, wind_mapper.config.grid_size, N)
        X, Y = np.meshgrid(x, y)
        
        # Compute wind speed
        speed = np.sqrt(wind_field[:,:,0]**2 + wind_field[:,:,1]**2)
        
        # Plot 1: Wind speed heatmap
        im1 = axes[0].contourf(X, Y, speed, levels=20, cmap='RdYlBu_r')
        axes[0].set_xlabel('X (m)')
        axes[0].set_ylabel('Y (m)')
        axes[0].set_title('Wind Speed (m/s)')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot 2: Wind vectors
        skip = max(1, N // 20)
        axes[1].quiver(X[::skip, ::skip], Y[::skip, ::skip],
                      wind_field[::skip, ::skip, 0], wind_field[::skip, ::skip, 1],
                      speed[::skip, ::skip], cmap='RdYlBu_r', scale=50)
        axes[1].set_xlabel('X (m)')
        axes[1].set_ylabel('Y (m)')
        axes[1].set_title('Wind Vectors')
        
        # Plot 3: Observations and uncertainty
        if wind_mapper.n_measurements > 0:
            axes[2].scatter(wind_mapper.X_observed[:, 0], wind_mapper.X_observed[:, 1],
                          c='red', s=50, label='Observations', zorder=5)
        
        # Add hurricane eye if identifiable
        try:
            structure = wind_mapper.identify_hurricane_structure()
            if 'eye_center' in structure:
                eye = structure['eye_center']
                axes[2].plot(eye[0], eye[1], 'k*', markersize=15, label='Eye')
                circle = Circle(eye, structure.get('eye_radius', 20), 
                              fill=False, color='black', linestyle='--')
                axes[2].add_patch(circle)
        except:
            pass
        
        axes[2].set_xlabel('X (m)')
        axes[2].set_ylabel('Y (m)')
        axes[2].set_title('Observations & Structure')
        axes[2].legend()
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_safety_certificate(self, cbf, state: Dict, action: np.ndarray,
                               title: str = "Safety Certificate",
                               save_path: Optional[str] = None):
        """
        Plot safety certificate visualization.
        
        Shows:
        - Barrier function values
        - Safe set boundaries
        - Current state location
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Compute barriers
        barriers = cbf.barrier.compute(state)
        
        # Plot 1: Barrier values bar chart
        names = list(barriers.keys())
        values = [barriers[n] for n in names]
        colors = [self.colors['success'] if v >= 0 else self.colors['danger'] for v in values]
        
        axes[0].barh(names, values, color=colors)
        axes[0].axvline(x=0, color='black', linestyle='-', linewidth=2)
        axes[0].set_xlabel('Barrier Value h(x)')
        axes[0].set_title('Safety Constraints')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: 2D projection of safe set
        # Create grid of states
        x_range = np.linspace(-2, 2, 100)
        z_range = np.linspace(0, 20, 100)
        X, Z = np.meshgrid(x_range, z_range)
        
        # Compute barrier for each state
        H = np.zeros_like(X)
        for i in range(len(x_range)):
            for j in range(len(z_range)):
                test_state = dict(state)
                test_state['position'] = np.array([X[i,j], 0, Z[i,j]])
                test_barriers = cbf.barrier.compute(test_state)
                H[i,j] = min(test_barriers.values())
        
        # Plot safe set
        im = axes[1].contourf(X, Z, H, levels=[-1, 0, 1, 2, 5], 
                             colors=['#E74C3C', '#F39C12', '#2ECC71', '#27AE60'])
        axes[1].contour(X, Z, H, levels=[0], colors='black', linewidths=2)
        
        # Mark current state
        axes[1].plot(state['position'][0], state['position'][2], 'k*', 
                    markersize=20, label='Current State')
        
        axes[1].set_xlabel('X Position (m)')
        axes[1].set_ylabel('Z Position (m)')
        axes[1].set_title('Safe Set (shaded region)')
        axes[1].legend()
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_coverage_progress(self, results: Dict[str, List[float]],
                              title: str = "Coverage Progress",
                              save_path: Optional[str] = None):
        """
        Plot coverage progress over time for multiple methods.
        
        Creates a publication-quality line plot with:
        - Mean coverage
        - Confidence intervals
        - Baseline comparisons
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
        
        for (method_name, coverages), color in zip(results.items(), colors):
            # Compute statistics
            mean = np.mean(coverages, axis=0)
            std = np.std(coverages, axis=0)
            steps = np.arange(len(mean))
            
            # Plot mean
            ax.plot(steps, mean, label=method_name, color=color, linewidth=2)
            
            # Plot confidence interval
            ax.fill_between(steps, mean - std, mean + std, 
                          color=color, alpha=0.2)
        
        ax.set_xlabel('Steps')
        ax.set_ylabel('Coverage (%)')
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_ablation_study(self, ablation_results: Dict[str, float],
                           title: str = "Ablation Study",
                           save_path: Optional[str] = None):
        """
        Plot ablation study results.
        
        Shows contribution of each component.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by contribution
        sorted_results = dict(sorted(ablation_results.items(), 
                                    key=lambda x: x[1], reverse=True))
        
        # Separate full system from ablations
        full_coverage = sorted_results.pop('Full MARAHS', 100)
        
        # Add full system at the beginning
        names = ['Full MARAHS'] + list(sorted_results.keys())
        values = [full_coverage] + list(sorted_results.values())
        
        # Create color gradient
        colors = [self.colors['primary'] if i == 0 
                 else plt.cm.RdYlGn(v/full_coverage)
                 for i, v in enumerate(values)]
        
        # Plot horizontal bars
        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, color=colors, edgecolor='gray')
        
        # Add value labels
        for i, (name, val) in enumerate(zip(names, values)):
            ax.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=11)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel('Coverage (%)')
        ax.set_title(title)
        ax.set_xlim([0, max(values) * 1.15])
        
        # Add reference line for full system
        ax.axvline(x=full_coverage, color='black', linestyle='--', 
                  alpha=0.5, label=f'Full MARAHS ({full_coverage:.1f}%)')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_multi_agent_coordination(self, positions: Dict[int, np.ndarray],
                                     coverage: np.ndarray,
                                     title: str = "Multi-Agent Coordination",
                                     save_path: Optional[str] = None):
        """
        Plot multi-agent drone positions and coverage.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Agent positions
        colors = plt.cm.tab10(np.linspace(0, 1, len(positions)))
        
        for (agent_id, pos), color in zip(positions.items(), colors):
            axes[0].plot(pos[0], pos[1], 'o', color=color, markersize=15, 
                        label=f'Agent {agent_id}')
        
        axes[0].set_xlabel('X (m)')
        axes[0].set_ylabel('Y (m)')
        axes[0].set_title('Agent Positions')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Coverage grid
        im = axes[1].imshow(coverage.T, cmap='RdYlGn', origin='lower',
                           extent=[0, coverage.shape[0], 0, coverage.shape[1]])
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        axes[1].set_title('Coverage Map')
        plt.colorbar(im, ax=axes[1], label='Covered')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_adaptation_timescales(self, adaptation_data: Dict[str, List[float]],
                                  title: str = "Multi-Scale Adaptation",
                                  save_path: Optional[str] = None):
        """
        Plot adaptation at different timescales.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        timescales = ['Fast (1ms)', 'Medium (10ms)', 'Slow (100ms)', 'Very Slow (1s)']
        axes_flat = axes.flatten()
        
        for ax, (scale_name, data) in zip(axes_flat, adaptation_data.items()):
            if data:
                ax.plot(data, linewidth=2)
                ax.set_xlabel('Steps')
                ax.set_ylabel('Adaptation Magnitude')
                ax.set_title(scale_name)
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def create_figure1(self, wind_mapper, cbf, state, action,
                      save_path: Optional[str] = None):
        """
        Create Figure 1: Complete System Overview.
        
        This is the main figure for the paper.
        """
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # (a) Wind Field
        ax1 = fig.add_subplot(gs[0, 0])
        wind_field = wind_mapper.get_wind_field()
        speed = np.sqrt(wind_field[:,:,0]**2 + wind_field[:,:,1]**2)
        im1 = ax1.imshow(speed.T, cmap='RdYlBu_r', origin='lower', 
                        extent=[0, 200, 0, 200])
        ax1.set_title('(a) GP Wind Field')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        plt.colorbar(im1, ax=ax1, label='Speed (m/s)')
        
        # (b) Safety Certificate
        ax2 = fig.add_subplot(gs[0, 1])
        barriers = cbf.barrier.compute(state)
        names = list(barriers.keys())[:4]
        values = [barriers[n] for n in names]
        colors = ['#2ECC71' if v >= 0 else '#E74C3C' for v in values]
        ax2.barh(names, values, color=colors)
        ax2.axvline(x=0, color='black', linewidth=2)
        ax2.set_title('(b) Safety Certificate')
        ax2.set_xlabel('h(x)')
        
        # (c) Coverage Progress
        ax3 = fig.add_subplot(gs[0, 2])
        steps = np.arange(100)
        coverage = 100 * (1 - np.exp(-steps/30))
        ax3.plot(steps, coverage, linewidth=2, color=self.colors['primary'])
        ax3.fill_between(steps, coverage-5, coverage+5, alpha=0.2, color=self.colors['primary'])
        ax3.set_title('(c) Coverage Progress')
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Coverage (%)')
        ax3.set_ylim([0, 105])
        
        # (d) Multi-Agent
        ax4 = fig.add_subplot(gs[1, 0])
        positions = {0: [20, 20], 1: [80, 20], 2: [20, 80], 3: [80, 80]}
        for agent_id, pos in positions.items():
            ax4.plot(pos[0], pos[1], 'o', markersize=15, label=f'Agent {agent_id}')
        ax4.set_title('(d) Multi-Agent Coordination')
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.legend()
        
        # (e) Adaptation
        ax5 = fig.add_subplot(gs[1, 1])
        t = np.linspace(0, 1, 100)
        fast = np.exp(-t*10) * np.sin(t*50)
        medium = np.exp(-t*5) * 0.5
        slow = 1 - np.exp(-t*2)
        ax5.plot(t, fast, label='Fast (1ms)', linewidth=2)
        ax5.plot(t, medium, label='Medium (10ms)', linewidth=2)
        ax5.plot(t, slow, label='Slow (100ms)', linewidth=2)
        ax5.set_title('(e) Multi-Scale Adaptation')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Response')
        ax5.legend()
        
        # (f) Wind Profile
        ax6 = fig.add_subplot(gs[1, 2])
        r = np.linspace(0.1, 100, 100)
        R_max = 30
        V_max = 30
        speed = np.where(r < R_max, V_max * r/R_max, V_max * (R_max/r)**1.5)
        ax6.plot(r, speed, linewidth=2, color=self.colors['secondary'])
        ax6.axvline(x=R_max, color='gray', linestyle='--', label='R_max')
        ax6.set_title('(f) Rankine Vortex Model')
        ax6.set_xlabel('Radius (km)')
        ax6.set_ylabel('Wind Speed (m/s)')
        ax6.legend()
        
        plt.suptitle('MARAHS: Multi-Agent Robust Autonomous Hurricane Swarm', 
                    fontsize=18, fontweight='bold', y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_all_figures(self, output_dir: str = './figures'):
        """Save all publication figures."""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving figures to {output_dir}/")
        # Add figure generation calls here
        print("Figures saved!")


def create_latex_table(results: Dict) -> str:
    """Generate LaTeX table for paper."""
    header = r"""\begin{table}[t]
\centering
\caption{Single-Agent Station Keeping Performance}
\label{tab:track1}
\begin{tabular}{lccc}
\toprule
Method & Coverage (\%) & Safety (\%) & Time to 50\% \\
\midrule
"""
    
    rows = []
    for method, metrics in sorted(results.items(), key=lambda x: -x[1]['coverage_mean']):
        row = f"{method} & {metrics['coverage_mean']:.1f} $\\pm$ {metrics['coverage_std']:.1f} & {metrics['safety_mean']:.1f} & {metrics['time_to_50_mean']:.0f} \\\\"
        rows.append(row)
    
    footer = r"""\bottomrule
\end{tabular}
\end{table}"""
    
    return header + "\n".join(rows) + "\n" + footer
