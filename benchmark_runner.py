"""
MARAHS Benchmark Runner
========================

Runs complete experimental evaluation and produces publication-ready results.

Usage:
    python benchmark_runner.py --track 1 --num_seeds 50
    python benchmark_runner.py --track 2 --num_seeds 30
    python benchmark_runner.py --ablation
    python benchmark_runner.py --all
"""

import numpy as np
import time
import json
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    track: int = 1
    num_seeds: int = 50
    num_episodes: int = 50
    wind_profiles: List[str] = field(default_factory=lambda: ['katrina', 'harvey', 'irma'])
    wind_intensities: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    output_dir: str = './benchmark_results'


class MethodEvaluator:
    """Evaluates a single method across all conditions."""
    
    def __init__(self, method_name: str):
        self.method_name = method_name
        self.results = []
    
    def evaluate(self, env, method, num_episodes: int, seed: int) -> Dict:
        """Run one evaluation episode."""
        np.random.seed(seed)
        
        total_coverage = 0
        safety_violations = 0
        steps_to_50 = None
        info_gain = 0
        
        for ep in range(num_episodes):
            obs, _ = env.reset()
            ep_coverage = 0
            ep_steps = 0
            
            for step in range(env.config.max_steps if hasattr(env, 'config') else 300):
                # Get action from method
                if hasattr(method, 'get_action'):
                    action = method.get_action(obs)
                else:
                    action = env.action_space.sample()
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                ep_steps += 1
                
                # Track coverage
                if 'coverage_pct' in info:
                    ep_coverage = info['coverage_pct']
                    if ep_coverage >= 50 and steps_to_50 is None:
                        steps_to_50 = ep_steps
                
                # Check safety
                if terminated and reward < -50:
                    safety_violations += 1
                    break
            
            total_coverage += ep_coverage
        
        return {
            'coverage': total_coverage / num_episodes,
            'safety_rate': 100 * (1 - safety_violations / num_episodes),
            'time_to_50': steps_to_50 or float('inf'),
            'episodes': num_episodes,
            'seed': seed,
        }


class BenchmarkRunner:
    """
    Complete benchmark runner for MARAHS evaluation.
    
    Runs experiments and produces publication-ready results.
    """
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.results = {}
        self.methods = {}
        
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def register_method(self, name: str, method):
        """Register a method for evaluation."""
        self.methods[name] = method
    
    def run_track1(self) -> Dict:
        """
        Track 1: Single-Agent Station Keeping.
        
        Evaluates single drone coverage under different wind conditions.
        """
        print(f"\n{'='*60}")
        print(f"TRACK 1: Single-Agent Station Keeping")
        print(f"{'='*60}\n")
        
        from hurricane_env import HurricaneStationKeepingEnv, HurricaneConfig
        from real_wind_provider import RealWindProvider
        
        results = {}
        
        for method_name, method in self.methods.items():
            print(f"Evaluating: {method_name}")
            method_results = []
            
            for wind_intensity in self.config.wind_intensities:
                for wind_profile in self.config.wind_profiles:
                    for seed in range(self.config.num_seeds):
                        # Create environment
                        config = HurricaneConfig(wind_provider=wind_profile)
                        env = HurricaneStationKeepingEnv(config=config)
                        
                        # Set wind
                        wind = RealWindProvider(wind_profile)
                        env.set_wind_provider(wind)
                        
                        # Evaluate
                        evaluator = MethodEvaluator(method_name)
                        result = evaluator.evaluate(env, method, 1, seed)
                        
                        result['wind_intensity'] = wind_intensity
                        result['wind_profile'] = wind_profile
                        method_results.append(result)
            
            # Aggregate results
            results[method_name] = self._aggregate_results(method_results)
            print(f"  Coverage: {results[method_name]['coverage_mean']:.1f}% ± {results[method_name]['coverage_std']:.1f}%")
            print(f"  Safety: {results[method_name]['safety_mean']:.1f}%")
        
        self.results['track1'] = results
        return results
    
    def run_track2(self) -> Dict:
        """
        Track 2: Multi-Agent Coverage.
        
        Evaluates 4-drone swarm coordination.
        """
        print(f"\n{'='*60}")
        print(f"TRACK 2: Multi-Agent Coverage")
        print(f"{'='*60}\n")
        
        from swarm_grid_env import SwarmGridWorld, SwarmGridConfig
        
        results = {}
        
        for method_name, method in self.methods.items():
            print(f"Evaluating: {method_name}")
            method_results = []
            
            for seed in range(self.config.num_seeds):
                # Create environment
                config = SwarmGridConfig(
                    grid_size=15,
                    num_drones=4,
                    max_steps=300,
                    wind_intensity=0.5,
                )
                env = SwarmGridWorld(config)
                
                # Evaluate
                evaluator = MethodEvaluator(method_name)
                result = evaluator.evaluate(env, method, 1, seed)
                method_results.append(result)
            
            results[method_name] = self._aggregate_results(method_results)
            print(f"  Coverage: {results[method_name]['coverage_mean']:.1f}% ± {results[method_name]['coverage_std']:.1f}%")
        
        self.results['track2'] = results
        return results
    
    def run_ablation(self) -> Dict:
        """
        Ablation Study.
        
        Evaluates each component's contribution.
        """
        print(f"\n{'='*60}")
        print(f"ABLATION STUDY")
        print(f"{'='*60}\n")
        
        # Define ablation configurations
        ablations = {
            'Full MARAHS': ['wind_mapper', 'adaptive_cbf', 'inverse_dynamics', 
                           'adversarial', 'information', 'multiscale', 'formal'],
            '-Wind Mapper': ['adaptive_cbf', 'inverse_dynamics', 
                           'adversarial', 'information', 'multiscale', 'formal'],
            '-Adaptive CBF': ['wind_mapper', 'inverse_dynamics', 
                           'adversarial', 'information', 'multiscale', 'formal'],
            '-Inverse Dynamics': ['wind_mapper', 'adaptive_cbf',
                           'adversarial', 'information', 'multiscale', 'formal'],
            '-Adversarial': ['wind_mapper', 'adaptive_cbf', 'inverse_dynamics',
                           'information', 'multiscale', 'formal'],
            '-Information': ['wind_mapper', 'adaptive_cbf', 'inverse_dynamics',
                           'adversarial', 'multiscale', 'formal'],
            '-Multi-Scale': ['wind_mapper', 'adaptive_cbf', 'inverse_dynamics',
                           'adversarial', 'information', 'formal'],
            '-Formal': ['wind_mapper', 'adaptive_cbf', 'inverse_dynamics',
                       'adversarial', 'information', 'multiscale'],
        }
        
        results = {}
        
        for config_name, components in ablations.items():
            print(f"Testing: {config_name}")
            
            # Create method with specified components
            method = self._create_ablated_method(components)
            
            # Evaluate
            method_results = []
            for seed in range(self.config.num_seeds):
                from hurricane_env import HurricaneStationKeepingEnv, HurricaneConfig
                config = HurricaneConfig(wind_provider='katrina')
                env = HurricaneStationKeepingEnv(config=config)
                
                evaluator = MethodEvaluator(config_name)
                result = evaluator.evaluate(env, method, 1, seed)
                method_results.append(result)
            
            results[config_name] = self._aggregate_results(method_results)
            print(f"  Coverage: {results[config_name]['coverage_mean']:.1f}%")
        
        self.results['ablation'] = results
        return results
    
    def _create_ablated_method(self, components: List[str]):
        """Create a method with only specified components."""
        from safe_adaptive_controller import SafeAdaptiveController, SafeAdaptiveConfig
        
        config = SafeAdaptiveConfig(
            enable_cbf='adaptive_cbf' in components,
            enable_wind_mapping='wind_mapper' in components,
            enable_inverse_dynamics='inverse_dynamics' in components,
            enable_adversarial='adversarial' in components,
            enable_information='information' in components,
            enable_multiscale='multiscale' in components,
        )
        
        return SafeAdaptiveController(config)
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate results across seeds."""
        coverages = [r['coverage'] for r in results]
        safety_rates = [r['safety_rate'] for r in results]
        times_to_50 = [r['time_to_50'] for r in results if r['time_to_50'] is not None]
        
        return {
            'coverage_mean': float(np.mean(coverages)),
            'coverage_std': float(np.std(coverages)),
            'coverage_median': float(np.median(coverages)),
            'coverage_95ci': float(1.96 * np.std(coverages) / np.sqrt(len(coverages))),
            'safety_mean': float(np.mean(safety_rates)),
            'safety_std': float(np.std(safety_rates)),
            'time_to_50_mean': float(np.mean(times_to_50)) if times_to_50 else float('inf'),
            'time_to_50_std': float(np.std(times_to_50)) if times_to_50 else 0,
            'num_seeds': len(results),
        }
    
    def generate_tables(self) -> str:
        """Generate publication-ready result tables."""
        tables = []
        
        if 'track1' in self.results:
            tables.append(self._generate_track1_table())
        
        if 'track2' in self.results:
            tables.append(self._generate_track2_table())
        
        if 'ablation' in self.results:
            tables.append(self._generate_ablation_table())
        
        return "\n\n".join(tables)
    
    def _generate_track1_table(self) -> str:
        """Generate Table 1: Single-Agent Performance."""
        results = self.results['track1']
        
        header = "| Method | Coverage | Safety | Time to 50% |"
        separator = "|--------|----------|--------|-------------|"
        
        rows = [header, separator]
        
        for method_name, metrics in sorted(results.items(), 
                                          key=lambda x: -x[1]['coverage_mean']):
            row = f"| {method_name} | {metrics['coverage_mean']:.1f} ± {metrics['coverage_std']:.1f}% | {metrics['safety_mean']:.1f}% | {metrics['time_to_50_mean']:.0f} |"
            rows.append(row)
        
        return "**Table 1: Single-Agent Station Keeping Performance**\n\n" + "\n".join(rows)
    
    def _generate_track2_table(self) -> str:
        """Generate Table 2: Multi-Agent Performance."""
        results = self.results['track2']
        
        header = "| Method | Coverage (4 agents) | Safety |"
        separator = "|--------|---------------------|--------|"
        
        rows = [header, separator]
        
        for method_name, metrics in sorted(results.items(),
                                          key=lambda x: -x[1]['coverage_mean']):
            row = f"| {method_name} | {metrics['coverage_mean']:.1f} ± {metrics['coverage_std']:.1f}% | {metrics['safety_mean']:.1f}% |"
            rows.append(row)
        
        return "**Table 2: Multi-Agent Coverage Performance**\n\n" + "\n".join(rows)
    
    def _generate_ablation_table(self) -> str:
        """Generate Table 3: Ablation Study."""
        results = self.results['ablation']
        full_coverage = results.get('Full MARAHS', {}).get('coverage_mean', 100)
        
        header = "| Configuration | Coverage | Δ from Full |"
        separator = "|---------------|----------|-------------|"
        
        rows = [header, separator]
        
        for config_name, metrics in results.items():
            delta = metrics['coverage_mean'] - full_coverage
            delta_str = f"{delta:+.1f}%" if config_name != 'Full MARAHS' else "—"
            row = f"| {config_name} | {metrics['coverage_mean']:.1f}% | {delta_str} |"
            rows.append(row)
        
        return "**Table 3: Ablation Study Results**\n\n" + "\n".join(rows)
    
    def save_results(self):
        """Save all results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.output_dir}/benchmark_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
        
        # Also save tables
        tables_file = f"{self.config.output_dir}/tables_{timestamp}.md"
        with open(tables_file, 'w') as f:
            f.write(self.generate_tables())
        
        print(f"Tables saved to: {tables_file}")
    
    def run_all(self):
        """Run all benchmark tracks."""
        print(f"\n{'#'*60}")
        print(f"# MARAHS Benchmark Suite")
        print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*60}\n")
        
        start_time = time.time()
        
        # Run tracks
        self.run_track1()
        self.run_track2()
        self.run_ablation()
        
        # Generate and save results
        self.save_results()
        
        # Print summary
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"BENCHMARK COMPLETE")
        print(f"{'='*60}")
        print(f"Time elapsed: {elapsed:.1f} seconds")
        print(f"Results directory: {self.config.output_dir}")
        print()
        
        # Print publication tables
        print(self.generate_tables())


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='MARAHS Benchmark Runner')
    parser.add_argument('--track', type=int, default=1, help='Benchmark track (1, 2, or 0 for all)')
    parser.add_argument('--num_seeds', type=int, default=50, help='Number of random seeds')
    parser.add_argument('--ablation', action='store_true', help='Run ablation study')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--output', type=str, default='./benchmark_results', help='Output directory')
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        num_seeds=args.num_seeds,
        output_dir=args.output,
    )
    
    runner = BenchmarkRunner(config)
    
    # Register baseline methods
    from pid_baseline import PIDBaseline, GreedyBaseline, RandomBaseline
    from network import ActorCritic
    
    runner.register_method('Random', RandomBaseline())
    runner.register_method('PID', PIDBaseline())
    runner.register_method('Greedy', GreedyBaseline())
    
    # Register MARAHS
    from safe_adaptive_controller import SafeAdaptiveController
    runner.register_method('MARAHS', SafeAdaptiveController())
    
    if args.all or args.ablation:
        runner.run_all()
    elif args.track == 1:
        runner.run_track1()
    elif args.track == 2:
        runner.run_track2()
    else:
        runner.run_all()


if __name__ == '__main__':
    main()
