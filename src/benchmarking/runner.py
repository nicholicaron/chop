"""Benchmark runner for CHOP."""

import os
import time
import datetime
import pathlib
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from ..core.solver import BnBSolver
from ..problems.base import OptimizationProblem
from .metrics import InstanceMetrics, SolverMetrics
from .suite import BenchmarkSuite, BenchmarkResult, save_results


class BenchmarkRunner:
    """Runner for executing benchmarks on problem instances."""
    
    def __init__(
        self, 
        suite: BenchmarkSuite,
        output_dir: str = 'benchmark_results',
        time_limit: Optional[float] = None,
        solver_configs: Optional[List[Dict[str, Any]]] = None,
        parallel: bool = False,
        max_workers: int = None
    ):
        """Initialize a benchmark runner.
        
        Args:
            suite: Benchmark suite to run
            output_dir: Directory to save results to
            time_limit: Maximum time (in seconds) to spend on each instance
            solver_configs: List of solver configurations to test
            parallel: Whether to run benchmarks in parallel
            max_workers: Maximum number of worker processes (default: number of CPUs)
        """
        self.suite = suite
        self.output_dir = output_dir
        self.time_limit = time_limit
        self.parallel = parallel
        self.max_workers = max_workers
        
        # Default solver configuration if none provided
        if solver_configs is None:
            solver_configs = [
                {'name': 'default', 'use_gomory_cuts': False, 'early_stop_gap': 0.0}
            ]
        self.solver_configs = solver_configs
        
        # Create output directory if it doesn't exist
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def run_benchmark(
        self, 
        problem_type: Optional[str] = None,
        difficulty: Optional[str] = None,
        instances: Optional[List[OptimizationProblem]] = None,
        solver_configs: Optional[List[Dict[str, Any]]] = None,
        save: bool = True
    ) -> List[BenchmarkResult]:
        """Run a benchmark on the given instances.
        
        Args:
            problem_type: Type of problem to benchmark (if None, benchmark all types)
            difficulty: Difficulty level to benchmark (if None, benchmark all difficulties)
            instances: Specific instances to benchmark (if None, use all instances matching the criteria)
            solver_configs: Solver configurations to test (if None, use default configs)
            save: Whether to save results to disk
            
        Returns:
            List of benchmark results
        """
        # Get instances to benchmark
        if instances is None:
            instances_to_benchmark = []
            
            if problem_type is None:
                # Use all problem types
                problem_types = self.suite.get_problem_types()
            else:
                problem_types = [problem_type]
            
            for pt in problem_types:
                if difficulty is None:
                    # Use all instances of this problem type
                    instances_to_benchmark.extend(
                        (pt, instance) for instance in self.suite.instances[pt]
                    )
                else:
                    # Use only instances of the specified difficulty
                    instances_to_benchmark.extend(
                        (pt, instance) for instance in self.suite.get_instances_by_difficulty(pt, difficulty)
                    )
        else:
            # Use the provided instances
            instances_to_benchmark = [(instance.__class__.__name__, instance) for instance in instances]
        
        # Use default solver configs if none provided
        if solver_configs is None:
            solver_configs = self.solver_configs
        
        # Define the benchmark task
        def benchmark_task(args):
            problem_type, instance, config = args
            return self._run_single_benchmark(problem_type, instance, config)
        
        # Generate all benchmark tasks
        tasks = []
        for problem_type, instance in instances_to_benchmark:
            for config in solver_configs:
                tasks.append((problem_type, instance, config))
        
        # Run benchmarks
        results = []
        if self.parallel and len(tasks) > 1:
            # Run in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(benchmark_task, task) for task in tasks]
                
                # Show progress
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    result = future.result()
                    results.append(result)
        else:
            # Run sequentially
            for task in tqdm(tasks):
                result = benchmark_task(task)
                results.append(result)
        
        # Save results if requested
        if save:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.suite.name}_benchmark_{timestamp}.json"
            save_results(results, self.output_dir, filename)
        
        return results
    
    def _run_single_benchmark(
        self, 
        problem_type: str, 
        instance: OptimizationProblem, 
        solver_config: Dict[str, Any]
    ) -> BenchmarkResult:
        """Run a benchmark on a single problem instance with a specific solver configuration.
        
        Args:
            problem_type: Type of the problem
            instance: Problem instance to benchmark
            solver_config: Solver configuration to use
            
        Returns:
            Benchmark result
        """
        # Get instance metrics
        instance_id = f"{problem_type}_{instance.name}"
        instance_metrics = self.suite.metrics[problem_type][instance_id]
        
        # Create solver with the given configuration
        config_name = solver_config.get('name', 'default')
        use_gomory_cuts = solver_config.get('use_gomory_cuts', False)
        early_stop_gap = solver_config.get('early_stop_gap', 0.0)
        time_limit = solver_config.get('time_limit', self.time_limit)
        
        # Convert instance to ILP
        ilp_model = instance.to_ilp()
        
        # Create solver
        solver = BnBSolver(
            ilp_model,
            use_gomory_cuts=use_gomory_cuts,
            early_stop_gap=early_stop_gap,
            time_limit=time_limit
        )
        
        # Apply additional configuration options
        for key, value in solver_config.items():
            if key not in ['name', 'use_gomory_cuts', 'early_stop_gap', 'time_limit']:
                if hasattr(solver, key):
                    setattr(solver, key, value)
        
        # Solve the instance and measure time
        start_time = time.time()
        solver.solve()
        duration = time.time() - start_time
        
        # Extract solver metrics
        solver_metrics = SolverMetrics.from_solver(solver)
        
        # Create result
        result = BenchmarkResult.create(
            problem=instance,
            instance_metrics=instance_metrics,
            solver_metrics=solver_metrics,
            solver_config=solver_config,
            duration=duration
        )
        
        return result
    
    def analyze_results(self, results: List[BenchmarkResult]) -> pd.DataFrame:
        """Analyze benchmark results and return a DataFrame.
        
        Args:
            results: List of benchmark results
            
        Returns:
            DataFrame containing result analysis
        """
        data = []
        
        for result in results:
            # Extract solver metrics
            sm = result.solver_metrics
            im = result.instance_metrics
            
            # Basic information
            row = {
                'problem_type': result.problem_type,
                'problem_name': result.problem_name,
                'instance_id': result.instance_id,
                'difficulty': im['difficulty'],
                'config_name': result.solver_config.get('name', 'default'),
                'solution_time': sm['solution_time'],
                'solution_found': sm['solution_found'],
                'solution_value': sm['solution_value'],
                'proven_optimal': sm['proven_optimal'],
                'nodes_created': sm['nodes_created'],
                'nodes_processed': sm['nodes_processed'],
                'lp_relaxations_solved': sm['lp_relaxations_solved'],
                'density': im.get('density', 0),
                'num_variables': im.get('num_variables', 0),
                'num_constraints': im.get('num_constraints', 0),
                'lp_relaxation_value': im.get('lp_relaxation_value', 0),
                'lp_relaxation_time': im.get('lp_relaxation_time', 0),
            }
            
            # Add problem-specific size metrics
            for key, value in im.get('size', {}).items():
                row[f'size_{key}'] = value
            
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add computed columns
        if 'solution_value' in df.columns and 'lp_relaxation_value' in df.columns:
            # Add LP gap for solved instances
            mask = df['solution_found'] & (df['solution_value'].abs() > 1e-10)
            df.loc[mask, 'lp_gap'] = (df.loc[mask, 'solution_value'] - df.loc[mask, 'lp_relaxation_value']) / df.loc[mask, 'solution_value'].abs()
        
        return df
    
    def visualize_results(
        self, 
        results: List[BenchmarkResult], 
        output_dir: Optional[str] = None,
        save_format: str = 'png'
    ) -> None:
        """Visualize benchmark results.
        
        Args:
            results: List of benchmark results
            output_dir: Directory to save visualizations to (if None, use runner's output_dir)
            save_format: Format to save visualizations in ('png', 'pdf', 'svg')
        """
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, 'plots')
        
        # Create output directory if it doesn't exist
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Analyze results
        df = self.analyze_results(results)
        
        # Plot solution time distribution by problem type
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df, x='problem_type', y='solution_time', hue='difficulty')
        plt.title('Solution Time by Problem Type and Difficulty')
        plt.xlabel('Problem Type')
        plt.ylabel('Solution Time (s)')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'solution_time_by_type.{save_format}'))
        plt.close()
        
        # Plot nodes processed distribution by problem type
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df, x='problem_type', y='nodes_processed', hue='difficulty')
        plt.title('Nodes Processed by Problem Type and Difficulty')
        plt.xlabel('Problem Type')
        plt.ylabel('Nodes Processed')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'nodes_by_type.{save_format}'))
        plt.close()
        
        # Plot LP gap distribution by problem type (if available)
        if 'lp_gap' in df.columns:
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=df, x='problem_type', y='lp_gap', hue='difficulty')
            plt.title('LP Relaxation Gap by Problem Type and Difficulty')
            plt.xlabel('Problem Type')
            plt.ylabel('LP Gap')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'lp_gap_by_type.{save_format}'))
            plt.close()
        
        # Create heatmap of solution time by problem type and difficulty
        pivot_time = df.pivot_table(
            index='problem_type', 
            columns='difficulty', 
            values='solution_time', 
            aggfunc='mean'
        )
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_time, annot=True, fmt='.3g', cmap='viridis')
        plt.title('Average Solution Time by Problem Type and Difficulty')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'time_heatmap.{save_format}'))
        plt.close()
        
        # Create comparative bar chart for different solver configurations
        if 'config_name' in df.columns and df['config_name'].nunique() > 1:
            plt.figure(figsize=(14, 8))
            sns.barplot(data=df, x='problem_type', y='solution_time', hue='config_name')
            plt.title('Solution Time by Problem Type and Solver Configuration')
            plt.xlabel('Problem Type')
            plt.ylabel('Solution Time (s)')
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'config_comparison.{save_format}'))
            plt.close()
        
        # Plot correlation between instance metrics and solution time
        corr_cols = ['solution_time', 'nodes_processed', 'num_variables', 
                     'num_constraints', 'density', 'lp_relaxation_time']
        size_cols = [col for col in df.columns if col.startswith('size_')]
        corr_cols.extend(size_cols)
        
        if 'lp_gap' in df.columns:
            corr_cols.append('lp_gap')
        
        # Filter only numerical columns that exist in the DataFrame
        corr_cols = [col for col in corr_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if len(corr_cols) > 1:
            plt.figure(figsize=(12, 10))
            sns.heatmap(df[corr_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm')
            plt.title('Correlation between Instance Metrics and Solution Time')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'metrics_correlation.{save_format}'))
            plt.close()
        
        # Create summary statistics file
        summary_path = os.path.join(output_dir, 'benchmark_summary.txt')
        with open(summary_path, 'w') as f:
            # Write overall statistics
            f.write(f"Benchmark Summary\n")
            f.write(f"================\n\n")
            f.write(f"Total instances: {len(df)}\n")
            f.write(f"Problem types: {', '.join(df['problem_type'].unique())}\n")
            f.write(f"Difficulty levels: {', '.join(df['difficulty'].unique())}\n\n")
            
            # Write problem-specific statistics
            f.write(f"Statistics by Problem Type\n")
            f.write(f"-------------------------\n\n")
            
            for problem_type in df['problem_type'].unique():
                pt_df = df[df['problem_type'] == problem_type]
                f.write(f"{problem_type}:\n")
                f.write(f"  Instances: {len(pt_df)}\n")
                f.write(f"  Solution time (mean): {pt_df['solution_time'].mean():.3f}s\n")
                f.write(f"  Solution time (median): {pt_df['solution_time'].median():.3f}s\n")
                f.write(f"  Nodes processed (mean): {pt_df['nodes_processed'].mean():.1f}\n")
                f.write(f"  Optimal solutions: {pt_df['proven_optimal'].sum()} / {len(pt_df)}\n")
                
                if 'lp_gap' in pt_df.columns:
                    f.write(f"  LP gap (mean): {pt_df['lp_gap'].mean():.3f}\n")
                
                f.write("\n")
            
            # Write difficulty-specific statistics
            f.write(f"Statistics by Difficulty\n")
            f.write(f"----------------------\n\n")
            
            for difficulty in df['difficulty'].unique():
                diff_df = df[df['difficulty'] == difficulty]
                f.write(f"{difficulty}:\n")
                f.write(f"  Instances: {len(diff_df)}\n")
                f.write(f"  Solution time (mean): {diff_df['solution_time'].mean():.3f}s\n")
                f.write(f"  Solution time (median): {diff_df['solution_time'].median():.3f}s\n")
                f.write(f"  Nodes processed (mean): {diff_df['nodes_processed'].mean():.1f}\n")
                f.write(f"  Optimal solutions: {diff_df['proven_optimal'].sum()} / {len(diff_df)}\n")
                
                if 'lp_gap' in diff_df.columns:
                    f.write(f"  LP gap (mean): {diff_df['lp_gap'].mean():.3f}\n")
                
                f.write("\n")
        
        print(f"Visualization results saved to {output_dir}")
        print(f"Summary statistics saved to {summary_path}")
        
    def compare_solvers(
        self, 
        results: List[BenchmarkResult],
        output_dir: Optional[str] = None,
        save_format: str = 'png'
    ) -> None:
        """Compare different solver configurations.
        
        Args:
            results: List of benchmark results
            output_dir: Directory to save visualizations to (if None, use runner's output_dir)
            save_format: Format to save visualizations in ('png', 'pdf', 'svg')
        """
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, 'comparisons')
        
        # Create output directory if it doesn't exist
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Analyze results
        df = self.analyze_results(results)
        
        # Check if we have different configurations to compare
        config_names = df['config_name'].unique()
        if len(config_names) <= 1:
            print("No different solver configurations to compare")
            return
        
        # Compute speedup relative to the first configuration
        # Create a pivot table with instance_id as the index and config_name as columns
        pivot_df = df.pivot_table(
            index=['instance_id', 'problem_type', 'difficulty'], 
            columns='config_name', 
            values=['solution_time', 'nodes_processed']
        )
        
        # Compute speedup for each configuration relative to the first one
        base_config = config_names[0]
        speedup_df = pd.DataFrame(index=pivot_df.index)
        
        for config in config_names[1:]:
            speedup_df[f'speedup_{config}'] = pivot_df['solution_time'][base_config] / pivot_df['solution_time'][config]
            speedup_df[f'node_reduction_{config}'] = (
                pivot_df['nodes_processed'][base_config] - pivot_df['nodes_processed'][config]
            ) / pivot_df['nodes_processed'][base_config]
        
        # Reset index to make it easier to work with
        speedup_df = speedup_df.reset_index()
        
        # Plot speedup by problem type
        for config in config_names[1:]:
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=speedup_df, x='problem_type', y=f'speedup_{config}', hue='difficulty')
            plt.axhline(y=1.0, color='r', linestyle='--')
            plt.title(f'Speedup: {config} vs {base_config} by Problem Type')
            plt.xlabel('Problem Type')
            plt.ylabel('Speedup (>1 is better)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'speedup_{config}_vs_{base_config}.{save_format}'))
            plt.close()
            
            plt.figure(figsize=(12, 8))
            sns.boxplot(data=speedup_df, x='problem_type', y=f'node_reduction_{config}', hue='difficulty')
            plt.axhline(y=0.0, color='r', linestyle='--')
            plt.title(f'Node Reduction: {config} vs {base_config} by Problem Type')
            plt.xlabel('Problem Type')
            plt.ylabel('Node Reduction (>0 is better)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'node_reduction_{config}_vs_{base_config}.{save_format}'))
            plt.close()
        
        # Create a summary table
        summary_df = speedup_df.groupby('problem_type').agg({
            f'speedup_{config}': ['mean', 'median', 'std'] for config in config_names[1:]
        })
        
        # Create summary statistics file
        summary_path = os.path.join(output_dir, 'solver_comparison_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Solver Comparison Summary\n")
            f.write(f"========================\n\n")
            f.write(f"Base configuration: {base_config}\n")
            f.write(f"Compared configurations: {', '.join(config_names[1:])}\n\n")
            
            for config in config_names[1:]:
                f.write(f"Configuration: {config}\n")
                f.write(f"-------------------\n\n")
                
                # Overall statistics
                overall_speedup = speedup_df[f'speedup_{config}']
                f.write(f"Overall speedup (mean): {overall_speedup.mean():.3f}x\n")
                f.write(f"Overall speedup (median): {overall_speedup.median():.3f}x\n")
                f.write(f"Instances with speedup > 1: {(overall_speedup > 1).sum()} / {len(overall_speedup)}\n\n")
                
                # Per problem type statistics
                f.write(f"Speedup by Problem Type:\n")
                for problem_type in speedup_df['problem_type'].unique():
                    pt_speedup = speedup_df[speedup_df['problem_type'] == problem_type][f'speedup_{config}']
                    f.write(f"  {problem_type}:\n")
                    f.write(f"    Mean speedup: {pt_speedup.mean():.3f}x\n")
                    f.write(f"    Median speedup: {pt_speedup.median():.3f}x\n")
                    f.write(f"    Instances with speedup > 1: {(pt_speedup > 1).sum()} / {len(pt_speedup)}\n\n")
                
                f.write("\n")
        
        print(f"Solver comparison results saved to {output_dir}")
        print(f"Comparison summary saved to {summary_path}")