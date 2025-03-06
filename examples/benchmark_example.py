"""
Benchmark suite example script for CHOP.

This script demonstrates how to create and run benchmark suites for different problem types,
analyze the results, and visualize performance metrics.
"""

import os
import sys
import numpy as np
from time import time
from typing import Dict, List, Any

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.problems import (
    TSP, 
    Knapsack, 
    Assignment, 
    BinPacking, 
    SetCover
)
from src.benchmarking import (
    BenchmarkSuite, 
    BenchmarkRunner, 
    BenchmarkResult,
    save_results, 
    load_results
)


def run_simple_benchmark():
    """Run a simple benchmark on predefined problem instances."""
    print("Creating benchmark suite from predefined instances...")
    
    # Create a benchmark suite from predefined instances
    suite = BenchmarkSuite.from_predefined_instances(
        name="predefined_benchmark", 
        problem_classes=[TSP, Knapsack, Assignment]
    )
    
    # Print information about the suite
    print(f"Created benchmark suite with {suite.count_instances()} instances:")
    for problem_type in suite.get_problem_types():
        print(f"  {problem_type}: {suite.count_instances(problem_type)} instances")
    
    # Create different solver configurations to compare
    solver_configs = [
        {'name': 'default', 'use_gomory_cuts': False, 'early_stop_gap': 0.0},
        {'name': 'with_cuts', 'use_gomory_cuts': True, 'early_stop_gap': 0.0},
        {'name': 'early_stop', 'use_gomory_cuts': False, 'early_stop_gap': 0.05}
    ]
    
    # Create and run the benchmark
    runner = BenchmarkRunner(
        suite=suite,
        output_dir="benchmark_results/predefined",
        time_limit=60.0,  # 1 minute per instance
        solver_configs=solver_configs,
        parallel=True  # Run benchmarks in parallel
    )
    
    # Run the benchmark for all problems
    print("Running benchmark...")
    results = runner.run_benchmark()
    
    # Visualize the results
    print("Visualizing results...")
    runner.visualize_results(results)
    
    # Compare different solver configurations
    print("Comparing solver configurations...")
    runner.compare_solvers(results)
    
    return results


def run_comprehensive_benchmark():
    """Run a comprehensive benchmark on generated problem instances."""
    print("Creating benchmark suite from generated instances...")
    
    # Create a benchmark suite with generated instances
    suite = BenchmarkSuite.from_problem_generators(
        name="comprehensive_benchmark",
        problem_classes=[TSP, Knapsack, Assignment, BinPacking, SetCover],
        difficulties=['easy', 'medium']  # Exclude 'hard' to keep runtime reasonable
    )
    
    # Print information about the suite
    print(f"Created benchmark suite with {suite.count_instances()} instances:")
    for problem_type in suite.get_problem_types():
        print(f"  {problem_type}: {suite.count_instances(problem_type)} instances")
        
    # Different solver configurations to compare
    solver_configs = [
        {'name': 'default', 'use_gomory_cuts': False, 'early_stop_gap': 0.0},
        {'name': 'with_cuts', 'use_gomory_cuts': True, 'early_stop_gap': 0.0},
        {'name': 'early_stop_5pct', 'use_gomory_cuts': False, 'early_stop_gap': 0.05},
        {'name': 'cuts_and_early_stop', 'use_gomory_cuts': True, 'early_stop_gap': 0.05}
    ]
    
    # Create the benchmark runner
    runner = BenchmarkRunner(
        suite=suite,
        output_dir="benchmark_results/comprehensive",
        time_limit=300.0,  # 5 minutes per instance
        solver_configs=solver_configs,
        parallel=True
    )
    
    # Run benchmarks by problem type and difficulty
    all_results = []
    
    for problem_type in suite.get_problem_types():
        for difficulty in ['easy', 'medium']:
            print(f"Running benchmark for {problem_type} ({difficulty})...")
            results = runner.run_benchmark(
                problem_type=problem_type,
                difficulty=difficulty
            )
            all_results.extend(results)
    
    # Visualize all results together
    print("Visualizing all results...")
    runner.visualize_results(all_results)
    
    # Compare solver configurations across all problems
    print("Comparing solver configurations...")
    runner.compare_solvers(all_results)
    
    return all_results


def analyze_specific_properties():
    """Analyze specific properties of problem instances and their correlation with solving difficulty."""
    print("Analyzing specific problem properties...")
    
    # Create a benchmark suite from predefined instances
    suite = BenchmarkSuite.from_predefined_instances(
        name="property_analysis",
        problem_classes=[TSP, Knapsack, Assignment, BinPacking, SetCover]
    )
    
    # Create the benchmark runner
    runner = BenchmarkRunner(
        suite=suite,
        output_dir="benchmark_results/properties",
        solver_configs=[{'name': 'default'}]  # Just one configuration for this analysis
    )
    
    # Run the benchmark
    results = runner.run_benchmark()
    
    # Analyze results
    df = runner.analyze_results(results)
    
    # Print correlations with solution time
    print("\nCorrelation with solution time:")
    for column in df.select_dtypes(include=['number']).columns:
        if column != 'solution_time' and not pd.isna(df[column]).all():
            corr = df['solution_time'].corr(df[column])
            if not pd.isna(corr):
                print(f"  {column}: {corr:.3f}")
    
    # For each problem type, analyze specific metrics
    for problem_type in df['problem_type'].unique():
        print(f"\nAnalysis for {problem_type}:")
        pt_df = df[df['problem_type'] == problem_type]
        
        # Get problem-specific columns
        specific_cols = [col for col in pt_df.columns if col.startswith('size_')]
        
        # Print correlations for this problem type
        for col in specific_cols:
            if not pd.isna(pt_df[col]).all():
                corr = pt_df['solution_time'].corr(pt_df[col])
                if not pd.isna(corr):
                    print(f"  {col}: {corr:.3f}")
    
    return results


def main():
    """Main function to run the benchmark examples."""
    # Create output directory
    os.makedirs("benchmark_results", exist_ok=True)
    
    # Import pandas here to avoid import if the script is imported
    import pandas as pd
    global pd
    
    print("=== Running Simple Benchmark ===")
    simple_results = run_simple_benchmark()
    
    print("\n=== Running Comprehensive Benchmark ===")
    comprehensive_results = run_comprehensive_benchmark()
    
    print("\n=== Analyzing Problem Properties ===")
    property_results = analyze_specific_properties()
    
    print("\nAll benchmarks complete!")


if __name__ == "__main__":
    main()