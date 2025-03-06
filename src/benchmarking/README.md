# CHOP Benchmarking Framework

This document provides an overview of how to use the benchmarking framework in CHOP to evaluate the performance of various solver configurations on different optimization problems.

## Overview

The CHOP benchmarking framework allows you to:

1. Run benchmarks for different optimization problem types (TSP, Knapsack, Assignment, Bin Packing, Set Cover)
2. Test various difficulty levels (easy, medium, hard)
3. Compare multiple solver configurations
4. Generate visualizations and statistical analyses of the results

## Quick Start

To quickly run benchmarks, use the provided `benchmark.py` script at the root of the project:

```bash
# Activate your Python environment
source .venv/bin/activate

# Run a simple benchmark for all problem types at easy difficulty
python benchmark.py

# Run a benchmark for specific problem types
python benchmark.py --problems tsp knapsack

# Run a benchmark with different difficulty levels
python benchmark.py --problems tsp --difficulties easy medium

# Run a benchmark with a specific time limit per instance
python benchmark.py --time-limit 30.0

# Run benchmarks sequentially instead of in parallel
python benchmark.py --sequential
```

## Command-Line Arguments

- `--problems`: Problem types to benchmark (default: all)
  - Choices: tsp, knapsack, assignment, bin_packing, set_cover
  - Example: `--problems tsp knapsack`

- `--difficulties`: Difficulty levels to benchmark (default: easy)
  - Choices: easy, medium, hard
  - Example: `--difficulties easy medium`

- `--time-limit`: Time limit per instance in seconds (default: 60.0)
  - Example: `--time-limit 120.0`

- `--sequential`: Run benchmarks sequentially instead of in parallel
  - Example: `--sequential`

## Benchmark Outputs

The benchmark script creates a timestamped directory (e.g., `benchmark_results_20250306_154605`) with the following structure:

```
benchmark_results_TIMESTAMP/
├── benchmark_results_TIMESTAMP.json  # Raw results data
├── plots/                           # Visualizations
│   ├── benchmark_summary.txt        # Summary statistics
│   ├── solution_time_distribution.png
│   ├── nodes_processed_distribution.png
│   └── ... other plots
└── comparisons/                     # Solver configuration comparisons
    ├── solver_comparison_summary.txt
    ├── solution_time_by_solver.png
    └── ... other comparison plots
```

## Solver Configurations

The benchmarking framework tests multiple solver configurations:

1. `default`: Basic solver without Gomory cuts and exact solution finding
2. `with_cuts`: Solver using Gomory cutting planes
3. `early_stop_5pct`: Solver with early stopping (5% optimality gap)
4. `cuts_and_early_stop`: Solver with both Gomory cuts and early stopping

## Adding Custom Problem Metrics

To add custom metrics for a problem type, implement the following methods in your problem class:

```python
def size_metrics(self) -> Dict[str, int]:
    """Return a dictionary of size-related metrics"""
    return {
        'size_custom_metric1': value1,
        'size_custom_metric2': value2,
        # ...
    }

def get_specific_metrics(self) -> Dict[str, Any]:
    """Return a dictionary of problem-specific metrics"""
    return {
        'metric1': value1,
        'metric2': value2,
        'is_minimization': True,  # Important for gap calculations
        # ...
    }
```

These metrics will automatically be incorporated into benchmark results and visualizations.

## Programmatic Usage

You can also use the benchmarking framework programmatically in your own scripts:

```python
from src.problems import TSP, Knapsack
from src.benchmarking import BenchmarkSuite, BenchmarkRunner

# Create a benchmark suite
suite = BenchmarkSuite.from_problem_generators(
    name="custom_benchmark",
    problem_classes=[TSP, Knapsack],
    difficulties=['easy']
)

# Define solver configurations
solver_configs = [
    {'name': 'default', 'use_gomory_cuts': False, 'early_stop_gap': 0.0},
    {'name': 'custom_config', 'use_gomory_cuts': True, 'early_stop_gap': 0.1}
]

# Create the benchmark runner
runner = BenchmarkRunner(
    suite=suite,
    output_dir="custom_results",
    time_limit=30.0,
    solver_configs=solver_configs,
    parallel=True
)

# Run benchmarks
results = runner.run_benchmark()

# Visualize results
runner.visualize_results(results)

# Compare solver configurations
runner.compare_solvers(results)
```

## Extending the Framework

To extend the benchmarking framework:

1. Add new problem types by implementing the `OptimizationProblem` interface
2. Add custom solver configurations by modifying the `solver_configs` list
3. Create custom visualization methods in the `BenchmarkRunner` class
4. Implement new metric calculations in the problem class methods

For more detailed information, see the benchmarking module's README at `src/benchmarking/README.md`.