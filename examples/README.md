# CHOP Examples

This directory contains example scripts that demonstrate how to use the CHOP framework for combinatorial optimization problems.

## Basic Examples

- **simple_ilp.py**: Basic example of solving a simple ILP using the BnB solver
- **tsp_example.py**: Example of solving Traveling Salesman Problem instances
- **knapsack_example.py**: Example of solving Knapsack Problem instances
- **assignment_example.py**: Example of solving Assignment Problem instances
- **bin_packing_example.py**: Example of solving Bin Packing Problem instances
- **set_cover_example.py**: Example of solving Set Cover Problem instances

## Benchmark Examples

- **benchmark_example.py**: Example of creating and running benchmark suites
  - Creates benchmark suites from predefined and generated instances
  - Runs benchmarks with different solver configurations
  - Analyzes results and generates visualizations
  - Compares performance across problem types and solver configurations

- **benchmark_visualization.py**: Standalone script for visualizing saved benchmark results
  - Usage: `python benchmark_visualization.py --results <path_to_results.json> [--output <output_dir>] [--format png|pdf|svg] [--compare]`
  - Loads previously saved benchmark results and generates visualizations
  - Can be used to compare different solver configurations

## Running the Examples

Most examples can be run directly:

```bash
python examples/simple_ilp.py
python examples/tsp_example.py
python examples/benchmark_example.py
```

For the benchmark visualization script, you need to provide the path to a benchmark results JSON file:

```bash
python examples/benchmark_visualization.py --results benchmark_results/comprehensive_benchmark_20240306_123456.json --compare
```

## Creating Custom Benchmarks

You can create your own benchmark suites by:

1. Creating a `BenchmarkSuite` instance from problem generators or predefined instances
2. Configuring a `BenchmarkRunner` with your desired solver configurations
3. Running the benchmark on specific problem types or difficulties
4. Analyzing and visualizing the results

Example:

```python
from src.problems import TSP, Knapsack
from src.benchmarking import BenchmarkSuite, BenchmarkRunner

# Create a benchmark suite
suite = BenchmarkSuite.from_problem_generators(
    name="custom_benchmark",
    problem_classes=[TSP, Knapsack],
    difficulties=['easy', 'medium']
)

# Define solver configurations
solver_configs = [
    {'name': 'default', 'use_gomory_cuts': False},
    {'name': 'with_cuts', 'use_gomory_cuts': True}
]

# Create and run the benchmark
runner = BenchmarkRunner(
    suite=suite,
    output_dir="my_benchmark_results",
    solver_configs=solver_configs
)

# Run the benchmark
results = runner.run_benchmark()

# Visualize the results
runner.visualize_results(results)
```

## Benchmark Metrics

The benchmark suite collects and analyzes various metrics:

1. **Instance Metrics**:
   - Problem size (variables, constraints)
   - LP relaxation properties
   - Problem-specific structural properties

2. **Solver Metrics**:
   - Solution quality (objective value, optimality gap)
   - Computational efficiency (time, nodes processed)
   - Branch-and-bound statistics

These metrics can help identify which problem characteristics correlate with difficulty and which solver configurations perform best for different problem types.