# Benchmark Suite for CHOP

This module provides a comprehensive framework for benchmarking optimization problems and solver configurations in CHOP. It includes tools for creating benchmark suites, running benchmarks, analyzing results, and visualizing performance metrics.

## Key Components

### 1. Benchmark Metrics

- **InstanceMetrics**: Characterizes problem instances before solving
  - Basic properties (problem type, size, difficulty)
  - LP relaxation properties
  - Theoretical complexity measures
  - Problem-specific metrics

- **SolverMetrics**: Evaluates solver performance
  - Solution quality (objective value, optimality gap)
  - Computational efficiency (nodes, time)
  - Branch-and-bound statistics

### 2. Benchmark Suite

- **BenchmarkSuite**: Collection of problem instances for benchmarking
  - Create from problem generators or predefined instances
  - Access instances by problem type, difficulty
  - Calculate instance metrics automatically

- **BenchmarkResult**: Records results of running a solver on a problem instance
  - Instance information
  - Solver metrics
  - Configuration details

### 3. Benchmark Runner

- **BenchmarkRunner**: Executes benchmarks and analyzes results
  - Run benchmarks in parallel or sequentially
  - Compare different solver configurations
  - Analyze and visualize results

## Usage Examples

### Creating a Benchmark Suite

```python
from src.problems import TSP, Knapsack, Assignment
from src.benchmarking import BenchmarkSuite

# Create a suite from predefined instances
suite = BenchmarkSuite.from_predefined_instances(
    name="predefined_benchmark", 
    problem_classes=[TSP, Knapsack, Assignment]
)

# Create a suite from generated instances
suite = BenchmarkSuite.from_problem_generators(
    name="generated_benchmark",
    problem_classes=[TSP, Knapsack, Assignment],
    difficulties=['easy', 'medium', 'hard']
)
```

### Running Benchmarks

```python
from src.benchmarking import BenchmarkRunner

# Define solver configurations to compare
solver_configs = [
    {'name': 'default', 'use_gomory_cuts': False, 'early_stop_gap': 0.0},
    {'name': 'with_cuts', 'use_gomory_cuts': True, 'early_stop_gap': 0.0}
]

# Create the runner
runner = BenchmarkRunner(
    suite=suite,
    output_dir="benchmark_results",
    time_limit=60.0,  # 1 minute per instance
    solver_configs=solver_configs,
    parallel=True
)

# Run benchmarks
results = runner.run_benchmark()
```

### Analyzing and Visualizing Results

```python
# Analyze results
df = runner.analyze_results(results)

# Visualize results
runner.visualize_results(results)

# Compare solver configurations
runner.compare_solvers(results)
```

### Loading and Visualizing Saved Results

```python
from src.benchmarking import load_results, BenchmarkSuite, BenchmarkRunner

# Load previously saved results
results = load_results("benchmark_results/results_20240306_123456.json")

# Create a dummy suite and runner for visualization
suite = BenchmarkSuite("dummy")
runner = BenchmarkRunner(suite, output_dir="visualizations")

# Visualize the loaded results
runner.visualize_results(results)
```

## Visualizations Generated

The benchmark suite generates several visualizations to help analyze performance:

1. **Solution Time Distribution**: Boxplots showing the distribution of solution times by problem type and difficulty

2. **Nodes Processed Distribution**: Boxplots showing the distribution of nodes processed during branch-and-bound

3. **LP Gap Distribution**: Boxplots showing the distribution of LP relaxation gaps

4. **Solution Time Heatmap**: Heatmap showing average solution time by problem type and difficulty

5. **Configuration Comparison**: Bar charts comparing different solver configurations

6. **Metrics Correlation**: Heatmap showing correlation between instance metrics and solver performance

7. **Speedup Analysis**: When comparing solver configurations, shows the relative speedup of each configuration

## Extending the Benchmark Suite

To add new problem-specific metrics:

1. Add implementations of `size_metrics()` and `get_specific_metrics()` to your problem class
2. The metrics will automatically be included in benchmark results and visualizations

To add new solver configurations:

1. Define the configuration parameters in the `solver_configs` list
2. Run the benchmark with the new configurations
3. Use `compare_solvers()` to analyze the performance difference