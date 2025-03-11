# Problem Generation Framework

This module provides a standardized interface for creating, solving, and visualizing different optimization problems.

## Key Components

- **Base Interface**: Abstract class defining the required methods for all optimization problems
- **Problem Implementations**: Current implementations include TSP, Knapsack, Assignment, Bin Packing, and Set Cover
- **Benchmark Generators**: Utilities for creating problem instances at different difficulty levels
- **Visualization**: Tools for visualizing problem instances and solutions

## Usage

To use the framework, you can:

1. Load predefined problem instances:

```python
from src.problems import get_predefined_instances

instances = get_predefined_instances()
tsp_instances = instances['tsp']
knapsack_instances = instances['knapsack']
```

2. Generate random instances:

```python
from src.problems import TSP, Knapsack

# Generate a random TSP instance
tsp = TSP.generate_random_instance(
    n_cities=10,
    seed=42,
    name="Random_TSP"
)

# Generate a random Knapsack instance
knapsack = Knapsack.generate_random_instance(
    n_items=15,
    capacity_factor=0.5,
    seed=42
)
```

3. Generate benchmark suites:

```python
# Generate benchmark suites at different difficulty levels
tsp_suite = TSP.generate_benchmark_suite(['easy', 'medium', 'hard'])
knapsack_suite = Knapsack.generate_benchmark_suite(['easy', 'medium', 'hard'])
```

4. Solve instances with the Branch-and-Bound solver:

```python
from src.core.solver import BranchAndBoundSolver

solver = BranchAndBoundSolver()

# Convert problem to ILP form
c, A_eq, b_eq, A_ub, b_ub = problem.to_ilp()

# Solve
solution, obj_value, node_count, _ = solver.solve(
    c=c,
    A_ub=A_ub,
    b_ub=b_ub,
    A_eq=A_eq,
    b_eq=b_eq,
    problem_name=problem.name
)

# Validate solution
is_valid, actual_obj = problem.validate_solution(solution)
```

5. Visualize instances and solutions:

```python
# Visualize problem instance
instance_image = problem.visualize_instance()

# Visualize solution (basic)
solution_image = problem.visualize_solution(solution, is_optimal=True)

# Enhanced visualization with extra information (e.g., for RL environments)
rich_viz = problem.visualize_solution(
    solution,
    is_optimal=True,
    step=42,                   # Current step in solving process
    nodes_explored=156,        # Number of nodes explored
    elapsed_time=10.5,         # Time spent solving
    best_obj_value=568.7,      # Best objective value found
    animated=True,             # Indicates this is part of a sequence
    title="Custom Solution Visualization"
)
```

Each problem type has specialized visualizations:
- **TSP**: Map-based display with subtour detection and tour validity information
- **Knapsack**: Backpack representation with capacity utilization and item layout
- **Bin Packing**: 3D visualization of bins with packed items and utilization metrics

## Example Scripts

Check the `examples` directory for demonstration scripts:

- `problem_instances.py`: Shows how to use the problem generation framework
- `tsp_example.py`: Demonstrates TSP instances and solutions
- `knapsack_example.py`: Demonstrates Knapsack instances and solutions
- `assignment_example.py`: Demonstrates Assignment problem instances and solutions
- `bin_packing_example.py`: Demonstrates Bin Packing instances and solutions
- `set_cover_example.py`: Demonstrates Set Cover instances and solutions
- `simple_ilp.py`: Shows how to work with ILPs directly without problem classes
- `rl_branch_and_bound_example.py`: Demonstrates the RL environment with different agents
- `rl_environment_visualization_test.py`: Shows enhanced visualizations for all problem types

## Implementing New Problem Types

To add a new problem type, create a new class that inherits from `OptimizationProblem` and implements all required methods:

```python
from src.problems import OptimizationProblem

class NewProblem(OptimizationProblem):
    # Implement all required methods
    def to_ilp(self):
        # Convert to ILP form
        
    def validate_solution(self, solution):
        # Validate a solution
        
    # ... and so on
```

Then update the `__init__.py` file to expose your new problem class.