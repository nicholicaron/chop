<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/nicholicaron/chop">
    <img src="images/peutinger-table-map-1619.jpg" alt="Tabula Peutingeriana, a first century abstract depiction of roads as a network or graph" width="960" height="480">
  </a>

<h3 align="center">CHOP</h3>

  <p align="center">
CHOP (Combinatorial Heuristic Optimization Powerhouse) is a research project in which we explore ways to solve Combinatorial Optimization problems faster by using Deep Reinforcement Learning to learn better heuristics in Mixed-Integer Linear Program solvers.  
<br />
    <a href="https://github.com/nicholicaron/chop"><strong>Explore the docs (coming soon)»</strong></a>
    <br />
    <br />
    <a href="https://github.com/nicholicaron/chop">View Demo (coming soon)</a>
    ·
    <a href="https://github.com/nicholicaron/chop/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/nicholicaron/chop/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#dependencies">Dependencies</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

CHOP (Combinatorial Heuristic Optimization Powerhouse) is a research project focused on applying Deep Reinforcement Learning (DRL) and Graph Neural Networks (GNNs) to improve Integer Linear Programming (ILP) solvers. The core idea is to learn heuristics for branch-and-bound algorithms that can accelerate convergence to optimal solutions.

Key features of this project:
* Modular Branch-and-Bound implementation with pluggable strategies
* Various node selection and branching heuristics
* Visualization tools for tree exploration and solution analysis
* Framework for integrating RL-based heuristics and GNNs

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Python][python]][python-url]
* [![NumPy][numpy]][numpy-url]
* [![PyTorch][pytorch]][pytorch-url]
* [![NetworkX][networkx]][networkx-url]
* [![Matplotlib][matplotlib]][matplotlib-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

To set up the project locally, follow these steps:

### Dependencies

This project relies on several Python packages:
* Python 3.8+
* NumPy
* Numba
* NetworkX
* Matplotlib
* PyTorch
* PyTorch Geometric
* Jupyter Notebook (optional, for examples)

### Installation

1. Clone the repository
   ```sh
   git clone https://github.com/nicholicaron/chop.git
   cd chop
   ```

2. Create a virtual environment
   ```sh
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the package and dependencies
   ```sh
   pip install -e .
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### Basic ILP Example

```python
from src.core.solver import BranchAndBoundSolver
from src.strategies.priority_queue import BestBoundPrioritizer
from src.strategies.branching import MostFractionalBranching
import numpy as np

# Create a solver with chosen strategies
solver = BranchAndBoundSolver(
    prioritizer=BestBoundPrioritizer(),
    branching_strategy=MostFractionalBranching()
)

# Define a simple ILP problem
c = np.array([1, 2])  # Objective coefficients
A_ub = np.array([[-1, 1], [1, 1]])  # Inequality constraints
b_ub = np.array([1, 2])  # RHS values

# Solve the problem
solution, value, nodes, _ = solver.solve(c, A_ub, b_ub, problem_name="Simple Example")

print(f"Optimal solution: {solution}")
print(f"Objective value: {value}")
print(f"Nodes explored: {nodes}")
```

### Using Problem Generation Framework

The project provides a comprehensive framework for generating optimization problems:

```python
from src.problems import TSP, Knapsack, Assignment, BinPacking, SetCover

# Create a TSP instance
tsp = TSP.generate_random_instance(n_cities=10, seed=42)

# Convert to ILP formulation
c, A_eq, b_eq, A_ub, b_ub = tsp.to_ilp()

# Solve with Branch-and-Bound
solver = BranchAndBoundSolver()
solution, obj_value, nodes, _ = solver.solve(
    c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq
)

# Visualize solution
tsp.visualize_solution(solution, is_optimal=True)
```

### Running Examples

The repository includes example scripts for each problem type in the `examples/` directory:

1. General problem framework demonstration
   ```sh
   python examples/problem_instances.py
   ```

2. Problem-specific examples
   ```sh
   python examples/tsp_example.py --visualize
   python examples/knapsack_example.py
   python examples/assignment_example.py
   python examples/bin_packing_example.py
   python examples/set_cover_example.py
   ```

3. Simple ILP examples without problem classes
   ```sh
   python examples/simple_ilp.py --visualize
   ```

4. Benchmarking examples
   ```sh
   # Run the comprehensive benchmark example
   python examples/benchmark_example.py
   
   # Visualize saved benchmark results
   python examples/benchmark_visualization.py --results benchmark_results/some_results.json --compare
   ```

The `--visualize` flag generates branch-and-bound tree visualizations.

### Using the Benchmarking Framework

The benchmarking framework makes it easy to evaluate and compare solver performance:

```python
from src.problems import TSP, Knapsack
from src.benchmarking import BenchmarkSuite, BenchmarkRunner

# Create a benchmark suite with predefined instances
suite = BenchmarkSuite.from_predefined_instances(
    name="predefined_benchmark", 
    problem_classes=[TSP, Knapsack]
)

# Define different solver configurations to compare
solver_configs = [
    {'name': 'default', 'use_gomory_cuts': False, 'early_stop_gap': 0.0},
    {'name': 'with_cuts', 'use_gomory_cuts': True, 'early_stop_gap': 0.0},
    {'name': 'early_stop', 'use_gomory_cuts': False, 'early_stop_gap': 0.05}
]

# Create and run the benchmark
runner = BenchmarkRunner(
    suite=suite,
    output_dir="benchmark_results",
    time_limit=60.0,  # 1 minute per instance
    solver_configs=solver_configs,
    parallel=True  # Run benchmarks in parallel
)

# Run the benchmark
results = runner.run_benchmark()

# Visualize the results
runner.visualize_results(results)

# Compare solver configurations
runner.compare_solvers(results)
```

The framework generates detailed visualizations and statistics, helping identify:
- Which problem characteristics affect solution difficulty
- How solver configurations perform on different problem types
- Where performance bottlenecks occur
- The impact of cutting planes and early stopping criteria

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- PROJECT STRUCTURE -->
## Project Structure

The project is organized into the following modules:

```
chop/
├── examples/                # Example scripts demonstrating usage
│   ├── problem_instances.py # General problem framework demo
│   ├── tsp_example.py       # TSP examples
│   ├── knapsack_example.py  # Knapsack examples
│   ├── assignment_example.py # Assignment examples
│   ├── bin_packing_example.py # Bin Packing examples
│   ├── set_cover_example.py  # Set Cover examples
│   ├── simple_ilp.py        # Simple ILP problems
│   ├── benchmark_example.py # Benchmark suite example
│   └── benchmark_visualization.py # Visualize benchmark results
│
├── src/                     # Source code
│   ├── benchmarking/        # Benchmarking framework
│   │   ├── metrics.py       # Instance and solver metrics
│   │   ├── runner.py        # Benchmark execution and visualization
│   │   ├── suite.py         # Benchmark suite management
│   │   └── README.md        # Benchmarking documentation
│   │
│   ├── core/                # Core B&B components
│   │   ├── node.py          # B&B tree node representation
│   │   ├── priority_queue.py # Priority queue implementation
│   │   └── solver.py        # Main B&B solver
│   │
│   ├── problems/            # Problem generation framework
│   │   ├── base.py          # Abstract base class for all problems
│   │   ├── tsp.py           # Traveling Salesman Problem
│   │   ├── knapsack.py      # Knapsack Problem
│   │   ├── assignment.py    # Assignment Problem
│   │   ├── bin_packing.py   # Bin Packing Problem
│   │   ├── set_cover.py     # Set Cover Problem
│   │   └── README.md        # Documentation for problem framework
│   │
│   ├── strategies/          # Pluggable strategies
│   │   ├── branching.py     # Variable selection strategies
│   │   └── priority_queue.py # Node selection strategies
│   │
│   ├── utils/               # Utility functions
│   │   └── logging.py       # Logging and performance tracking
│   │
│   ├── simplex.py           # Simplex LP solver
│   └── pivoting.py          # Pivoting operations for simplex
│
├── logs/                    # Log files (not tracked in git)
├── plots/                   # Generated visualizations (not tracked in git)
├── benchmark_results/       # Benchmark results and visualizations (not tracked in git)
├── pyproject.toml           # Project metadata and dependencies
└── README.md                # This file
```

### Core Components

1. **Branch-and-Bound Solver**: Modular implementation of the B&B algorithm with:
   - Customizable node selection strategies
   - Pluggable branching heuristics
   - Visualization capabilities
   - Performance tracking

2. **Problem Generation Framework**: Comprehensive framework for creating and solving optimization problems:
   - Common interface for all problem types
   - Random instance generation with configurable parameters
   - Benchmark suite generation at different difficulty levels
   - Problem-specific visualization tools
   - Solution validation

3. **Benchmarking Framework**: System for evaluating solver performance across problem types:
   - Instance metrics for measuring problem characteristics
   - Solver metrics for evaluating computational efficiency
   - Benchmark suite management and execution
   - Comprehensive visualization and statistical analysis
   - Parallel execution for faster benchmarking
   - Solver configuration comparison

4. **Simplex Solver**: Custom implementation of the simplex algorithm for solving LP relaxations:
   - Efficient pivoting operations
   - Numerical stability improvements
   - Tableau maintenance for cut generation

### Problem Types

1. **Traveling Salesman Problem (TSP)**:
   - Find the shortest tour visiting all cities
   - Subtour elimination via lazy constraints
   - Network visualization of tours

2. **Knapsack Problem**:
   - Maximize value with limited capacity
   - Capacity constraint handling
   - Item visualization with value/weight ratios

3. **Assignment Problem**:
   - Minimize cost of agent-task assignments
   - One-to-one matching constraints
   - Bipartite graph visualization

4. **Bin Packing Problem**:
   - Minimize number of bins used
   - Bin capacity constraints
   - Visual representation of packed items

5. **Set Cover Problem**:
   - Minimize cost of selected sets to cover all elements
   - Coverage constraints
   - Visualization of coverage relationships

### Strategies

1. **Branching Strategies**: 
   - Most Fractional: Branches on the variable with value closest to 0.5
   - Pseudo Cost: Uses historical performance to guide branching
   - Strong Branching: Evaluates multiple variables by solving child LPs
   - Reliability Branching: Hybrid of pseudo cost and strong branching

2. **Priority Queue Strategies**:
   - Best Bound: Prioritizes nodes with best objective bounds
   - Depth First: Prioritizes nodes with greatest depth
   - Breadth First: Prioritizes nodes with smallest depth
   - Hybrid: Weighted combination of bound and depth
   - Decaying Best Bound: Best bound with exponential depth decay

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

### 1. Core Functionality
- [x] Implement basic Branch-and-Bound algorithm
- [x] Create modular structure with pluggable components
- [x] Develop visualization capabilities
- [x] Integrate performance monitoring and logging

### 2. Branching Strategies
- [x] Most fractional variable selection
- [x] Pseudo-cost branching
- [x] Strong branching
- [x] Reliability branching
- [ ] Full strong branching
- [ ] Gradient-based branching

### 3. Node Selection Strategies
- [x] Best-bound search
- [x] Depth-first search
- [x] Breadth-first search
- [x] Hybrid approaches
- [ ] Estimated value with look-ahead

### 4. Cutting Planes
- [x] Basic Gomory cuts
- [ ] Mixed-integer rounding cuts
- [ ] Lift-and-project cuts
- [ ] Clique cuts for set covering problems

### 5. Problem Library
- [x] General ILP solver
- [x] Traveling Salesman Problem
- [x] Knapsack Problem
- [x] Set Covering
- [x] Bin Packing
- [x] Assignment Scheduling

### 6. Benchmarking Framework
- [x] Instance metrics for problem characterization
- [x] Solver performance metrics
- [x] Benchmark suite generation and management
- [x] Visualization and statistical analysis
- [x] Parallel benchmark execution
- [x] Solver configuration comparison
- [ ] Integration with external solvers (e.g., CPLEX, Gurobi)
- [ ] Advanced problem-specific metrics

### 7. RL Integration
- [ ] State representation for B&B nodes
- [ ] Action spaces for node and variable selection
- [ ] Reward functions balancing quality and efficiency
- [ ] Training pipeline for RL agents
- [ ] GNN integration for learning tree structures

See the [open issues](https://github.com/nicholicaron/chop/issues) for a full list of proposed features and known issues.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- CONTACT -->
## Contact

Nicholi Caron - nmooreca@students.kennesaw.edu

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
* I want to give a huge thank you to [Dr. Misha Lavrov](https://misha.fish/) for supervising this research project.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/nicholicaron/chop.svg?style=for-the-badge
[contributors-url]: https://github.com/nicholicaron/chop/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/nicholicaron/chop.svg?style=for-the-badge
[forks-url]: https://github.com/nicholicaron/chop/network/members
[stars-shield]: https://img.shields.io/github/stars/nicholicaron/chop.svg?style=for-the-badge
[stars-url]: https://github.com/nicholicaron/chop/stargazers
[issues-shield]: https://img.shields.io/github/issues/nicholicaron/chop.svg?style=for-the-badge
[issues-url]: https://github.com/nicholicaron/chop/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge 
[licnse-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/nicholicaron
[product-screenshot]: images/screenshot.png
[python]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
[numpy]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[numpy-url]: https://numpy.org/
[pytorch]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[pytorch-url]: https://pytorch.org/
[networkx]: https://img.shields.io/badge/NetworkX-2C3E50?style=for-the-badge&logo=python&logoColor=white
[networkx-url]: https://networkx.org/
[matplotlib]: https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white
[matplotlib-url]: https://matplotlib.org/
