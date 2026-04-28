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
CHOP (Combinatorial Heuristic Optimization Powerhouse) is a research project that uses Deep Reinforcement Learning to learn node-selection heuristics for branch-and-bound MILP solvers.
<br />
    <a href="#results">Jump to results</a>
    ·
    <a href="https://github.com/nicholicaron/chop/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/nicholicaron/chop/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

> **Status (April 2026):** the RL training pipeline is functional and there are two headline results — see [Results](#results):
> 1. On random Knapsack, REINFORCE recovers the best-bound heuristic from scratch in ~50 s of CPU training and generalizes to unseen problem sizes.
> 2. On Set Cover (where best-bound is provably suboptimal), the same REINFORCE pipeline learns a policy that **beats best-bound by ~40%** (11.3 vs 19.0 nodes), comparable to the strongest non-LP heuristic for that problem class.



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
    <li><a href="#results">Results</a></li>
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
* Gymnasium (for RL environments)
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



<!-- RESULTS -->
## Results

Two complementary experiments. On Knapsack the learned policy converges to the strongest classical heuristic; on Set Cover (where the best-bound heuristic is provably suboptimal) it actually beats it.

### Shared setup

* **Algorithm:** REINFORCE with EMA baseline, entropy bonus (0.01-0.02), gradient clipping
* **Policy:** small MLP (2x64 tanh) over a fixed-shape vector of top-K candidate-node features (K=16)
* **Action:** the agent emits K real-valued scores over the K best-LP-bound candidates; argmax selects which open node the B&B solver expands next
* **Reward:** -1 per node expanded, +5 on each new incumbent, +50 on proving optimality
* **Hardware:** all training and evaluation on laptop CPU; no GPU required

### Result 1 — Knapsack: matches best-bound, generalizes across sizes

* **Problem:** random binary Knapsack, `n_items=25`, "medium" difficulty, 800 training episodes (~50 s)

Held-out evaluation on 40 fresh instances, deterministic policy:

| Policy                   | Nodes to optimum (mean ± std) | vs. learned |
|--------------------------|-------------------------------|-------------|
| **Learned (REINFORCE)**  | **66.7 ± 52.9**               | 1.00x       |
| BestBound (classical)    | 66.7 ± 52.9                   | 1.00x       |
| DepthFirst               | 86.1 ± 61.3                   | 1.29x       |
| BreadthFirst             | 250.3 ± 69.2                  | 3.75x       |
| Random                   | 207.1 ± 60.2                  | 3.10x       |

The trained policy matches BestBound exactly on the held-out set (the policy converged to the same node ordering on every test instance) and beats Random / BreadthFirst by 3-4x.

![Learning curve on Knapsack(25, medium)](plots/reinforce_learning_curve_n25_medium.png)

The same policy generalizes to unseen `n_items` ∈ {15, 20, 25, 30} without retraining — the purple "Learned" line is hidden directly under the green "BestBound" line:

![Generalization across Knapsack sizes](plots/generalization_across_sizes.png)

### Result 2 — Set Cover: beats best-bound by ~40%

* **Problem:** random Set Cover, `n_elements=50`, `n_sets=80`, density=0.10, 600 training episodes (~105 s)
* **Why this regime?** Set Cover's LP relaxation is highly fractional, so a greedy best-bound traversal dives into deep fractional subtrees before reaching an integer solution. Random/breadth-first stumble onto integer solutions sooner. This is exactly the kind of problem where a learned policy has room to outperform the classical heuristic.

Held-out evaluation on 25 fresh instances, deterministic policy:

| Policy                   | Nodes to optimum (mean ± std) | vs. learned |
|--------------------------|-------------------------------|-------------|
| **Learned (REINFORCE)**  | **11.3 ± 9.7**                | 1.00x       |
| BestBound (classical)    | 19.0 ± 15.0                   | **1.68x worse** |
| DepthFirst               | 9.7 ± 8.4                     | 0.86x       |
| BreadthFirst             | 12.2 ± 11.0                   | 1.08x       |
| Random                   | 13.2 ± 9.2                    | 1.17x       |

The trained policy explores **40% fewer nodes than best-bound** and is competitive with depth-first (the strongest non-LP heuristic on this distribution).

![Set Cover learning curve](plots/setcover_learning_curve_mlp.png)
![Set Cover comparison](plots/setcover_comparison_mlp.png)

### Reproducing

```sh
# Knapsack experiment + plots (~50 s)
python examples/train_reinforce.py --policy mlp --episodes 800 --n_items 25 \
    --difficulty medium --max_steps 600 --time_limit 30 --n_eval 40 \
    --save checkpoints/reinforce_knapsack_n25.pt

# Generalization eval across n_items (~1 min)
python examples/eval_generalization.py --sizes 15 20 25 30 --n_eval 25

# Set Cover experiment + plots (~2 min) -- the "RL beats best-bound" result
python examples/train_setcover.py --policy mlp --episodes 600 --n_eval 25
```

Plots land under `plots/`, raw stats under `checkpoints/`.

### Caveats and ongoing work

* **GNN policy** — the GNN scaffolding works (clean acting + training pipeline against the B&B tree as a graph), but the trained GNN's deterministic-eval mode has so far collapsed to best-bound on Set Cover (per-episode performance during stochastic training was competitive with the MLP, ~10-15 nodes). Probably a tuning issue (entropy regularization, longer training); listed under [Roadmap](#roadmap).
* **Variance** — Set Cover instances at this size have high run-to-run variance; the 40% gap reported above survives n=25 held-out instances but a finer experiment with larger n would tighten the confidence interval.
* **Single problem class** — the "beats best-bound" result is on Set Cover specifically; we have not yet shown the policy beats best-bound on Bin Packing or other classes.

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

5. RL environment & training
   ```sh
   # Smoke test: confirm action causally affects the search
   python examples/rl_env_smoke_test.py

   # Train a node-selection policy with REINFORCE on Knapsack (~50s on CPU)
   python examples/train_reinforce.py --policy mlp --episodes 800 \
       --n_items 25 --difficulty medium

   # Train on Set Cover -- the regime where RL beats best-bound (~2 min)
   python examples/train_setcover.py --policy mlp --episodes 600

   # Generalization eval of a trained policy across problem sizes
   python examples/eval_generalization.py --sizes 15 20 25 30 --n_eval 25

   # Side-by-side comparison plot using a trained checkpoint
   python examples/rl_branch_and_bound_example.py

   # Heuristic baseline scan on Set Cover (find which configs have headroom)
   python examples/setcover_baseline_check.py
   ```

   Both training scripts accept `--policy {mlp, gnn}`. The MLP is the recommended default for now; the GNN is functional but under-tuned (see [Roadmap](#roadmap)).

The `--visualize` flag on the legacy `simple_ilp.py` / `*_example.py` scripts generates branch-and-bound tree visualizations.

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

### Using the RL Environment

The reinforcement learning environment provides a Gymnasium-compatible interface for training agents to optimize the Branch-and-Bound algorithm:

```python
import numpy as np
import gymnasium as gym
from src.problems import Knapsack
from src.environments import BranchAndBoundEnv

# Create a problem generator
def knapsack_generator():
    return Knapsack.generate_random_instance(n_items=20, difficulty='medium')

# Create the environment
env = BranchAndBoundEnv(
    problem_generator=knapsack_generator,
    max_steps=100,
    reward_type='improvement',
    observation_type='graph',  # Can be 'graph' or 'vector'
    render_mode='human'
)

# Reset the environment to get initial state
observation, info = env.reset(seed=42)

# Agent interaction loop
done = False
total_reward = 0

while not done:
    # Agent selects an action to modify the priority queue
    # Here we just use a random action for demonstration
    action = np.random.uniform(-1.0, 1.0, size=(1,))
    
    # Environment step
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Render state (shows rich problem-specific visualization)
    env.render()
    
    # Update tracking
    total_reward += reward
    done = terminated or truncated

# Display results
print(f"Total reward: {total_reward:.2f}")
print(f"Objective value: {info['current_best_obj']:.2f}")
print(f"Nodes explored: {info['nodes_explored']}")
```

The environment includes:
- Rich state representations as graphs (PyTorch Geometric) or vectors
- Action spaces for influencing the priority queue ordering
- Multiple reward functions for different learning objectives
- Enhanced visualizations specific to each problem type
- Integration with all problem classes in the framework

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
│   ├── benchmark_visualization.py # Visualize benchmark results
│   ├── rl_branch_and_bound_example.py # RL environment example
│   └── rl_environment_visualization_test.py # Rich visualization examples
│
├── src/                     # Source code
│   ├── agents/              # Learnable RL agents (NEW)
│   │   ├── policy.py        # MLP policy: scores top-K candidate nodes
│   │   ├── gnn_policy.py    # GNN policy: GCN over the full B&B tree
│   │   └── reinforce.py     # REINFORCE-with-baseline trainer (policy-agnostic)
│   │
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
│   ├── environments/        # RL environments
│   │   ├── __init__.py      # Environment exports
│   │   └── branch_and_bound_env.py # Gymnasium-compatible B&B environment + heuristic agents
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
│   │   ├── logging.py       # Logging and performance tracking
│   │   └── eval.py          # Shared eval helpers (env factories, metrics) (NEW)
│   │
│   ├── simplex.py           # Simplex LP solver
│   └── pivoting.py          # Pivoting operations for simplex
│
├── checkpoints/             # Trained policy weights + training stats (not tracked in git)
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

2. **RL Environment**: Gymnasium-compatible environment for reinforcement learning:
   - Graph and vector state representations
   - Action space for priority queue reordering
   - Multiple reward functions (time-based, improvement-based)
   - Rich problem-specific visualizations
   - Integration with PyTorch Geometric for GNN-based agents

3. **Problem Generation Framework**: Comprehensive framework for creating and solving optimization problems:
   - Common interface for all problem types
   - Random instance generation with configurable parameters
   - Benchmark suite generation at different difficulty levels
   - Problem-specific visualization tools
   - Solution validation

4. **Benchmarking Framework**: System for evaluating solver performance across problem types:
   - Instance metrics for measuring problem characteristics
   - Solver metrics for evaluating computational efficiency
   - Benchmark suite management and execution
   - Comprehensive visualization and statistical analysis
   - Parallel execution for faster benchmarking
   - Solver configuration comparison

5. **Simplex Solver**: Custom implementation of the simplex algorithm for solving LP relaxations:
   - Efficient pivoting operations
   - Numerical stability improvements
   - Tableau maintenance for cut generation

### Problem Types

1. **Traveling Salesman Problem (TSP)**:
   - Find the shortest tour visiting all cities
   - Subtour elimination via lazy constraints
   - Map-based visualization with directional arrows
   - Color-coded subtours with detailed statistics
   - Tour validity tracking and visualization

2. **Knapsack Problem**:
   - Maximize value with limited capacity
   - Capacity constraint handling
   - Item visualization with value/weight ratios
   - Backpack representation showing selected items
   - Dynamic tracking of capacity utilization

3. **Assignment Problem**:
   - Minimize cost of agent-task assignments
   - One-to-one matching constraints
   - Bipartite graph visualization
   - Assignment tracking and cost visualization

4. **Bin Packing Problem**:
   - Minimize number of bins used
   - Bin capacity constraints
   - 3D visualization of bin packing with items as cubes
   - Color-coded items with size representation
   - Utilization statistics for each bin

5. **Set Cover Problem**:
   - Minimize cost of selected sets to cover all elements
   - Coverage constraints
   - Visualization of coverage relationships
   - Interactive coverage display

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
- [x] State representation for B&B nodes with graph and vector formats
- [x] Action spaces for priority queue reordering (top-K node scoring)
- [x] Reward functions balancing quality and efficiency
- [x] Gymnasium-compatible environment for training RL agents
- [x] Enhanced visualizations for all problem types
- [x] **Training pipeline for RL agents (REINFORCE-with-baseline)**
- [x] **Knapsack: trained policy matches BestBound, generalizes across n_items**
- [x] **Set Cover: trained policy beats BestBound by ~40% on the adversarial regime**
- [x] **GNN policy** that consumes the B&B tree as a graph (functional; final-eval performance still under tuning)
- [ ] PPO + minibatching for larger problem sizes (n_items >= 50)
- [ ] Curriculum learning across problem-size schedules
- [ ] Bin Packing experiments + adversarial-instance generator

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
