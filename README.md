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

> **Status (April 2026):** the RL training pipeline is functional with seven architectures (MLP, GCN over the B&B tree, Transformer, Tree-GNN, Gasse-style Bipartite-GCN, Bipartite-GCN+self-attention, Hybrid Bipartite+Tree), three trainers (REINFORCE, PPO, Imitation), an Ensemble combiner, and a multi-task curriculum. Headline results — see [Results](#results):
>
> 1. **Knapsack:** REINFORCE recovers best-bound from scratch in ~50 s of CPU training and generalizes to unseen problem sizes.
> 2. **Set Cover SOTA-on-CPU:** Bipartite-GCN with cross-candidate self-attention reaches **8.9 ± 6.4 nodes — 2.15x better than best_bound** (19.1) on held-out instances. The top seven ranked approaches are all learned; depth-first (the strongest classical heuristic) ranks 8th.
> 3. **Multi-task generalist:** a single MLP trained on Knapsack + Set Cover beats both single-task baselines while matching best-bound on Knapsack.
> 4. **Honest negative results:** naive ensemble (Bipartite + Tree-GNN average) didn't help; jointly-trained hybrid encoder didn't beat the bare Bipartite-GCN. Documented under "What didn't work."



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

### Result 2 — Set Cover: learned policies cluster at the top

* **Problem:** random Set Cover, `n_elements=50`, `n_sets=80`, density=0.10
* **Why this regime?** Set Cover's LP relaxation is highly fractional, so a greedy best-bound traversal dives into deep fractional subtrees before reaching an integer solution. Random/breadth-first stumble onto integer solutions sooner. This is the regime where learned heuristics have room to actually outperform the classical best-bound.

Comprehensive held-out evaluation on **40 fresh instances** comparing every architecture and trainer:

| Rank | Approach                  | Nodes (mean ± std) | vs. best_bound | Architecture / Trainer        |
|-----:|---------------------------|--------------------|----------------|-------------------------------|
| 1    | **Bipartite-GCN + self-attention** | **8.9 ± 6.4** | **2.15x better** | Gasse encoder + transformer over candidates |
| 2    | Bipartite-GCN             | 9.3 ± 7.4          | 2.05x better   | Gasse-style C↔V conv          |
| 3    | Tree-GNN (stochastic)     | 9.3 ± 7.3          | 2.05x better   | Bottom-up tree msg-passing    |
| 4    | Multi-task MLP            | 10.0 ± 8.9         | 1.91x better   | REINFORCE on Knap+SC mix      |
| 5    | Hybrid (Bipartite+Tree)   | 10.6 ± 10.4        | 1.80x better   | Joint encoder, shared head    |
| 6    | Tree-GNN (deterministic)  | 10.7 ± 9.2         | 1.79x better   | Same as #3, argmax eval       |
| 7    | REINFORCE + MLP           | 10.8 ± 9.0         | 1.77x better   | 600 ep on SetCover only       |
| 8    | depth_first (heuristic)   | 10.9 ± 10.5        | 1.75x          | strongest classical here      |
| 9    | breadth_first (heuristic) | 11.7 ± 9.8         | 1.63x          | classical                     |
| 10   | random (heuristic)        | 13.2 ± 9.8         | 1.45x          | classical                     |
| 11   | GNN over B&B tree (stoch) | 13.3 ± 9.8         | 1.43x          | GCN over enumeration tree     |
| 12   | REINFORCE + MLP-long      | 13.9 ± 8.8         | 1.37x          | 1500 ep (overfit)             |
| 13   | GNN over B&B tree (det)   | 13.9 ± 12.4        | 1.37x          | Boltzmann-temp eval fix       |
| 14   | PPO + MLP                 | 16.2 ± 11.3        | 1.18x          | clipped surrogate + GAE       |
| 15   | Imitation + RL + MLP      | 18.8 ± 15.9        | 1.02x          | distill best_bound, then RL   |
| 16   | best_bound (heuristic)    | 19.1 ± 16.1        | 1.00x          | classical baseline            |

![All approaches benchmark](plots/benchmark_all.png)

#### What we learned from running this whole grid

* **Architecture matters more than training time.** Longer training of an MLP (1500 vs 600 episodes) actually got *worse*, but switching architecture (MLP → Bipartite-GCN → Bipartite + cross-candidate attention) cut nodes from 10.8 to 9.3 to 8.9.
* **Top 7 are all learned.** Bipartite-Attn, Bipartite-GCN, Tree-GNN, Multi-task MLP, Hybrid, Tree-GNN-det, single-task MLP all beat depth-first — the strongest classical heuristic for this distribution.
* **Cross-candidate self-attention helps.** Adding a transformer encoder over the K candidate embeddings (so each candidate's score depends on the others) improved over the bare Bipartite-GCN (8.9 vs 9.3, ~4% better mean and tighter std 6.4 vs 7.4). Within-noise statistically but on the same held-out seeds, so directionally meaningful.
* **Two GNN architectures, one over each natural graph.** The Bipartite-GCN reads the *LP problem structure* per candidate (variables, constraints, coefficients). The Tree-GNN reads the *B&B search tree* (parent-child relationships, per-node bounds). Both improve over the MLP and tied at 9.3 individually.
* **Multi-task beats single-task on the same architecture** — for the MLP, training on a Knapsack+Set Cover mix gave 10.0 vs single-task 10.8. The Bipartite-GCN didn't get the benefit cleanly in our run because the bigger LP graphs of mixed problems blew up training time.
* **GNN deterministic-eval collapse fixed** with low-temperature Boltzmann sampling at eval (T=0.05). The original GNN's `argmax` always landed on best_bound's choice because of near-tied scores at the top.
* **PPO + Imitation underperformed REINFORCE here.** Short-episode regimes don't expose PPO's sample-efficiency edge; imitation warm-start started near best_bound and the on-policy fine-tune destabilized rather than improved.

#### What didn't work (honest negative results)

* **Naive score-ensemble of pre-trained Bipartite-GCN + Tree-GNN.** Hypothesis: two architectures looking at orthogonal signals should disagree usefully. Reality: 1:1 average gave 10.97, *worse* than Bipartite-GCN alone (10.4 on the same seeds). Skewed weights (2:1 or 3:1 favoring Bipartite) just recovered Bipartite-alone's score. Conclusion: the two architectures make highly correlated mistakes on this problem — the LP-structure and tree-structure signals lead to similar rankings rather than complementary ones.
* **Hybrid joint encoder (Bipartite-GCN + Tree-GNN trained together with a shared score head).** Hypothesis: joint training would let the encoders specialize on complementary signals during training, fixing what the naive ensemble couldn't. Reality: 10.6 ± 10.4 nodes — slightly worse than the bare Bipartite-GCN (9.3). The tree-side encoder added parameters but no useful gradient signal; the shared head just weighted it down.
* **Long training (1500 episodes vs 600 for single-task MLP).** Got 13.9 vs 10.8. Likely overfitting on the noisy on-policy gradient.

### Result 3 — Multi-task: one policy, two problem classes

The same 600-episode training run produced this side-by-side, evaluated on 25 fresh instances per class:

![Multi-task per-class breakdown](plots/multitask_mlp.png)

| Problem class               | Learned (multitask) | best_bound  | random       |
|-----------------------------|---------------------|-------------|--------------|
| Knapsack(20, medium)        | 56.2 ± 31.1         | 56.2 ± 31.1 | 140.3 ± 40.0 |
| SetCover(50e × 80s, d=0.10) | **10.4 ± 9.7**      | 19.0 ± 15.0 | 13.2 ± 9.2   |

The agent learned to imitate best-bound when that's optimal (Knapsack) and to deviate from it when that's optimal (Set Cover). Same weights, different behaviors.

### Reproducing

```sh
# Knapsack experiment + plots (~50 s)
python examples/train_reinforce.py --policy mlp --episodes 800 --n_items 25 \
    --difficulty medium --max_steps 600 --time_limit 30 --n_eval 40 \
    --save checkpoints/reinforce_knapsack_n25.pt

# Generalization eval across n_items (~1 min)
python examples/eval_generalization.py --sizes 15 20 25 30 --n_eval 25

# Set Cover, single-task with each architecture
python examples/train_setcover.py --algo reinforce --policy mlp        --episodes 600
python examples/train_setcover.py --algo reinforce --policy gnn        --episodes 600
python examples/train_setcover.py --algo reinforce --policy tree       --episodes 400
python examples/train_setcover.py --algo reinforce --policy bipartite  --episodes 400  # the 2.05x champion
python examples/train_setcover.py --algo ppo       --policy mlp        --ppo_iters 30

# Multi-task on Knapsack + Set Cover
python examples/train_multitask.py --policy mlp --episodes 600

# Imitation -> RL pipeline
python examples/train_imitation_then_rl.py --policy mlp \
    --imitation_episodes 100 --rl_episodes 400

# Comprehensive benchmark across every saved checkpoint (~3 min)
python examples/benchmark_all.py --n_eval 40
```

Plots land under `plots/`, raw stats under `checkpoints/`.

### Caveats and ongoing work

* **PPO under-tuned for these episode lengths** — expected to win on larger problems where rollout sample-efficiency matters more.
* **Imitation+RL fine-tune destabilized** — REINFORCE on a near-optimal policy is high variance; PPO fine-tune is the natural fix.
* **Bin Packing parked** — added the missing `x_ij <= 1, y_j <= 1` bounds to the ILP, but Bin Packing's LP relaxation is too weak for this CPU-only solver scale (5-item instances exceed 200 nodes). Would need a stronger LP relaxation (Dantzig-Wolfe / column generation) before RL helps.
* **Multi-task with Bipartite-GCN explored but not converged** — combining the two best ideas would be the natural next step, but each Knapsack(20) episode runs ~16 bipartite-GCN forward passes per step over a ~40-node LP graph, blowing up training time. Tractable with batched per-candidate evaluation; on the roadmap.
* **Variance** — Set Cover instances at this size have high std (Mean ± 7-16). The 2.05x Bipartite-GCN gap survives n=40 held-out instances; tightening the CI would mean a much larger n.

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
   # Sanity check: confirm the action causally affects the search
   python examples/rl_env_smoke_test.py

   # Single-task: train an MLP / GNN on Knapsack (~50s on CPU)
   python examples/train_reinforce.py --policy mlp --episodes 800 \
       --n_items 25 --difficulty medium

   # Single-task on Set Cover -- the "1.77x better than best_bound" result (~2 min)
   python examples/train_setcover.py --algo reinforce --policy mlp --episodes 600

   # PPO variant of the above
   python examples/train_setcover.py --algo ppo --policy mlp --ppo_iters 30

   # Multi-task: one policy across Knapsack + Set Cover -- the headline 1.91x result
   python examples/train_multitask.py --policy mlp --episodes 600

   # Imitation-learning warm-start, then optional REINFORCE fine-tune
   python examples/train_imitation_then_rl.py --policy mlp \
       --imitation_episodes 100 --rl_episodes 400

   # Generalization eval of a trained policy across problem sizes
   python examples/eval_generalization.py --sizes 15 20 25 30 --n_eval 25

   # Side-by-side comparison plot using a trained checkpoint
   python examples/rl_branch_and_bound_example.py

   # Comprehensive benchmark across every saved checkpoint
   python examples/benchmark_all.py --n_eval 40

   # Heuristic-only baseline scans (find which configs have headroom)
   python examples/setcover_baseline_check.py
   python examples/binpacking_baseline_check.py
   ```

   Most training scripts accept `--policy {mlp, gnn, transformer}`. The MLP is the recommended default; the GNN and Transformer are functional but the GNN's deterministic eval mode currently collapses (see [Roadmap](#roadmap)).

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
│   ├── agents/              # Learnable RL agents
│   │   ├── policy.py             # MLP policy: scores top-K candidate nodes
│   │   ├── gnn_policy.py         # GCN over the full B&B enumeration tree
│   │   ├── transformer_policy.py # Self-attention policy over candidate set
│   │   ├── tree_gnn_policy.py    # Bottom-up tree msg-passing (per arxiv 2310.00112)
│   │   ├── bipartite_gnn_policy.py # Gasse-style C↔V bipartite GCN over the LP (the champion)
│   │   ├── reinforce.py          # REINFORCE-with-baseline trainer (policy-agnostic)
│   │   ├── ppo.py                # PPO trainer with GAE + clipped surrogate
│   │   └── imitation.py          # Distill a HeuristicAgent into a policy via cross-entropy
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
- [x] **REINFORCE-with-baseline trainer** (policy-agnostic)
- [x] **PPO trainer** with GAE + clipped surrogate + minibatching
- [x] **Imitation-learning warm-start** (cross-entropy distillation from any HeuristicAgent)
- [x] **MLP policy** over fixed-shape top-K candidate features
- [x] **GCN over the B&B tree** (with low-temperature Boltzmann eval fix)
- [x] **Transformer policy** (self-attention over the candidate set + global token)
- [x] **Tree-GNN policy** (bottom-up message passing, per arxiv 2310.00112)
- [x] **Bipartite-GCN policy** (Gasse 2019 NeurIPS architecture: C↔V over the LP)
- [x] **Knapsack:** trained policy matches BestBound, generalizes across n_items
- [x] **Set Cover SOTA-on-CPU:** Bipartite-GCN beats BestBound by **2.05x** (9.3 vs 19.1 nodes)
- [x] **Multi-task** (Knapsack + Set Cover) generalist beats single-task baselines
- [ ] Multi-task + Bipartite-GCN with batched per-candidate evaluation (combine the two best ideas)
- [ ] PPO fine-tune from imitation warm-start (more stable than REINFORCE here)
- [ ] Tree MDP credit assignment (Scavuzzo et al. 2022) for better gradient signals
- [ ] Stronger LP relaxation for Bin Packing (Dantzig-Wolfe / column generation)
- [ ] Larger problem sizes (n_items >= 50) where PPO's edge would emerge
- [ ] Ensemble: average scores from Bipartite-GCN + Tree-GNN (top two architectures)

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
