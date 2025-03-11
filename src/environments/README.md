# Reinforcement Learning Environments

This module provides Gymnasium-compatible environments for training reinforcement learning agents to optimize various aspects of the Branch-and-Bound algorithm.

## Key Components

- **BranchAndBoundEnv**: Main RL environment for Branch-and-Bound with priority queue optimization
- **RLPrioritizer**: Custom prioritizer that uses agent-determined values
- **State Representations**: Both graph-based and vector-based state options
- **Enhanced Visualizations**: Rich, problem-specific visualizations for each problem type

## Environment Interface

The `BranchAndBoundEnv` class implements the standard Gymnasium interface:

- **reset(seed=None, options=None)**: Reset the environment with a new problem instance
- **step(action)**: Execute one step by applying the agent's action to the priority queue
- **render()**: Visualize the current state with problem-specific representations
- **close()**: Clean up resources

## State Representation

The environment supports two state representation formats:

### 1. Graph-based (PyTorch Geometric)

The graph representation encodes the Branch-and-Bound tree as a PyTorch Geometric `Data` object:

- **Nodes**: Each node in the B&B tree becomes a graph node with features:
  - Depth in the tree
  - LP relaxation value
  - Integer feasibility status
  - Pruning status
  - Number of fractional variables
  - Optimality gap
  - Queue status

- **Edges**: Represent parent-child relationships in the B&B tree

- **Global Attributes**: Information about the overall search process:
  - Best found integer solution value
  - Global bounds
  - Number of nodes explored
  - Queue size
  - Elapsed time
  - Root relaxation value

### 2. Vector-based

The vector representation provides fixed-sized feature vectors for agents that don't use GNNs:

- **Tree Features**: Statistics about the B&B tree (depth distribution, node types, etc.)
- **Queue Features**: Information about the current priority queue contents
- **Global Features**: Overall search progress and bounds

## Action Space

The environment uses a continuous action space where the agent can influence the priority queue ordering:

- **Direct Approach**: The agent provides an offset value to add to the priority of each node, biasing the search toward specific strategies
- **Future Extensions**: More sophisticated action spaces can be implemented for fine-grained control

## Reward Functions

The environment supports multiple reward functions to train agents with different objectives:

### 1. Time-based

- Fixed negative reward per step (-1.0)
- Encourages finding solutions quickly
- Simple but effective for basic training

### 2. Improvement-based

- Large positive reward for finding better solutions
- Small positive reward for any feasible solution
- Small negative reward for wasted work (pruned nodes)
- Terminal rewards based on solution quality

## Enhanced Visualizations

Each problem type has a specialized visualization that shows:

### Knapsack Problem

- Backpack representation of selected items
- Value vs. weight scatter plot
- Capacity utilization gauge
- Search statistics

### Traveling Salesman Problem (TSP)

- Map-based visualization with directional arrows
- Color-coded subtours (if present)
- Tour validity information
- Distance measurements
- Search progress

### Bin Packing Problem

- 3D representation of items packed in bins
- Color-coded items with size proportional to dimensions
- Bin utilization statistics
- Cross-sectional views

## Usage Example

```python
import numpy as np
import gymnasium as gym
from src.problems import TSP
from src.environments import BranchAndBoundEnv

# Create a problem generator
def tsp_generator():
    return TSP.generate_random_instance(n_cities=8)

# Create environment
env = BranchAndBoundEnv(
    problem_generator=tsp_generator,
    max_steps=100,
    reward_type='improvement',
    observation_type='graph',
    render_mode='human'
)

# Reset environment
observation, info = env.reset()

# Run simple random agent
for i in range(20):
    action = np.random.uniform(-1.0, 1.0, size=(1,))
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()  # Shows rich visualization
    
    if terminated or truncated:
        break

env.close()
```

## Extending the Environment

To extend the RL environment with new capabilities:

1. **New Prioritizers**: Create custom prioritizers in `src/strategies/priority_queue.py`
2. **Additional Reward Functions**: Add new reward calculation methods in `BranchAndBoundEnv._calculate_reward()`
3. **Enhanced State Features**: Expand the node feature extraction in `Node.get_state_features()`
4. **Custom Actions**: Implement more sophisticated action handling in `BranchAndBoundEnv._apply_action()`

See the example scripts in the `examples/` directory for demonstrations of the RL environment.