# Reinforcement Learning Environments

A Gymnasium-compatible environment that exposes the Branch-and-Bound search loop to a learnable agent. The agent's job is to choose **which open node B&B should expand next**.

## Components

| Symbol | What it is |
|--------|-----------|
| `BranchAndBoundEnv` | The Gym env. Holds a `BranchAndBoundSolver`, manages the open-node list, and exposes obs / accepts an action / advances one node-expansion per `step()`. |
| `HeuristicAgent` | Non-learning agent that emits the same action contract using classical rules (`best_bound`, `depth_first`, `breadth_first`, `random`). Used as baseline. |
| `DEFAULT_K` | The number of candidate nodes presented to the agent per step (default 16). |
| `F_PER_NODE`, `F_GLOBAL` | Per-node and global feature counts in the flat observation. |

## Action / observation spaces

```
Action:        Box(low=-1e6, high=1e6, shape=(K,), dtype=float32)
Observation:   Box(shape=(K * F_PER_NODE + F_GLOBAL,), dtype=float32)
```

Each step:

1. The env sorts the open-node list by LP bound (descending) and exposes the top K candidates.
2. The agent emits K real-valued scores; argmax over the scores selects the candidate to expand.
3. The env removes that node from `open_nodes`, runs the B&B per-node logic (prune-by-bound, integer check, branch on most-fractional), and pushes any new children back into `open_nodes`.
4. Reward is returned, observation rebuilt, episode terminates when `open_nodes` is empty (problem proven optimal).

The env intentionally bypasses the solver's internal heap — it manages `open_nodes` itself so the agent's action is the **only** thing that controls which node is processed next. (The previous design wired the action through a perturbation on the heap key, which had no effect.)

## Per-candidate features (`F_PER_NODE = 7`)

For each of the K candidates the env emits:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `rel_bound` | LP bound normalized by the root LP value |
| 1 | `depth_norm` | Tree depth, capped and divided by 50 |
| 2 | `frac_share` | Fraction of variables that are still fractional |
| 3 | `best_frac` | Closest-to-0.5 fractionality among the node's variables |
| 4 | `gap` | Normalized gap between this node's bound and the current incumbent |
| 5 | `can_improve` | 1 if `node.value > incumbent` (or no incumbent yet), else 0 |
| 6 | `is_real` | 1 for a real candidate, 0 for a padded slot — used by the policy as a mask |

## Global features (`F_GLOBAL = 6`)

| Index | Feature |
|-------|---------|
| 0 | Step counter normalized by `max_steps` |
| 1 | Open-queue size (capped at 1000) / 1000 |
| 2 | Normalized bound-incumbent gap |
| 3 | `have_incumbent` flag |
| 4 | Total nodes created / 1000 |
| 5 | Wall-clock fraction of `time_limit` consumed |

## Graph observation (for GNN policies)

`env.graph_observation()` returns a `(torch_geometric.data.Data, candidate_indices)` pair:

* `data.x` — `(N, 9)` features for every tree node (depth, rel-bound, frac counts, status flags, an `is_candidate` mask)
* `data.edge_index` — `(2, 2*E)` undirected expansion of the tree edges
* `candidate_indices` — `LongTensor` of size up to K, indices into `data.x` for the open candidates ordered by LP bound

Inf-valued bounds are sanitized before they hit the GNN so convolutions don't NaN.

## Reward functions

`reward_type="nodes"` (default):
```
-1 per node expanded
+5 if a new incumbent was found this step
+50 on terminating with the queue empty (problem proven optimal)
```

`reward_type="dense"`:
```
-1 per node, plus a normalized incumbent-improvement bonus on each new_best,
small terminal bonus
```

## Usage

```python
import numpy as np
from src.environments.branch_and_bound_env import BranchAndBoundEnv, HeuristicAgent
from src.problems.knapsack import Knapsack

def gen():
    return Knapsack.generate_random_instance(n_items=20, seed=42, difficulty="medium")

env = BranchAndBoundEnv(problem_generator=gen, k_nodes=16, max_steps=400)
obs, info = env.reset(seed=42)

agent = HeuristicAgent(mode="best_bound", k=16)
done, truncated = False, False
while not (done or truncated):
    action = agent.act(obs)
    obs, reward, done, truncated, info = env.step(action)

print(info["nodes_explored"], info["current_best_obj"])
```

For learnable policies see [`src/agents/README.md`](../agents/README.md).
