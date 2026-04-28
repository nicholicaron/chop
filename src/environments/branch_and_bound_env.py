"""
Reinforcement Learning environment for Branch-and-Bound node selection.

The agent's job is to choose which open node to expand next. Each step the
environment exposes the K most promising open nodes (ranked by LP bound) along
with global search-state features; the agent emits K real-valued scores and
the env processes the argmax-scored node.

Action space:  Box(shape=(K,), dtype=float32)
Obs space:     Box(shape=(K * F_PER_NODE + F_GLOBAL,), dtype=float32)

Reward: -1 per node processed (encourage solving with fewer expansions).
Episode ends when the queue empties (problem proven optimal) or the step
budget is exhausted.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.core.solver import BranchAndBoundSolver
from src.core.node import Node
from src.problems.base import OptimizationProblem
from src.utils.logging import BnBLogger


F_PER_NODE = 7
F_GLOBAL = 6
DEFAULT_K = 16


def _node_features(node: Node, root_value: float, incumbent: float, max_depth_obs: int = 50) -> np.ndarray:
    """Compact feature vector for a single open node."""
    # LP bound normalized vs root relaxation
    if root_value not in (None, 0.0) and np.isfinite(root_value):
        rel_bound = node.value / root_value
    else:
        rel_bound = 0.0

    depth_norm = min(node.depth, max_depth_obs) / max_depth_obs

    n_vars = max(len(node.relaxed_soln), 1) if node.relaxed_soln is not None else 1
    frac_share = (node.num_frac or 0) / n_vars

    # Best fractionality of any var in the node (closest to 0.5 is most "informative")
    if node.relaxed_soln is not None and len(node.relaxed_soln) > 0:
        fracs = np.abs(node.relaxed_soln - np.round(node.relaxed_soln))
        best_frac = float(np.max(np.minimum(fracs, 1.0 - fracs))) if len(fracs) else 0.0
    else:
        best_frac = 0.0

    if np.isfinite(incumbent) and node.value > incumbent:
        gap = (node.value - incumbent) / max(abs(node.value), 1.0)
    else:
        gap = 0.0

    # Whether this node could plausibly improve the incumbent
    can_improve = 1.0 if (not np.isfinite(incumbent) or node.value > incumbent) else 0.0

    # 1.0 marks a real node (used for masking padded slots)
    is_real = 1.0

    return np.array(
        [rel_bound, depth_norm, frac_share, best_frac, gap, can_improve, is_real],
        dtype=np.float32,
    )


def _padding_features() -> np.ndarray:
    """Zero vector for unused candidate slots; last entry (is_real) = 0."""
    return np.zeros(F_PER_NODE, dtype=np.float32)


class BranchAndBoundEnv(gym.Env):
    """Gymnasium env where the agent picks which open B&B node to expand next."""

    metadata = {"render_modes": ["human", "tree", "rgb_array"]}

    def __init__(
        self,
        problem_generator: Callable[[], OptimizationProblem],
        k_nodes: int = DEFAULT_K,
        max_steps: int = 500,
        time_limit: float = 60.0,
        reward_type: str = "nodes",
        render_mode: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.problem_generator = problem_generator
        self.k_nodes = k_nodes
        self.max_steps = max_steps
        self.time_limit = time_limit
        self.reward_type = reward_type
        self.render_mode = render_mode
        self.verbose = verbose

        self.action_space = spaces.Box(
            low=-1e6, high=1e6, shape=(k_nodes,), dtype=np.float32
        )
        obs_dim = k_nodes * F_PER_NODE + F_GLOBAL
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # State (set in reset)
        self.problem: Optional[OptimizationProblem] = None
        self.solver: Optional[BranchAndBoundSolver] = None
        self.problem_c: Optional[np.ndarray] = None
        self.open_nodes: List[Node] = []
        self.steps_taken = 0
        self.start_time: Optional[float] = None
        self.history: Dict[str, list] = {}

    # ----- Gymnasium API -----

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.problem = self.problem_generator()
        c, A_eq, b_eq, A_ub, b_ub = self.problem.to_ilp()
        self.problem_c = c

        # Use a logger that doesn't dump per-step to stderr
        logger = BnBLogger(verbose=False) if self.verbose is False else BnBLogger()
        self.solver = BranchAndBoundSolver(logger=logger, max_nodes=10**9, use_cuts=False)
        self.solver.reset()

        root = self.solver._initialize_root_node(c, A_ub, b_ub, A_eq, b_eq)
        if root is None:
            raise ValueError("Root LP infeasible")

        # We bypass the solver's heap and manage open nodes ourselves so the
        # agent's action actually controls the search order.
        self.solver.priority_queue.clear()
        self.open_nodes = [root]

        self.steps_taken = 0
        self.start_time = time.time()
        self.history = {"rewards": [], "obj_values": [], "queue_sizes": [], "actions": []}

        return self._observation(), self._info()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).flatten()
        if action.shape[0] < self.k_nodes:
            action = np.pad(action, (0, self.k_nodes - action.shape[0]))

        self.steps_taken += 1

        terminated, truncated = False, False
        result_status = "branched"

        if not self.open_nodes:
            terminated = True
            result_status = "completed"
        else:
            # Pick the top-K nodes by LP bound, then let the agent pick among them
            self.open_nodes.sort(key=lambda n: -n.value)
            candidates = self.open_nodes[: self.k_nodes]
            n_real = len(candidates)

            scores = action[:n_real]
            chosen_idx = int(np.argmax(scores))
            chosen = candidates[chosen_idx]

            # Remove the chosen node from open_nodes
            self.open_nodes.remove(chosen)
            self.solver.processed_nodes.add(chosen.id)

            result_status = self._process_node(chosen)

        # Time / step budget check
        if self.steps_taken >= self.max_steps:
            truncated = True
        if (time.time() - self.start_time) >= self.time_limit:
            truncated = True

        reward = self._reward(result_status, terminated)
        obs = self._observation()
        info = self._info(extra={"status": result_status})

        self.history["rewards"].append(reward)
        self.history["obj_values"].append(self.solver.optimal_obj_value)
        self.history["queue_sizes"].append(len(self.open_nodes))
        self.history["actions"].append(int(np.argmax(action[: max(1, len(self.open_nodes))])))

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if self.render_mode == "tree":
            self.solver._visualize_tree(self.problem.name)
            return f"plots/{self.problem.name.replace(' ', '_').lower()}_tree.png"
        return None

    def close(self):
        if self.solver and self.solver.logger:
            self.solver.logger.finish()

    # ----- Internals -----

    def _process_node(self, node: Node) -> str:
        """Mirror of the solver's per-node logic, but children land in open_nodes."""
        # Bound prune
        if node.value <= self.solver.global_lower_bound:
            return "pruned_bound"

        # Integer feasibility check
        is_integer = all(
            abs(x - round(x)) < self.solver.tolerance for x in node.relaxed_soln
        )
        if is_integer:
            return self._integer_solution(node)

        # Identify fractional vars and branch
        node.indices_frac = [
            i for i, x in enumerate(node.relaxed_soln)
            if abs(x - round(x)) > self.solver.tolerance
        ]
        node.num_frac = len(node.indices_frac)
        node.num_int = len(self.problem_c) - node.num_frac

        # Use the solver's _branch but harvest children from its queue
        self.solver.priority_queue.clear()
        self.solver._branch(node, self.problem_c)
        new_children = list(self.solver.priority_queue.items())
        self.solver.priority_queue.clear()
        self.open_nodes.extend(new_children)
        return "branched"

    def _integer_solution(self, node: Node) -> str:
        # Skip TSP subtour handling here (out of scope for this env)
        if self.problem.get_constraint_generator() is not None:
            return "integer_skip_tsp"
        if node.value > self.solver.global_lower_bound:
            self.solver.global_lower_bound = node.value
            self.solver.optimal_obj_value = node.value
            self.solver.optimal_solution = node.relaxed_soln
            self.solver.optimal_node = node
            return "new_best"
        return "integer_feasible"

    def _observation(self) -> np.ndarray:
        # Sort once for consistent ordering across step() and _observation()
        self.open_nodes.sort(key=lambda n: -n.value)
        candidates = self.open_nodes[: self.k_nodes]

        root_v = self.solver.root_relaxation_value or 0.0
        incumbent = self.solver.global_lower_bound

        node_blocks = []
        for n in candidates:
            node_blocks.append(_node_features(n, root_v, incumbent))
        for _ in range(self.k_nodes - len(candidates)):
            node_blocks.append(_padding_features())
        node_part = np.concatenate(node_blocks) if node_blocks else np.zeros(0, dtype=np.float32)

        # Global features
        step_frac = self.steps_taken / max(self.max_steps, 1)
        queue_size_norm = min(len(self.open_nodes), 1000) / 1000.0
        if np.isfinite(incumbent) and root_v != 0:
            gap_norm = (root_v - incumbent) / max(abs(root_v), 1.0)
        else:
            gap_norm = 1.0
        have_incumbent = 1.0 if np.isfinite(incumbent) else 0.0
        nodes_created = self.solver.node_counter / 1000.0
        elapsed_frac = (time.time() - self.start_time) / max(self.time_limit, 1e-3) if self.start_time else 0.0

        global_part = np.array(
            [step_frac, queue_size_norm, gap_norm, have_incumbent, nodes_created, elapsed_frac],
            dtype=np.float32,
        )

        return np.concatenate([node_part, global_part])

    def _reward(self, status: str, terminated: bool) -> float:
        if self.reward_type == "nodes":
            # Per-node penalty; small bonus on a new incumbent; final bonus when proven optimal
            r = -1.0
            if status == "new_best":
                r += 5.0
            if terminated:
                r += 50.0
            return r
        if self.reward_type == "dense":
            # Mix per-node penalty with normalized incumbent improvement
            r = -1.0
            if status == "new_best" and self.solver.root_relaxation_value:
                gap_close = max(0.0, self.solver.optimal_obj_value) / abs(self.solver.root_relaxation_value)
                r += 10.0 * gap_close
            if terminated:
                r += 20.0
            return r
        return 0.0

    def _info(self, extra: Optional[dict] = None) -> dict:
        info = {
            "problem_name": self.problem.name if self.problem else None,
            "current_best_obj": self.solver.optimal_obj_value if self.solver else None,
            "nodes_explored": self.steps_taken,
            "queue_size": len(self.open_nodes),
            "time_elapsed": (time.time() - self.start_time) if self.start_time else 0.0,
        }
        if extra:
            info.update(extra)
        return info


# ----- Baseline non-learning agents that fit the new action contract -----


class HeuristicAgent:
    """Wrapper that emits action vectors implementing classical node-selection rules."""

    def __init__(self, mode: str = "best_bound", k: int = DEFAULT_K):
        if mode not in {"best_bound", "depth_first", "breadth_first", "random"}:
            raise ValueError(mode)
        self.mode = mode
        self.k = k
        self.rng = np.random.default_rng()

    def reset(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray) -> np.ndarray:
        # Candidates are ordered by LP bound desc in the env, so:
        #   best_bound: prefer index 0 (the highest-bound candidate)
        #   depth_first: prefer the deepest among candidates (depth_norm at idx 1)
        #   breadth_first: prefer the shallowest
        #   random: uniform over real (non-padded) slots
        node_part = observation[: self.k * F_PER_NODE].reshape(self.k, F_PER_NODE)
        is_real = node_part[:, -1]

        if self.mode == "best_bound":
            scores = node_part[:, 0]  # rel_bound
        elif self.mode == "depth_first":
            scores = node_part[:, 1]  # depth_norm
        elif self.mode == "breadth_first":
            scores = -node_part[:, 1]
        else:  # random
            scores = self.rng.standard_normal(self.k)

        # Mask out padding slots so argmax never lands on a fake node
        scores = np.where(is_real > 0, scores, -1e9)
        return scores.astype(np.float32)
