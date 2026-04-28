"""
Shared evaluation utilities for CHOP RL examples.

Centralizes the env-factory boilerplate and heuristic/policy evaluation
loops so example scripts stay focused on the experiment they're running.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch

from src.environments.branch_and_bound_env import (
    DEFAULT_K,
    BranchAndBoundEnv,
    HeuristicAgent,
)
from src.problems.base import OptimizationProblem


EnvFactory = Callable[[int], BranchAndBoundEnv]


def make_env_factory(
    problem_factory: Callable[[np.random.Generator], OptimizationProblem],
    *,
    k_nodes: int = DEFAULT_K,
    max_steps: int = 500,
    time_limit: float = 30.0,
    reward_type: str = "nodes",
) -> EnvFactory:
    """Build an env factory that returns a fresh env for each integer seed.

    `problem_factory` receives a numpy Generator so callers can derive any
    instance-level RNG state they want from a single seed.
    """

    def factory(seed: int) -> BranchAndBoundEnv:
        rng = np.random.default_rng(seed)

        def gen() -> OptimizationProblem:
            return problem_factory(rng)

        return BranchAndBoundEnv(
            problem_generator=gen,
            k_nodes=k_nodes,
            max_steps=max_steps,
            time_limit=time_limit,
            reward_type=reward_type,
        )

    return factory


def _run_episode(env: BranchAndBoundEnv, action_fn: Callable[[np.ndarray], np.ndarray], seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    done, truncated = False, False
    while not (done or truncated):
        obs, _, done, truncated, info = env.step(action_fn(obs))
    return info


def evaluate_heuristic(
    mode: str,
    env_factory: EnvFactory,
    *,
    n_eval: int,
    seed_offset: int = 10_000,
    k_nodes: int = DEFAULT_K,
) -> Dict[str, float]:
    """Run a HeuristicAgent for n_eval episodes and report aggregate metrics."""
    nodes, objs, completed = [], [], []
    for ep in range(n_eval):
        seed = seed_offset + ep
        env = env_factory(seed)
        agent = HeuristicAgent(mode=mode, k=k_nodes)
        agent.reset(seed=seed)
        info = _run_episode(env, agent.act, seed)
        nodes.append(info["nodes_explored"])
        objs.append(info["current_best_obj"])
        completed.append(info["queue_size"] == 0)

    return _summarize(nodes, objs, completed)


def evaluate_policy(
    policy: torch.nn.Module,
    env_factory: EnvFactory,
    *,
    n_eval: int,
    deterministic: bool = True,
    seed_offset: int = 10_000,
) -> Dict[str, float]:
    """Run a learnable policy for n_eval episodes and report aggregate metrics.

    The policy must expose ``act(env, deterministic=...)`` returning a
    ``(action, log_prob, entropy)`` tuple (both NodeSelectionPolicy and
    GNNNodeSelectionPolicy follow this contract).
    """
    nodes, objs, completed = [], [], []
    with torch.no_grad():
        for ep in range(n_eval):
            seed = seed_offset + ep
            env = env_factory(seed)
            env.reset(seed=seed)
            done, truncated = False, False
            while not (done or truncated):
                action, _, _ = policy.act(env, deterministic=deterministic)
                _, _, done, truncated, info = env.step(action)
            nodes.append(info["nodes_explored"])
            objs.append(info["current_best_obj"])
            completed.append(info["queue_size"] == 0)

    return _summarize(nodes, objs, completed)


def _summarize(nodes, objs, completed) -> Dict[str, float]:
    return {
        "nodes_mean": float(np.mean(nodes)),
        "nodes_std": float(np.std(nodes)),
        "nodes_min": float(np.min(nodes)),
        "nodes_max": float(np.max(nodes)),
        "obj_mean": float(np.mean(objs)),
        "solved_frac": float(np.mean(completed)),
        "n_eval": len(nodes),
        "raw_nodes": list(map(int, nodes)),
    }


def heuristic_sweep(
    env_factory: EnvFactory,
    *,
    n_eval: int,
    modes: Sequence[str] = ("best_bound", "depth_first", "breadth_first", "random"),
    seed_offset: int = 10_000,
    k_nodes: int = DEFAULT_K,
) -> Dict[str, Dict[str, float]]:
    """Convenience: evaluate all listed heuristics on the same env factory."""
    return {
        mode: evaluate_heuristic(
            mode, env_factory,
            n_eval=n_eval, seed_offset=seed_offset, k_nodes=k_nodes,
        )
        for mode in modes
    }
