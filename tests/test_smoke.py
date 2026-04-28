"""End-to-end smoke tests for the CHOP pipeline.

These run the full stack (simplex -> B&B -> RL env -> policy training -> eval)
on tiny instances. Total runtime ~10-20s. Catches the kind of broken-import /
broken-action regressions that previously made the project unrunnable.
"""

import os
import sys

# Make src/ importable when pytest is invoked from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch


def test_simplex_solves_2d_lp():
    from src.simplex import linprog_simplex

    c = np.array([3.0, 4.0])
    A_ub = np.array([[1.0, 2.0], [3.0, 1.0]])
    b_ub = np.array([10.0, 15.0])
    res = linprog_simplex(c=c, A_ub=A_ub, b_ub=b_ub)

    assert res.success
    assert res.fun == pytest.approx(24.0, abs=1e-6)
    assert res.x[0] == pytest.approx(4.0, abs=1e-6)
    assert res.x[1] == pytest.approx(3.0, abs=1e-6)


def test_branch_and_bound_solves_knapsack():
    from src.core.solver import BranchAndBoundSolver
    from src.problems.knapsack import Knapsack

    instance = Knapsack(
        values=np.array([60.0, 100.0, 120.0]),
        weights=np.array([10.0, 20.0, 30.0]),
        capacity=50.0,
        name="ClassicKnapsack",
    )
    c, A_eq, b_eq, A_ub, b_ub = instance.to_ilp()

    solver = BranchAndBoundSolver(use_cuts=False)
    solution, value, _, _ = solver.solve(
        c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, problem_name="smoke"
    )

    is_valid, obj = instance.validate_solution(solution)
    assert is_valid
    assert obj == pytest.approx(220.0, abs=1e-3)  # take items 1 and 2


def _knapsack_factory(n_items: int, seed: int):
    from src.environments.branch_and_bound_env import BranchAndBoundEnv
    from src.problems.knapsack import Knapsack

    rng = np.random.default_rng(seed)

    def gen():
        return Knapsack.generate_random_instance(
            n_items=n_items, seed=int(rng.integers(0, 10**6)), difficulty="easy"
        )

    return BranchAndBoundEnv(
        problem_generator=gen,
        k_nodes=8,
        max_steps=200,
        time_limit=10.0,
        reward_type="nodes",
    )


def test_rl_env_action_actually_changes_search():
    """Different heuristics MUST produce different node counts -- otherwise
    the action is a no-op and the env is broken (this was the original bug)."""
    from src.environments.branch_and_bound_env import HeuristicAgent

    counts_by_mode = {}
    for mode in ("best_bound", "breadth_first", "random"):
        ns = []
        for ep in range(3):
            env = _knapsack_factory(n_items=10, seed=100 + ep)
            obs, info = env.reset(seed=100 + ep)
            agent = HeuristicAgent(mode=mode, k=8)
            agent.reset(seed=100 + ep)
            done, truncated = False, False
            while not (done or truncated):
                obs, _, done, truncated, info = env.step(agent.act(obs))
            ns.append(info["nodes_explored"])
        counts_by_mode[mode] = float(np.mean(ns))

    # Best-bound should not be the same as breadth-first / random
    bb = counts_by_mode["best_bound"]
    bf = counts_by_mode["breadth_first"]
    rd = counts_by_mode["random"]
    assert bb < bf, f"best_bound ({bb}) should beat breadth_first ({bf})"
    assert bb < rd, f"best_bound ({bb}) should beat random ({rd})"


def test_reinforce_runs_and_decreases_loss_baseline():
    """Tiny training run that just confirms the gradient pipe is wired up."""
    from src.agents.policy import NodeSelectionPolicy
    from src.agents.reinforce import ReinforceTrainer, TrainConfig

    torch.manual_seed(0)
    np.random.seed(0)

    policy = NodeSelectionPolicy(k=8, hidden=16)
    trainer = ReinforceTrainer(
        policy=policy,
        env_factory=lambda seed: _knapsack_factory(n_items=8, seed=seed),
        config=TrainConfig(n_episodes=10, lr=1e-3, log_every=10, seed=0),
    )
    stats = trainer.train()

    assert len(stats.nodes_explored) == 10
    assert all(c for c in stats.completed)
    # Loss should have changed (i.e. gradients flowed)
    assert max(stats.pg_loss) != min(stats.pg_loss)


def test_policy_eval_smoke():
    from src.agents.policy import NodeSelectionPolicy
    from src.agents.reinforce import ReinforceTrainer, TrainConfig

    policy = NodeSelectionPolicy(k=8, hidden=16)
    trainer = ReinforceTrainer(
        policy=policy,
        env_factory=lambda seed: _knapsack_factory(n_items=8, seed=seed),
        config=TrainConfig(n_episodes=5, log_every=10, seed=0),
    )
    trainer.train()
    res = trainer.evaluate(n_eval=3, deterministic=True)
    assert res["nodes_mean"] > 0
    assert res["solved_frac"] == pytest.approx(1.0, abs=1e-6)


def test_gnn_policy_runs_episode():
    """GNN policy must consume the env's graph_observation and survive a full
    episode without NaN'ing the softmax."""
    from src.agents.gnn_policy import GNNNodeSelectionPolicy

    torch.manual_seed(0)
    policy = GNNNodeSelectionPolicy(k=8, hidden=16, n_conv=2)

    env = _knapsack_factory(n_items=10, seed=99)
    env.reset(seed=99)
    done, truncated = False, False
    total_reward = 0.0
    while not (done or truncated):
        action, log_prob, entropy = policy.act(env, deterministic=False)
        assert torch.isfinite(log_prob)
        assert torch.isfinite(entropy)
        _, r, done, truncated, info = env.step(action)
        total_reward += r
    assert info["nodes_explored"] > 0
    assert done, "GNN should solve the easy 10-item knapsack to completion"


def test_gnn_trains_with_reinforce():
    """Same trainer drives both MLP and GNN (uniform act(env) interface)."""
    from src.agents.gnn_policy import GNNNodeSelectionPolicy
    from src.agents.reinforce import ReinforceTrainer, TrainConfig

    torch.manual_seed(0)
    policy = GNNNodeSelectionPolicy(k=8, hidden=16, n_conv=2)
    trainer = ReinforceTrainer(
        policy=policy,
        env_factory=lambda seed: _knapsack_factory(n_items=8, seed=seed),
        config=TrainConfig(n_episodes=8, lr=1e-3, log_every=20, seed=0),
    )
    stats = trainer.train()
    assert len(stats.nodes_explored) == 8
    assert all(stats.completed)
    assert max(stats.pg_loss) != min(stats.pg_loss)
