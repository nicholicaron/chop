"""
Smoke-test the rebuilt RL env: do different heuristic policies actually produce
different node counts? If yes, the action causally affects the search and the
env is sane to train on.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.environments.branch_and_bound_env import DEFAULT_K
from src.problems.knapsack import Knapsack
from src.utils.eval import heuristic_sweep, make_env_factory


N_EPISODES = 8
N_ITEMS = 20
MAX_STEPS = 400
K = DEFAULT_K


def knapsack_factory():
    def make_problem(rng):
        return Knapsack.generate_random_instance(
            n_items=N_ITEMS,
            seed=int(rng.integers(0, 10**6)),
            difficulty="easy",
        )

    return make_env_factory(
        make_problem, k_nodes=K, max_steps=MAX_STEPS, time_limit=30.0
    )


def main():
    factory = knapsack_factory()
    print(f"\nRL env smoke test: Knapsack({N_ITEMS}), K={K}, max_steps={MAX_STEPS}, n={N_EPISODES}")
    print(f"{'Agent':<14} {'Nodes (mean ± std)':<22} {'Obj (mean)':<12} {'Solved %':<10}")
    print("-" * 60)

    summary = heuristic_sweep(factory, n_eval=N_EPISODES, seed_offset=1000, k_nodes=K)
    for mode, res in summary.items():
        print(
            f"{mode:<14} {res['nodes_mean']:>6.1f} ± {res['nodes_std']:<6.1f}    "
            f"{res['obj_mean']:>10.2f}  {100*res['solved_frac']:>7.0f}%"
        )

    # Sanity check: at least two agents should produce different node counts.
    means = [res["nodes_mean"] for res in summary.values()]
    if max(means) - min(means) < 1e-9:
        print("\n[FAIL] All agents produced identical node counts -- action has no effect!")
        return 1
    print("\n[OK] Agents produce different node counts -- env action causally affects search.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
