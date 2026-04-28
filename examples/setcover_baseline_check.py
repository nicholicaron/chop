"""
Quick baseline check on Set Cover: do the heuristics differ enough to be worth
training a policy on? If best_bound is essentially optimal here too, there's
no headroom; if random/breadth_first lose by a lot AND best_bound itself
explores many nodes, RL has room to improve.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.environments.branch_and_bound_env import DEFAULT_K
from src.problems.set_cover import SetCover
from src.utils.eval import heuristic_sweep, make_env_factory


K = DEFAULT_K


def setcover_factory(n_elements: int, n_sets: int, density: float, max_steps: int):
    def make_problem(rng):
        return SetCover.generate_random_instance(
            n_elements=n_elements,
            n_sets=n_sets,
            density=density,
            seed=int(rng.integers(0, 10**6)),
            difficulty="medium",
        )

    return make_env_factory(make_problem, k_nodes=K, max_steps=max_steps, time_limit=30.0)


def main():
    print("Set Cover baselines (n_eval=20 per cell)\n")
    print(f"{'config':<28} {'best_bound':<22} {'depth_first':<22} "
          f"{'breadth_first':<22} {'random':<22}")
    print("-" * 116)

    configs = [
        (40, 50, 0.15, 1500),
        (50, 60, 0.15, 2000),
        (50, 80, 0.10, 2500),
        (60, 80, 0.12, 3000),
    ]

    for n_e, n_s, density, max_steps in configs:
        factory = setcover_factory(n_e, n_s, density, max_steps)
        sweep = heuristic_sweep(factory, n_eval=20, seed_offset=2000, k_nodes=K)
        cfg_str = f"{n_e}e x {n_s}s d={density}"
        cells = " ".join(
            f"{sweep[m]['nodes_mean']:>5.1f}±{sweep[m]['nodes_std']:<5.1f} ({100*sweep[m]['solved_frac']:>3.0f}%)  "
            for m in ("best_bound", "depth_first", "breadth_first", "random")
        )
        print(f"{cfg_str:<28} {cells}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
