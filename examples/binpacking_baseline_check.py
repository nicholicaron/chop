"""
Heuristic baseline scan for Bin Packing. Bin Packing has a notoriously weak
LP relaxation, so best_bound may struggle here -- worth knowing the spread
across heuristics before training.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.environments.branch_and_bound_env import DEFAULT_K
from src.problems.bin_packing import BinPacking
from src.utils.eval import heuristic_sweep, make_env_factory


K = DEFAULT_K


def binpacking_factory(n_items: int, bin_capacity: float, max_steps: int):
    def make_problem(rng):
        return BinPacking.generate_random_instance(
            n_items=n_items,
            bin_capacity=bin_capacity,
            seed=int(rng.integers(0, 10**6)),
            difficulty="medium",
        )

    return make_env_factory(make_problem, k_nodes=K, max_steps=max_steps, time_limit=30.0)


def main():
    print("Bin Packing baselines (n_eval=10 per cell)\n")
    print(f"{'config':<24} {'best_bound':<22} {'depth_first':<22} "
          f"{'breadth_first':<22} {'random':<22}")
    print("-" * 112)

    configs = [
        # (n_items, bin_capacity, max_steps)
        (5, 10, 200),
        (6, 10, 400),
        (8, 10, 800),
        (10, 12, 1500),
    ]

    for n_items, cap, max_steps in configs:
        factory = binpacking_factory(n_items, cap, max_steps)
        sweep = heuristic_sweep(factory, n_eval=10, seed_offset=4000, k_nodes=K)
        cfg_str = f"{n_items} items, cap={cap}"
        cells = " ".join(
            f"{sweep[m]['nodes_mean']:>5.1f}±{sweep[m]['nodes_std']:<5.1f} ({100*sweep[m]['solved_frac']:>3.0f}%)  "
            for m in ("best_bound", "depth_first", "breadth_first", "random")
        )
        print(f"{cfg_str:<24} {cells}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
