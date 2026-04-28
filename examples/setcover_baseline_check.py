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

from src.environments.branch_and_bound_env import BranchAndBoundEnv, HeuristicAgent
from src.problems.set_cover import SetCover


K = 16


def make_env_factory(n_elements: int, n_sets: int, density: float, max_steps: int):
    def factory(seed: int) -> BranchAndBoundEnv:
        rng = np.random.default_rng(seed)

        def gen():
            return SetCover.generate_random_instance(
                n_elements=n_elements,
                n_sets=n_sets,
                density=density,
                seed=int(rng.integers(0, 10**6)),
                difficulty="medium",
            )

        return BranchAndBoundEnv(
            problem_generator=gen,
            k_nodes=K,
            max_steps=max_steps,
            time_limit=30.0,
            reward_type="nodes",
        )

    return factory


def eval_heuristic(mode, factory, n_eval=10):
    nodes, completed = [], []
    for ep in range(n_eval):
        env = factory(2000 + ep)
        obs, _ = env.reset(seed=2000 + ep)
        agent = HeuristicAgent(mode=mode, k=K)
        agent.reset(seed=2000 + ep)
        done, truncated = False, False
        while not (done or truncated):
            obs, _, done, truncated, info = env.step(agent.act(obs))
        nodes.append(info["nodes_explored"])
        completed.append(done)
    return np.mean(nodes), np.std(nodes), np.mean(completed)


def main():
    print("Set Cover baselines (n_eval=10 per cell)\n")
    print(f"{'config':<28} {'best_bound':<18} {'depth_first':<18} {'breadth_first':<18} {'random':<18}")
    print("-" * 100)

    configs = [
        (15, 20, 0.3, 400),
        (20, 25, 0.25, 500),
        (25, 30, 0.2, 600),
    ]

    for n_e, n_s, density, max_steps in configs:
        factory = make_env_factory(n_e, n_s, density, max_steps)
        results = {}
        for mode in ("best_bound", "depth_first", "breadth_first", "random"):
            m, s, solved = eval_heuristic(mode, factory, n_eval=8)
            results[mode] = (m, s, solved)

        cfg_str = f"{n_e}e x {n_s}s d={density}"
        line = f"{cfg_str:<28}"
        for mode in ("best_bound", "depth_first", "breadth_first", "random"):
            m, s, solved = results[mode]
            line += f" {m:>6.1f}±{s:<5.1f} ({100*solved:>3.0f}%) "
        print(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
