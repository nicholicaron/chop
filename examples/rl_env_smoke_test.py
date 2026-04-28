"""
Smoke-test the rebuilt RL env: do different heuristic policies actually produce
different node counts? If yes, the action causally affects the search and the
env is sane to train on.
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.environments.branch_and_bound_env import BranchAndBoundEnv, HeuristicAgent
from src.problems.knapsack import Knapsack


N_EPISODES = 8
N_ITEMS = 20
MAX_STEPS = 400
K = 16


def make_env(seed: int) -> BranchAndBoundEnv:
    rng = np.random.default_rng(seed)

    def gen():
        return Knapsack.generate_random_instance(
            n_items=N_ITEMS,
            seed=int(rng.integers(0, 10**6)),
            difficulty="easy",
        )

    return BranchAndBoundEnv(
        problem_generator=gen,
        k_nodes=K,
        max_steps=MAX_STEPS,
        time_limit=30.0,
        reward_type="nodes",
    )


def run_one(env: BranchAndBoundEnv, agent: HeuristicAgent, seed: int):
    obs, info = env.reset(seed=seed)
    agent.reset(seed=seed)
    done, truncated = False, False
    total_reward = 0.0
    while not (done or truncated):
        action = agent.act(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
    return {
        "nodes": info["nodes_explored"],
        "obj": info["current_best_obj"],
        "queue_left": info["queue_size"],
        "completed": done,
        "reward": total_reward,
        "time": info["time_elapsed"],
    }


def main():
    agents = ["best_bound", "depth_first", "breadth_first", "random"]
    print(f"\nRL env smoke test: Knapsack({N_ITEMS}), K={K}, max_steps={MAX_STEPS}, n={N_EPISODES}")
    print(f"{'Agent':<14} {'Nodes (mean ± std)':<22} {'Obj (mean)':<12} {'Solved %':<10} {'Time (s)':<10}")
    print("-" * 70)

    summary = {}
    for mode in agents:
        ns, objs, solved, times = [], [], 0, []
        for ep in range(N_EPISODES):
            env = make_env(seed=1000 + ep)
            agent = HeuristicAgent(mode=mode, k=K)
            res = run_one(env, agent, seed=1000 + ep)
            ns.append(res["nodes"])
            objs.append(res["obj"])
            times.append(res["time"])
            if res["completed"]:
                solved += 1
        summary[mode] = (np.mean(ns), np.std(ns), np.mean(objs), solved / N_EPISODES, np.mean(times))
        print(
            f"{mode:<14} {np.mean(ns):>6.1f} ± {np.std(ns):<6.1f}    "
            f"{np.mean(objs):>10.2f}  "
            f"{100*solved/N_EPISODES:>7.0f}%   "
            f"{np.mean(times):>6.3f}"
        )

    # Sanity check: at least two agents should produce different node counts.
    means = [v[0] for v in summary.values()]
    if max(means) - min(means) < 1e-9:
        print("\n[FAIL] All agents produced identical node counts -- action has no effect!")
        return 1
    print("\n[OK] Agents produce different node counts -- env action causally affects search.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
