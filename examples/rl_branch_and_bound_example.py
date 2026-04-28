"""
Compare a trained REINFORCE policy against the classical heuristic agents on
random Knapsack instances. Designed as the headline runnable example for the
RL side of the project.

Run after training a policy:
    python examples/train_reinforce.py --episodes 800 --n_items 25 --difficulty medium
    python examples/rl_branch_and_bound_example.py
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.agents.policy import NodeSelectionPolicy
from src.environments.branch_and_bound_env import DEFAULT_K
from src.problems.knapsack import Knapsack
from src.utils.eval import (
    evaluate_heuristic,
    evaluate_policy,
    make_env_factory,
)


K = DEFAULT_K
SEED_OFFSET = 3000


def knapsack_factory(n_items: int, difficulty: str, max_steps: int):
    def make_problem(rng):
        return Knapsack.generate_random_instance(
            n_items=n_items,
            seed=int(rng.integers(0, 10**6)),
            difficulty=difficulty,
        )

    return make_env_factory(
        make_problem, k_nodes=K, max_steps=max_steps, time_limit=30.0
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/reinforce_knapsack_n25.pt")
    parser.add_argument("--n_items", type=int, default=20)
    parser.add_argument("--difficulty", default="medium")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=600)
    args = parser.parse_args()

    factory = knapsack_factory(args.n_items, args.difficulty, args.max_steps)

    # Heuristic results
    print(f"=== Knapsack({args.n_items}, {args.difficulty}), {args.episodes} held-out instances ===\n")
    print(f"{'Agent':<14} {'Nodes (mean ± std)':<22} {'Min':<6} {'Max':<6}")
    print("-" * 52)

    results = {}
    for mode in ("best_bound", "depth_first", "breadth_first", "random"):
        res = evaluate_heuristic(mode, factory, n_eval=args.episodes,
                                 seed_offset=SEED_OFFSET, k_nodes=K)
        results[mode] = res
        print(f"{mode:<14} {res['nodes_mean']:>6.1f} ± {res['nodes_std']:<6.1f}      "
              f"{int(res['nodes_min']):>4d}   {int(res['nodes_max']):>4d}")

    # Trained policy if available
    policy = None
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, weights_only=False)
        policy = NodeSelectionPolicy(k=K, hidden=64)
        policy.load_state_dict(ckpt["policy_state"])
        policy.eval()
        learned = evaluate_policy(policy, factory, n_eval=args.episodes,
                                  deterministic=True, seed_offset=SEED_OFFSET)
        results["learned"] = learned
        print(f"{'learned':<14} {learned['nodes_mean']:>6.1f} ± {learned['nodes_std']:<6.1f}      "
              f"{int(learned['nodes_min']):>4d}   {int(learned['nodes_max']):>4d}")
    else:
        print(f"\n(No trained policy at {args.checkpoint}; "
              f"run train_reinforce.py first to enable the 'learned' agent.)")

    # Bar chart
    os.makedirs("plots", exist_ok=True)
    order = (["learned"] if policy else []) + ["best_bound", "depth_first",
                                                 "breadth_first", "random"]
    means = [results[m]["nodes_mean"] for m in order]
    stds = [results[m]["nodes_std"] for m in order]
    palette = {"learned": "tab:purple", "best_bound": "tab:green",
               "depth_first": "tab:orange", "breadth_first": "tab:red", "random": "tab:gray"}
    colors = [palette[m] for m in order]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(order, means, yerr=stds, color=colors, capsize=4, alpha=0.85)
    ax.set_ylabel("Nodes explored to optimum")
    ax.set_title(f"Held-out Knapsack({args.n_items}, {args.difficulty}), n={args.episodes}")
    ax.grid(True, axis="y", alpha=0.3)
    out = f"plots/agent_comparison_{args.n_items}_{args.difficulty}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"\nSaved bar chart to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
