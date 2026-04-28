"""
Evaluate a trained node-selection policy across a range of Knapsack sizes
and compare to the classical heuristics. Produces a single summary plot.
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
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
SEED_OFFSET = 20_000


def knapsack_factory(n_items: int, difficulty: str, max_steps: int, time_limit: float):
    def make_problem(rng):
        return Knapsack.generate_random_instance(
            n_items=n_items,
            seed=int(rng.integers(0, 10**6)),
            difficulty=difficulty,
        )

    return make_env_factory(
        make_problem,
        k_nodes=K,
        max_steps=max_steps,
        time_limit=time_limit,
        reward_type="nodes",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/reinforce_knapsack_n25.pt")
    parser.add_argument("--sizes", type=int, nargs="+", default=[15, 20, 25, 30, 35])
    parser.add_argument("--difficulty", type=str, default="medium")
    parser.add_argument("--n_eval", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--time_limit", type=float, default=60.0)
    args = parser.parse_args()

    print(f"\n=== Generalization eval: policy {args.checkpoint} ===\n")

    ckpt = torch.load(args.checkpoint, weights_only=False)
    policy = NodeSelectionPolicy(k=K, hidden=64)
    policy.load_state_dict(ckpt["policy_state"])
    policy.eval()
    print(f"Trained on: n_items={ckpt['config']['n_items']} difficulty={ckpt['config']['difficulty']}\n")

    modes = ("best_bound", "depth_first", "breadth_first", "random")
    results = {key: [] for key in ("learned", *modes)}
    sizes = []

    print(f"{'Size':<6} {'Learned':<14} {'BestBound':<14} {'DepthFirst':<14} "
          f"{'BreadthFirst':<14} {'Random':<14}")
    print("-" * 88)

    for n_items in args.sizes:
        factory = knapsack_factory(n_items, args.difficulty, args.max_steps, args.time_limit)

        learned = evaluate_policy(policy, factory, n_eval=args.n_eval, deterministic=True,
                                  seed_offset=SEED_OFFSET)
        per_mode = {mode: evaluate_heuristic(mode, factory, n_eval=args.n_eval,
                                             seed_offset=SEED_OFFSET, k_nodes=K)
                    for mode in modes}

        sizes.append(n_items)
        results["learned"].append((learned["nodes_mean"], learned["nodes_std"]))
        for mode in modes:
            results[mode].append((per_mode[mode]["nodes_mean"], per_mode[mode]["nodes_std"]))

        print(f"{n_items:<6} {learned['nodes_mean']:>5.1f}±{learned['nodes_std']:<6.1f}  "
              + "  ".join(f"{per_mode[m]['nodes_mean']:>5.1f}±{per_mode[m]['nodes_std']:<6.1f}"
                          for m in modes))

    # Plot
    sizes = np.array(sizes)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    series = [
        ("Learned (REINFORCE)", "learned", "tab:purple", "o"),
        ("BestBound", "best_bound", "tab:green", "s"),
        ("DepthFirst", "depth_first", "tab:orange", "^"),
        ("BreadthFirst", "breadth_first", "tab:red", "v"),
        ("Random", "random", "tab:gray", "d"),
    ]
    for label, key, color, marker in series:
        means = np.array([m for m, s in results[key]])
        stds = np.array([s for m, s in results[key]])
        ax.errorbar(sizes, means, yerr=stds, label=label, color=color,
                    marker=marker, capsize=3, linewidth=2, markersize=7)
    ax.set_xlabel("Knapsack size (n_items)")
    ax.set_ylabel("Nodes explored to optimum (mean ± std)")
    ax.set_title("Generalization across problem sizes\n(policy trained on a single size)")
    ax.set_yscale("log")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)
    os.makedirs("plots", exist_ok=True)
    out = "plots/generalization_across_sizes.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"\nSaved generalization plot to {out}")

    # Persist as JSON
    os.makedirs("checkpoints", exist_ok=True)
    json_out = "checkpoints/generalization_results.json"
    with open(json_out, "w") as f:
        json.dump({
            "trained_on": ckpt["config"],
            "sizes": list(map(int, sizes)),
            "results": {k: results[k] for k in ("learned", *modes)},
        }, f, indent=2)
    print(f"Saved JSON to {json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
