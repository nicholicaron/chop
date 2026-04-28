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
from src.environments.branch_and_bound_env import BranchAndBoundEnv, HeuristicAgent
from src.problems.knapsack import Knapsack


K = 16


def env_factory(n_items: int, max_steps: int, time_limit: float, difficulty: str):
    def factory(seed: int) -> BranchAndBoundEnv:
        rng = np.random.default_rng(seed)

        def gen():
            return Knapsack.generate_random_instance(
                n_items=n_items,
                seed=int(rng.integers(0, 10**6)),
                difficulty=difficulty,
            )

        return BranchAndBoundEnv(
            problem_generator=gen,
            k_nodes=K,
            max_steps=max_steps,
            time_limit=time_limit,
            reward_type="nodes",
        )

    return factory


@torch.no_grad()
def eval_policy(policy: NodeSelectionPolicy, factory, n_eval: int):
    nodes, completed = [], []
    for ep in range(n_eval):
        env = factory(20_000 + ep)
        obs, info = env.reset(seed=20_000 + ep)
        done, truncated = False, False
        while not (done or truncated):
            action, _, _ = policy.sample_action(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
        nodes.append(info["nodes_explored"])
        completed.append(done)
    return float(np.mean(nodes)), float(np.std(nodes)), float(np.mean(completed))


def eval_heuristic(mode: str, factory, n_eval: int):
    nodes, completed = [], []
    for ep in range(n_eval):
        env = factory(20_000 + ep)
        obs, info = env.reset(seed=20_000 + ep)
        agent = HeuristicAgent(mode=mode, k=K)
        agent.reset(seed=20_000 + ep)
        done, truncated = False, False
        while not (done or truncated):
            obs, reward, done, truncated, info = env.step(agent.act(obs))
        nodes.append(info["nodes_explored"])
        completed.append(done)
    return float(np.mean(nodes)), float(np.std(nodes)), float(np.mean(completed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/reinforce_knapsack_n25.pt")
    parser.add_argument("--sizes", type=int, nargs="+", default=[15, 20, 25, 30, 35])
    parser.add_argument("--difficulty", type=str, default="medium")
    parser.add_argument("--n_eval", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--time_limit", type=float, default=60.0)
    args = parser.parse_args()

    print(f"\n=== Generalization eval: policy trained on n=25 medium ===\n")
    print(f"Loading {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, weights_only=False)
    policy = NodeSelectionPolicy(k=K, hidden=64)
    policy.load_state_dict(ckpt["policy_state"])
    policy.eval()

    print(f"Trained on: n_items={ckpt['config']['n_items']} difficulty={ckpt['config']['difficulty']}\n")

    results = {"size": [], "learned": [], "best_bound": [], "depth_first": [],
               "breadth_first": [], "random": []}

    print(f"{'Size':<6} {'Learned':<14} {'BestBound':<14} {'DepthFirst':<14} "
          f"{'BreadthFirst':<14} {'Random':<14}")
    print("-" * 88)

    for n_items in args.sizes:
        factory = env_factory(n_items, args.max_steps, args.time_limit, args.difficulty)

        learned_m, learned_s, learned_done = eval_policy(policy, factory, args.n_eval)
        bb_m, bb_s, _ = eval_heuristic("best_bound", factory, args.n_eval)
        df_m, df_s, _ = eval_heuristic("depth_first", factory, args.n_eval)
        bf_m, bf_s, _ = eval_heuristic("breadth_first", factory, args.n_eval)
        rd_m, rd_s, _ = eval_heuristic("random", factory, args.n_eval)

        results["size"].append(n_items)
        results["learned"].append((learned_m, learned_s))
        results["best_bound"].append((bb_m, bb_s))
        results["depth_first"].append((df_m, df_s))
        results["breadth_first"].append((bf_m, bf_s))
        results["random"].append((rd_m, rd_s))

        print(f"{n_items:<6} {learned_m:>5.1f}±{learned_s:<6.1f}  "
              f"{bb_m:>5.1f}±{bb_s:<6.1f}  {df_m:>5.1f}±{df_s:<6.1f}  "
              f"{bf_m:>5.1f}±{bf_s:<6.1f}  {rd_m:>5.1f}±{rd_s:<6.1f}")

    # Plot
    sizes = np.array(results["size"])
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for label, key, color, marker in [
        ("Learned (REINFORCE)", "learned", "tab:purple", "o"),
        ("BestBound", "best_bound", "tab:green", "s"),
        ("DepthFirst", "depth_first", "tab:orange", "^"),
        ("BreadthFirst", "breadth_first", "tab:red", "v"),
        ("Random", "random", "tab:gray", "d"),
    ]:
        means = np.array([m for m, s in results[key]])
        stds = np.array([s for m, s in results[key]])
        ax.errorbar(sizes, means, yerr=stds, label=label, color=color,
                    marker=marker, capsize=3, linewidth=2, markersize=7)
    ax.set_xlabel("Knapsack size (n_items)")
    ax.set_ylabel("Nodes explored to optimum (mean ± std)")
    ax.set_title(f"Generalization across problem sizes\n(policy trained on n=25 only)")
    ax.set_yscale("log")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)
    out = "plots/generalization_across_sizes.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"\nSaved generalization plot to {out}")

    # Persist as JSON
    json_out = "checkpoints/generalization_results.json"
    with open(json_out, "w") as f:
        json.dump({
            "trained_on": ckpt["config"],
            "sizes": list(map(int, sizes)),
            "results": {k: results[k] for k in ("learned", "best_bound", "depth_first",
                                                  "breadth_first", "random")},
        }, f, indent=2)
    print(f"Saved JSON to {json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
