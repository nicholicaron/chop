"""
Train an RL policy on Set Cover, where best_bound is provably suboptimal.

Set Cover's LP relaxation tends to be highly fractional, so a greedy
best-bound-first traversal produces deep, fractional dives that take many
expansions to reach an integer-feasible solution. Random / breadth-first
traversal stumbles onto integer solutions sooner. This is exactly the
regime where a learned policy could do better than the strongest classical
heuristic.

Usage:
    python examples/train_setcover.py --policy mlp --episodes 600
    python examples/train_setcover.py --policy gnn --episodes 400
"""

import argparse
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.agents.gnn_policy import GNNNodeSelectionPolicy
from src.agents.policy import NodeSelectionPolicy
from src.agents.ppo import PPOConfig, PPOTrainer
from src.agents.reinforce import ReinforceTrainer, TrainConfig
from src.environments.branch_and_bound_env import DEFAULT_K
from src.problems.set_cover import SetCover
from src.utils.eval import evaluate_heuristic, evaluate_policy, make_env_factory


K = DEFAULT_K


def setcover_factory(n_elements: int, n_sets: int, density: float, max_steps: int,
                     time_limit: float = 30.0):
    def make_problem(rng):
        return SetCover.generate_random_instance(
            n_elements=n_elements,
            n_sets=n_sets,
            density=density,
            seed=int(rng.integers(0, 10**6)),
            difficulty="medium",
        )

    return make_env_factory(make_problem, k_nodes=K, max_steps=max_steps, time_limit=time_limit)


def build_policy(name: str):
    if name == "mlp":
        return NodeSelectionPolicy(k=K, hidden=64)
    if name == "gnn":
        return GNNNodeSelectionPolicy(k=K, hidden=64, n_conv=2)
    if name == "bipartite":
        from src.agents.bipartite_gnn_policy import BipartiteGCNNodeSelectionPolicy
        return BipartiteGCNNodeSelectionPolicy(k=K, hidden=64)
    if name == "tree":
        from src.agents.tree_gnn_policy import TreeGNNNodeSelectionPolicy
        return TreeGNNNodeSelectionPolicy(k=K, hidden=64, n_iters=3)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="reinforce", choices=["reinforce", "ppo"])
    parser.add_argument("--policy", default="mlp", choices=["mlp", "gnn", "bipartite", "tree"])
    parser.add_argument("--n_elements", type=int, default=50)
    parser.add_argument("--n_sets", type=int, default=80)
    parser.add_argument("--density", type=float, default=0.10)
    parser.add_argument("--episodes", type=int, default=600,
                        help="REINFORCE only: number of episodes")
    parser.add_argument("--ppo_iters", type=int, default=60,
                        help="PPO only: number of rollout/update iterations")
    parser.add_argument("--ppo_eps_per_iter", type=int, default=10,
                        help="PPO only: episodes per rollout")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--time_limit", type=float, default=30.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--entropy", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_eval", type=int, default=30)
    parser.add_argument("--save", type=str,
                        default="checkpoints/{algo}_setcover_{policy}.pt")
    args = parser.parse_args()

    save_path = args.save.format(algo=args.algo, policy=args.policy)

    cfg_str = f"SetCover({args.n_elements}e x {args.n_sets}s d={args.density})"
    print(f"\n=== Training {args.policy.upper()} via {args.algo.upper()} on {cfg_str} ===\n")

    factory = setcover_factory(
        args.n_elements, args.n_sets, args.density,
        max_steps=args.max_steps, time_limit=args.time_limit,
    )

    policy = build_policy(args.policy)
    if args.algo == "reinforce":
        trainer = ReinforceTrainer(
            policy=policy,
            env_factory=factory,
            config=TrainConfig(
                n_episodes=args.episodes,
                lr=args.lr,
                entropy_coef=args.entropy,
                seed=args.seed,
                log_every=max(1, args.episodes // 20),
            ),
        )
    else:
        if args.policy != "mlp":
            raise SystemExit("PPO trainer is currently MLP-only.")
        trainer = PPOTrainer(
            policy=policy,
            env_factory=factory,
            config=PPOConfig(
                n_iterations=args.ppo_iters,
                episodes_per_iter=args.ppo_eps_per_iter,
                lr_policy=args.lr,
                entropy_coef=args.entropy,
                seed=args.seed,
                log_every=max(1, args.ppo_iters // 20),
            ),
        )

    t0 = time.time()
    stats = trainer.train()
    print(f"\nTraining took {time.time()-t0:.1f}s")

    # Evaluate
    print("\n=== Held-out eval (deterministic policy) ===")
    learned = evaluate_policy(policy, factory, n_eval=args.n_eval, deterministic=True)
    results = {"learned": learned}
    print(f"Learned       nodes={learned['nodes_mean']:>6.1f} ± {learned['nodes_std']:<6.1f}  "
          f"solved={100*learned['solved_frac']:>5.1f}%")

    for mode in ("best_bound", "depth_first", "breadth_first", "random"):
        h = evaluate_heuristic(mode, factory, n_eval=args.n_eval, k_nodes=K)
        results[mode] = h
        print(f"{mode:<13} nodes={h['nodes_mean']:>6.1f} ± {h['nodes_std']:<6.1f}  "
              f"solved={100*h['solved_frac']:>5.1f}%")

    # Save policy
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"policy_state": policy.state_dict(), "config": vars(args)}, save_path)
    print(f"\nSaved policy to {save_path}")

    # Learning curve plot
    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if args.algo == "reinforce":
        nodes = np.array(stats.nodes_explored)
        xs = np.array(stats.episode)
        x_label = "Episode"
    else:  # ppo
        nodes = np.array(stats.nodes_explored_mean)
        xs = np.array(stats.iteration)
        x_label = "Iteration"
    window = max(5, len(nodes) // 20)
    if len(nodes) >= window:
        rolling = np.convolve(nodes, np.ones(window) / window, mode="valid")
        ax.plot(xs, nodes, alpha=0.25, color="tab:blue", label="per-step")
        ax.plot(xs[window - 1:], rolling, color="tab:blue", linewidth=2,
                label=f"rolling mean (w={window})")
    else:
        ax.plot(xs, nodes, color="tab:blue", linewidth=2, label="per-step")
    ax.axhline(results["best_bound"]["nodes_mean"], color="tab:green",
               linestyle="--", label=f"best_bound ({results['best_bound']['nodes_mean']:.1f})")
    ax.axhline(results["random"]["nodes_mean"], color="tab:red",
               linestyle="--", label=f"random ({results['random']['nodes_mean']:.1f})")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Nodes explored to optimum")
    ax.set_title(f"{args.algo.upper()} ({args.policy.upper()}) on {cfg_str}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    out = f"plots/setcover_learning_curve_{args.algo}_{args.policy}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Saved learning curve to {out}")

    # Comparison bar chart
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = ["learned", "best_bound", "depth_first", "breadth_first", "random"]
    means = [results[m]["nodes_mean"] for m in order]
    stds = [results[m]["nodes_std"] for m in order]
    colors = ["tab:purple", "tab:green", "tab:orange", "tab:red", "tab:gray"]
    ax.bar(order, means, yerr=stds, color=colors, capsize=4, alpha=0.85)
    ax.set_ylabel("Nodes explored to optimum (mean ± std)")
    ax.set_title(f"Held-out eval, {cfg_str}, n={args.n_eval}")
    ax.grid(True, axis="y", alpha=0.3)
    out_bar = f"plots/setcover_comparison_{args.algo}_{args.policy}.png"
    plt.tight_layout()
    plt.savefig(out_bar, dpi=160)
    plt.close()
    print(f"Saved comparison bar to {out_bar}")

    # JSON dump (algo-specific stat shape)
    if args.algo == "reinforce":
        stats_dump = {
            "episode": stats.episode,
            "nodes_explored": stats.nodes_explored,
            "return_total": stats.return_total,
            "completed": stats.completed,
            "pg_loss": stats.pg_loss,
        }
    else:
        stats_dump = {
            "iteration": stats.iteration,
            "nodes_explored_mean": stats.nodes_explored_mean,
            "return_mean": stats.return_mean,
            "completed_frac": stats.completed_frac,
            "pg_loss": stats.pg_loss,
            "value_loss": stats.value_loss,
            "entropy": stats.entropy,
        }
    json_out = f"checkpoints/setcover_stats_{args.algo}_{args.policy}.json"
    with open(json_out, "w") as f:
        json.dump({"config": vars(args), "stats": stats_dump, "eval": results}, f, indent=2)
    print(f"Saved stats to {json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
