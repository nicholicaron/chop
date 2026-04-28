"""
Train a node-selection policy on small random Knapsack problems with REINFORCE.
Compares the trained policy against the heuristic baselines and produces plots.
"""

import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.agents.policy import NodeSelectionPolicy
from src.agents.reinforce import ReinforceTrainer, TrainConfig
from src.environments.branch_and_bound_env import BranchAndBoundEnv, HeuristicAgent
from src.problems.knapsack import Knapsack


K = 16
F_PER_NODE = 7  # mirrored from env for shape reference


def make_env_factory(n_items: int, max_steps: int, time_limit: float, difficulty: str):
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


def evaluate_heuristic(mode: str, env_factory, n_eval: int):
    nodes, objs, completed = [], [], []
    for ep in range(n_eval):
        env = env_factory(10_000 + ep)
        obs, info = env.reset(seed=10_000 + ep)
        agent = HeuristicAgent(mode=mode, k=K)
        agent.reset(seed=10_000 + ep)
        done, truncated = False, False
        while not (done or truncated):
            obs, reward, done, truncated, info = env.step(agent.act(obs))
        nodes.append(info["nodes_explored"])
        objs.append(info["current_best_obj"])
        completed.append(done)
    return {
        "nodes_mean": float(np.mean(nodes)),
        "nodes_std": float(np.std(nodes)),
        "obj_mean": float(np.mean(objs)),
        "solved_frac": float(np.mean(completed)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_items", type=int, default=15)
    parser.add_argument("--difficulty", type=str, default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--time_limit", type=float, default=15.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_eval", type=int, default=40)
    parser.add_argument("--save", type=str, default="checkpoints/reinforce_knapsack.pt")
    args = parser.parse_args()

    print(f"\n=== Training REINFORCE policy on Knapsack({args.n_items}, {args.difficulty}) ===\n")

    env_factory = make_env_factory(
        n_items=args.n_items,
        max_steps=args.max_steps,
        time_limit=args.time_limit,
        difficulty=args.difficulty,
    )

    policy = NodeSelectionPolicy(k=K, hidden=64)
    trainer = ReinforceTrainer(
        policy=policy,
        env_factory=env_factory,
        config=TrainConfig(
            n_episodes=args.episodes,
            lr=args.lr,
            entropy_coef=args.entropy,
            seed=args.seed,
            log_every=max(1, args.episodes // 20),
        ),
    )

    t0 = time.time()
    stats = trainer.train()
    print(f"\nTraining took {time.time()-t0:.1f}s")

    # Evaluate trained policy
    print("\n=== Evaluation (held-out instances, deterministic policy) ===")
    learned = trainer.evaluate(n_eval=args.n_eval, deterministic=True)
    print(f"Learned   nodes={learned['nodes_mean']:>6.1f} ± {learned['nodes_std']:<5.1f}  "
          f"obj={learned['obj_mean']:>7.2f}  solved={100*learned['solved_frac']:>5.1f}%")

    # Collect & print heuristic results
    heuristic_results = {"learned": learned}
    for mode in ("best_bound", "depth_first", "breadth_first", "random"):
        h = evaluate_heuristic(mode, env_factory, n_eval=args.n_eval)
        heuristic_results[mode] = h
        print(f"{mode:<13} nodes={h['nodes_mean']:>6.1f} ± {h['nodes_std']:<5.1f}  "
              f"obj={h['obj_mean']:>7.2f}  solved={100*h['solved_frac']:>5.1f}%")

    # Save policy
    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        torch.save({
            "policy_state": policy.state_dict(),
            "config": vars(args),
        }, args.save)
        print(f"\nSaved policy to {args.save}")

    # Plot learning curve
    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    nodes = np.array(stats.nodes_explored)
    eps = np.array(stats.episode)
    window = max(5, len(nodes) // 20)
    rolling = np.convolve(nodes, np.ones(window) / window, mode="valid")
    ax.plot(eps, nodes, alpha=0.25, color="tab:blue", label="per-episode")
    ax.plot(eps[window - 1:], rolling, color="tab:blue", linewidth=2,
            label=f"rolling mean (w={window})")
    ax.axhline(heuristic_results["best_bound"]["nodes_mean"], color="tab:green",
               linestyle="--", label=f"best_bound eval ({heuristic_results['best_bound']['nodes_mean']:.1f})")
    ax.axhline(heuristic_results["random"]["nodes_mean"], color="tab:red",
               linestyle="--", label=f"random eval ({heuristic_results['random']['nodes_mean']:.1f})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Nodes explored to optimum")
    ax.set_title(f"REINFORCE on Knapsack({args.n_items}, {args.difficulty})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plot_path = f"plots/reinforce_learning_curve_n{args.n_items}_{args.difficulty}.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()
    print(f"Saved learning curve to {plot_path}")

    # Plot comparison bar chart
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    order = ["learned", "best_bound", "depth_first", "breadth_first", "random"]
    means = [heuristic_results[m]["nodes_mean"] for m in order]
    stds = [heuristic_results[m]["nodes_std"] for m in order]
    colors = ["tab:purple", "tab:green", "tab:orange", "tab:red", "tab:gray"]
    ax.bar(order, means, yerr=stds, color=colors, capsize=4, alpha=0.85)
    ax.set_ylabel("Nodes explored (mean ± std)")
    ax.set_title(f"Held-out eval, Knapsack({args.n_items}, {args.difficulty}), n={args.n_eval}")
    ax.grid(True, axis="y", alpha=0.3)
    bar_path = f"plots/comparison_n{args.n_items}_{args.difficulty}.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=160)
    plt.close()
    print(f"Saved comparison bar chart to {bar_path}")

    # Save raw stats as JSON for downstream analysis
    stats_path = f"checkpoints/training_stats_n{args.n_items}_{args.difficulty}.json"
    with open(stats_path, "w") as f:
        json.dump({
            "config": vars(args),
            "stats": {
                "episode": stats.episode,
                "nodes_explored": stats.nodes_explored,
                "return_total": stats.return_total,
                "completed": stats.completed,
                "pg_loss": stats.pg_loss,
                "elapsed_s": stats.elapsed_s,
            },
            "eval": heuristic_results,
        }, f, indent=2)
    print(f"Saved stats to {stats_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
