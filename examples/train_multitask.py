"""
Multi-task training: train a single policy on a mix of problem classes,
then evaluate on each one separately. Tests whether the learned features
generalize across problem types.

Mix is currently Knapsack + Set Cover. Each episode samples one class
uniformly at random (or with configurable weights).
"""

import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.agents.policy import NodeSelectionPolicy
from src.agents.reinforce import ReinforceTrainer, TrainConfig
from src.agents.transformer_policy import TransformerNodeSelectionPolicy
from src.environments.branch_and_bound_env import BranchAndBoundEnv, DEFAULT_K
from src.problems.knapsack import Knapsack
from src.problems.set_cover import SetCover
from src.utils.eval import evaluate_heuristic, evaluate_policy, make_env_factory


K = DEFAULT_K


def knapsack_problem(rng, n_items=20, difficulty="medium"):
    return Knapsack.generate_random_instance(
        n_items=n_items, seed=int(rng.integers(0, 10**6)), difficulty=difficulty,
    )


def setcover_problem(rng, n_elements=50, n_sets=80, density=0.10):
    return SetCover.generate_random_instance(
        n_elements=n_elements, n_sets=n_sets, density=density,
        seed=int(rng.integers(0, 10**6)), difficulty="medium",
    )


def mixed_factory(weights, problem_factories, k_nodes, max_steps, time_limit):
    """Each episode samples one of the listed problem factories with given weights."""
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()

    def factory(seed):
        rng = np.random.default_rng(seed)
        idx = int(rng.choice(len(problem_factories), p=weights))
        chosen_factory = problem_factories[idx]

        def gen():
            return chosen_factory(rng)

        return BranchAndBoundEnv(
            problem_generator=gen, k_nodes=k_nodes,
            max_steps=max_steps, time_limit=time_limit, reward_type="nodes",
        )

    return factory


def single_factory(problem_factory, k_nodes, max_steps, time_limit):
    return make_env_factory(
        problem_factory, k_nodes=k_nodes,
        max_steps=max_steps, time_limit=time_limit,
    )


def build_policy(name: str):
    if name == "mlp":
        return NodeSelectionPolicy(k=K, hidden=64)
    if name == "transformer":
        return TransformerNodeSelectionPolicy(k=K, hidden=64, n_layers=2, n_heads=4)
    if name == "bipartite":
        from src.agents.bipartite_gnn_policy import BipartiteGCNNodeSelectionPolicy
        return BipartiteGCNNodeSelectionPolicy(k=K, hidden=64)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="mlp", choices=["mlp", "transformer", "bipartite"])
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--time_limit", type=float, default=45.0)
    parser.add_argument("--n_eval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default="checkpoints/multitask_{policy}.pt")
    args = parser.parse_args()

    save_path = args.save.format(policy=args.policy)
    print(f"\n=== Multi-task training (policy={args.policy}) ===\n")

    # 50/50 mix of Knapsack(20) and SetCover(50e×80s)
    problem_factories = [
        lambda rng: knapsack_problem(rng, n_items=20, difficulty="medium"),
        lambda rng: setcover_problem(rng, n_elements=50, n_sets=80, density=0.10),
    ]
    train_factory = mixed_factory(
        weights=[0.5, 0.5], problem_factories=problem_factories,
        k_nodes=K, max_steps=args.max_steps, time_limit=args.time_limit,
    )

    policy = build_policy(args.policy)
    trainer = ReinforceTrainer(
        policy=policy,
        env_factory=train_factory,
        config=TrainConfig(
            n_episodes=args.episodes, lr=args.lr, entropy_coef=0.015,
            seed=args.seed, log_every=max(1, args.episodes // 20),
        ),
    )
    t0 = time.time()
    trainer.train()
    print(f"\nTraining took {time.time()-t0:.1f}s")

    # Evaluate on each problem class separately
    eval_factories = {
        "Knapsack(20, medium)": single_factory(
            lambda rng: knapsack_problem(rng, n_items=20, difficulty="medium"),
            K, args.max_steps, args.time_limit,
        ),
        "SetCover(50e x 80s d=0.1)": single_factory(
            lambda rng: setcover_problem(rng, n_elements=50, n_sets=80, density=0.10),
            K, args.max_steps, args.time_limit,
        ),
    }

    rows = []
    print("\n=== Held-out eval on each problem class (deterministic) ===")
    for name, fac in eval_factories.items():
        learned = evaluate_policy(policy, fac, n_eval=args.n_eval, deterministic=True)
        bb = evaluate_heuristic("best_bound", fac, n_eval=args.n_eval, k_nodes=K)
        rd = evaluate_heuristic("random", fac, n_eval=args.n_eval, k_nodes=K)
        rows.append((name, learned, bb, rd))
        print(
            f"  {name}: "
            f"learned={learned['nodes_mean']:>5.1f}±{learned['nodes_std']:<5.1f}  "
            f"best_bound={bb['nodes_mean']:>5.1f}±{bb['nodes_std']:<5.1f}  "
            f"random={rd['nodes_mean']:>5.1f}±{rd['nodes_std']:<5.1f}  "
            f"(learned/best_bound={learned['nodes_mean']/max(bb['nodes_mean'],1e-9):.2f}x)"
        )

    # Save policy
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"policy_state": policy.state_dict(), "config": vars(args)}, save_path)
    print(f"\nSaved policy to {save_path}")

    # Plot
    os.makedirs("plots", exist_ok=True)
    fig, axes = plt.subplots(1, len(rows), figsize=(5 * len(rows), 4.5), sharey=False)
    if len(rows) == 1:
        axes = [axes]
    for ax, (name, learned, bb, rd) in zip(axes, rows):
        ax.bar(["learned\n(multitask)", "best_bound", "random"],
               [learned["nodes_mean"], bb["nodes_mean"], rd["nodes_mean"]],
               yerr=[learned["nodes_std"], bb["nodes_std"], rd["nodes_std"]],
               color=["tab:purple", "tab:green", "tab:gray"], capsize=4, alpha=0.85)
        ax.set_title(name)
        ax.set_ylabel("Nodes (mean ± std)")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"Multi-task ({args.policy}) eval, n={args.n_eval} per class")
    out = f"plots/multitask_{args.policy}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Saved comparison plot to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
