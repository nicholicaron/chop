"""
Two-stage training: imitate best_bound first, then fine-tune with REINFORCE.

The hypothesis: by initializing the policy to mimic best_bound (the strongest
classical heuristic in many regimes), REINFORCE doesn't waste budget
rediscovering it -- every gradient step can be spent improving on top of a
solid prior. Most useful on Set Cover where best_bound is a reasonable but
beatable starting point.

Usage:
    python examples/train_imitation_then_rl.py --policy mlp --imitation_episodes 100 --rl_episodes 400
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

from src.agents.imitation import ImitationConfig, ImitationLearner
from src.agents.policy import NodeSelectionPolicy
from src.agents.reinforce import ReinforceTrainer, TrainConfig
from src.agents.transformer_policy import TransformerNodeSelectionPolicy
from src.environments.branch_and_bound_env import DEFAULT_K
from src.problems.set_cover import SetCover
from src.utils.eval import evaluate_heuristic, evaluate_policy, make_env_factory


K = DEFAULT_K


def setcover_factory(n_elements, n_sets, density, max_steps, time_limit=30.0):
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
    if name == "transformer":
        return TransformerNodeSelectionPolicy(k=K, hidden=64, n_layers=2, n_heads=4)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="mlp", choices=["mlp", "transformer"])
    parser.add_argument("--n_elements", type=int, default=50)
    parser.add_argument("--n_sets", type=int, default=80)
    parser.add_argument("--density", type=float, default=0.10)
    parser.add_argument("--imitation_episodes", type=int, default=100,
                        help="best_bound rollouts to record for imitation")
    parser.add_argument("--imitation_epochs", type=int, default=15,
                        help="supervised passes through the imitation buffer")
    parser.add_argument("--imitation_lr", type=float, default=1e-3)
    parser.add_argument("--rl_episodes", type=int, default=400,
                        help="REINFORCE episodes after warm-start (0 to skip)")
    parser.add_argument("--rl_lr", type=float, default=2e-4,
                        help="lower default than from-scratch since policy is already shaped")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--time_limit", type=float, default=45.0)
    parser.add_argument("--n_eval", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default="checkpoints/imitation_setcover_{policy}.pt")
    args = parser.parse_args()

    save_path = args.save.format(policy=args.policy)
    cfg_str = f"SetCover({args.n_elements}e x {args.n_sets}s d={args.density})"
    print(f"\n=== Imitation -> RL on {cfg_str} (policy={args.policy}) ===\n")

    factory = setcover_factory(args.n_elements, args.n_sets, args.density,
                               args.max_steps, args.time_limit)
    policy = build_policy(args.policy)

    # Stage 1: imitation
    learner = ImitationLearner(
        policy=policy,
        env_factory=factory,
        config=ImitationConfig(
            n_rollout_episodes=args.imitation_episodes,
            epochs=args.imitation_epochs,
            lr=args.imitation_lr,
            seed=args.seed,
            log_every=max(1, args.imitation_epochs // 10),
        ),
        expert_mode="best_bound",
    )
    t0 = time.time()
    learner.train()
    print(f"Imitation took {time.time()-t0:.1f}s")
    print("\nPost-imitation eval (deterministic):")
    post_imitation = evaluate_policy(policy, factory, n_eval=args.n_eval, deterministic=True)
    print(f"  nodes={post_imitation['nodes_mean']:>6.1f} ± {post_imitation['nodes_std']:<5.1f}  "
          f"solved={100*post_imitation['solved_frac']:>5.1f}%")

    # Stage 2: RL fine-tune (optional)
    rl_stats = None
    if args.rl_episodes > 0:
        print(f"\n=== Stage 2: REINFORCE fine-tune for {args.rl_episodes} episodes ===")
        trainer = ReinforceTrainer(
            policy=policy,
            env_factory=factory,
            config=TrainConfig(
                n_episodes=args.rl_episodes,
                lr=args.rl_lr,
                entropy_coef=0.005,  # less exploration once warm-started
                seed=args.seed + 1000,
                log_every=max(1, args.rl_episodes // 20),
            ),
        )
        rl_stats = trainer.train()
        print("\nPost-RL-finetune eval (deterministic):")
        post_rl = evaluate_policy(policy, factory, n_eval=args.n_eval, deterministic=True)
        print(f"  nodes={post_rl['nodes_mean']:>6.1f} ± {post_rl['nodes_std']:<5.1f}  "
              f"solved={100*post_rl['solved_frac']:>5.1f}%")
    else:
        post_rl = post_imitation

    # Compare to baselines
    print("\n=== Baselines for context ===")
    baselines = {}
    for mode in ("best_bound", "depth_first", "breadth_first", "random"):
        h = evaluate_heuristic(mode, factory, n_eval=args.n_eval, k_nodes=K)
        baselines[mode] = h
        print(f"{mode:<13} nodes={h['nodes_mean']:>6.1f} ± {h['nodes_std']:<5.1f}  "
              f"solved={100*h['solved_frac']:>5.1f}%")

    # Save policy
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "policy_state": policy.state_dict(),
        "config": vars(args),
        "post_imitation_eval": post_imitation,
        "post_rl_eval": post_rl,
    }, save_path)
    print(f"\nSaved policy to {save_path}")

    # Plot: bar chart with imitation vs imitation+RL vs heuristics
    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = ["learned (im+rl)" if args.rl_episodes else "learned (im)", "imitation only"]
    means = [post_rl["nodes_mean"], post_imitation["nodes_mean"]]
    stds = [post_rl["nodes_std"], post_imitation["nodes_std"]]
    colors = ["tab:purple", "tab:cyan"]
    if args.rl_episodes == 0:
        order, means, stds, colors = order[1:], means[1:], stds[1:], colors[1:]
    for mode in ("best_bound", "depth_first", "breadth_first", "random"):
        order.append(mode)
        means.append(baselines[mode]["nodes_mean"])
        stds.append(baselines[mode]["nodes_std"])
        colors.append({"best_bound": "tab:green", "depth_first": "tab:orange",
                       "breadth_first": "tab:red", "random": "tab:gray"}[mode])
    ax.bar(order, means, yerr=stds, color=colors, capsize=4, alpha=0.85)
    ax.set_ylabel("Nodes explored to optimum")
    ax.set_title(f"Imitation -> RL on {cfg_str} ({args.policy}, n={args.n_eval})")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    out = f"plots/imitation_then_rl_{args.policy}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Saved comparison plot to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
