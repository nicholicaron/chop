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
from time import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.agents.policy import NodeSelectionPolicy
from src.environments.branch_and_bound_env import BranchAndBoundEnv, HeuristicAgent
from src.problems.knapsack import Knapsack


K = 16


def make_env(seed: int, n_items: int, difficulty: str, max_steps: int) -> BranchAndBoundEnv:
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
        time_limit=30.0,
        reward_type="nodes",
    )


def run_episode(env, action_fn, seed):
    obs, info = env.reset(seed=seed)
    done, truncated = False, False
    while not (done or truncated):
        action = action_fn(obs)
        obs, _, done, truncated, info = env.step(action)
    return info


def compare(checkpoint_path, n_items, difficulty, n_episodes, max_steps):
    # Heuristic agents
    agents = {mode: HeuristicAgent(mode=mode, k=K) for mode in
              ("best_bound", "depth_first", "breadth_first", "random")}

    # Trained policy if available
    policy = None
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, weights_only=False)
        policy = NodeSelectionPolicy(k=K, hidden=64)
        policy.load_state_dict(ckpt["policy_state"])
        policy.eval()
        print(f"Loaded trained policy from {checkpoint_path}\n")
    else:
        print(f"(No trained policy found at {checkpoint_path}; "
              f"run train_reinforce.py first to enable the 'learned' agent.)\n")

    seeds = [3000 + i for i in range(n_episodes)]
    results = {name: [] for name in list(agents.keys()) + (["learned"] if policy else [])}

    for name, agent in agents.items():
        print(f"Evaluating {name}...")
        for s in seeds:
            agent.reset(seed=s)
            env = make_env(s, n_items, difficulty, max_steps)
            info = run_episode(env, agent.act, s)
            results[name].append(info["nodes_explored"])

    if policy is not None:
        print("Evaluating learned (deterministic policy)...")
        for s in seeds:
            env = make_env(s, n_items, difficulty, max_steps)
            with torch.no_grad():
                def act_fn(obs):
                    a, _, _ = policy.sample_action(obs, deterministic=True)
                    return a
                info = run_episode(env, act_fn, s)
            results["learned"].append(info["nodes_explored"])

    print(f"\n=== Knapsack({n_items}, {difficulty}), {n_episodes} held-out instances ===\n")
    print(f"{'Agent':<14} {'Nodes (mean ± std)':<24} {'Min':<7} {'Max':<7}")
    print("-" * 56)
    for name in list(results.keys()):
        ns = np.array(results[name])
        print(f"{name:<14} {ns.mean():>6.1f} ± {ns.std():<6.1f}        {ns.min():>5.0f}  {ns.max():>5.0f}")

    # Bar chart
    os.makedirs("plots", exist_ok=True)
    order = (["learned"] if policy else []) + ["best_bound", "depth_first", "breadth_first", "random"]
    means = [np.mean(results[m]) for m in order]
    stds = [np.std(results[m]) for m in order]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["tab:purple", "tab:green", "tab:orange", "tab:red", "tab:gray"]
    if not policy:
        colors = colors[1:]
    ax.bar(order, means, yerr=stds, color=colors, capsize=4, alpha=0.85)
    ax.set_ylabel("Nodes explored to optimum")
    ax.set_title(f"Held-out Knapsack({n_items}, {difficulty}), n={n_episodes}")
    ax.grid(True, axis="y", alpha=0.3)
    out = f"plots/agent_comparison_{n_items}_{difficulty}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"\nSaved bar chart to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/reinforce_knapsack_n25.pt")
    parser.add_argument("--n_items", type=int, default=20)
    parser.add_argument("--difficulty", default="medium")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=600)
    args = parser.parse_args()

    compare(args.checkpoint, args.n_items, args.difficulty, args.episodes, args.max_steps)


if __name__ == "__main__":
    main()
