"""
Comprehensive benchmark: load every available trained policy and compare to
the heuristic baselines on a fixed held-out test suite. Produces a single
summary table + grouped bar chart.

Looks for these checkpoints (skips any that don't exist):
  checkpoints/reinforce_setcover_mlp.pt
  checkpoints/reinforce_setcover_mlp_long.pt   (if present)
  checkpoints/reinforce_setcover_gnn.pt
  checkpoints/ppo_setcover_mlp.pt
  checkpoints/imitation_setcover_mlp.pt        (if present)
  checkpoints/multitask_mlp.pt                 (if present)
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.agents.bipartite_attention_policy import BipartiteAttentionPolicy
from src.agents.bipartite_gnn_policy import BipartiteGCNNodeSelectionPolicy
from src.agents.ensemble_policy import EnsemblePolicy
from src.agents.gnn_policy import GNNNodeSelectionPolicy
from src.agents.hybrid_policy import HybridGNNPolicy
from src.agents.policy import NodeSelectionPolicy
from src.agents.transformer_policy import TransformerNodeSelectionPolicy
from src.agents.tree_gnn_policy import TreeGNNNodeSelectionPolicy
from src.environments.branch_and_bound_env import DEFAULT_K
from src.problems.set_cover import SetCover
from src.utils.eval import evaluate_heuristic, evaluate_policy, make_env_factory


K = DEFAULT_K


def setcover_factory(n_e=50, n_s=80, density=0.10, max_steps=2000, time_limit=45.0):
    def make_problem(rng):
        return SetCover.generate_random_instance(
            n_elements=n_e, n_sets=n_s, density=density,
            seed=int(rng.integers(0, 10**6)), difficulty="medium",
        )
    return make_env_factory(make_problem, k_nodes=K, max_steps=max_steps, time_limit=time_limit)


# Each entry: (label, checkpoint_path, policy_constructor, eval_modes)
# eval_modes = ('det',), ('stoch',), or ('det', 'stoch') for both
CHECKPOINTS = [
    ("REINFORCE+MLP",       "checkpoints/reinforce_setcover_mlp.pt",
     lambda: NodeSelectionPolicy(k=K, hidden=64), ("det",)),
    ("REINFORCE+MLP-long",  "checkpoints/reinforce_setcover_mlp_long.pt",
     lambda: NodeSelectionPolicy(k=K, hidden=64), ("det",)),
    ("REINFORCE+GNN (det)", "checkpoints/reinforce_setcover_gnn.pt",
     lambda: GNNNodeSelectionPolicy(k=K, hidden=64, n_conv=2), ("det",)),
    ("REINFORCE+GNN (stoch)", "checkpoints/reinforce_setcover_gnn.pt",
     lambda: GNNNodeSelectionPolicy(k=K, hidden=64, n_conv=2), ("stoch",)),
    ("PPO+MLP",             "checkpoints/ppo_setcover_mlp.pt",
     lambda: NodeSelectionPolicy(k=K, hidden=64), ("det",)),
    ("Imitation+RL+MLP",    "checkpoints/imitation_setcover_mlp.pt",
     lambda: NodeSelectionPolicy(k=K, hidden=64), ("det",)),
    ("Multitask+MLP",       "checkpoints/multitask_mlp.pt",
     lambda: NodeSelectionPolicy(k=K, hidden=64), ("det",)),
    ("Bipartite-GCN",       "checkpoints/reinforce_setcover_bipartite.pt",
     lambda: BipartiteGCNNodeSelectionPolicy(k=K, hidden=64), ("det",)),
    ("Multitask+Bipartite", "checkpoints/multitask_bipartite.pt",
     lambda: BipartiteGCNNodeSelectionPolicy(k=K, hidden=64), ("det",)),
    ("Tree-GNN",            "checkpoints/reinforce_setcover_tree.pt",
     lambda: TreeGNNNodeSelectionPolicy(k=K, hidden=64, n_iters=3), ("det",)),
    ("Tree-GNN (stoch)",    "checkpoints/reinforce_setcover_tree.pt",
     lambda: TreeGNNNodeSelectionPolicy(k=K, hidden=64, n_iters=3), ("stoch",)),
    ("Bipartite-Attn",      "checkpoints/reinforce_setcover_bipartite_attn.pt",
     lambda: BipartiteAttentionPolicy(k=K, hidden=64, n_attn_layers=2, n_heads=4), ("det",)),
    ("Hybrid (BP+Tree)",    "checkpoints/reinforce_setcover_hybrid.pt",
     lambda: HybridGNNPolicy(k=K, hidden=64, n_tree_iters=3), ("det",)),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_eval", type=int, default=50)
    parser.add_argument("--seed_offset", type=int, default=10_000)
    parser.add_argument("--n_e", type=int, default=50)
    parser.add_argument("--n_s", type=int, default=80)
    parser.add_argument("--density", type=float, default=0.10)
    args = parser.parse_args()

    factory = setcover_factory(args.n_e, args.n_s, args.density)
    print(f"\n=== Benchmark on SetCover({args.n_e}e x {args.n_s}s d={args.density}), n_eval={args.n_eval} ===\n")

    rows = []
    # Heuristics
    for mode in ("best_bound", "depth_first", "breadth_first", "random"):
        h = evaluate_heuristic(mode, factory, n_eval=args.n_eval, k_nodes=K, seed_offset=args.seed_offset)
        rows.append((mode, h["nodes_mean"], h["nodes_std"], h["solved_frac"], "heuristic"))
        print(f"  {mode:<22}  nodes={h['nodes_mean']:>5.1f} ± {h['nodes_std']:<5.1f}  solved={100*h['solved_frac']:>5.1f}%")

    # Learned policies
    for label, path, ctor, modes in CHECKPOINTS:
        if not os.path.exists(path):
            print(f"  {label:<22}  (no checkpoint at {path})")
            continue
        try:
            ckpt = torch.load(path, weights_only=False)
            policy = ctor()
            policy.load_state_dict(ckpt["policy_state"])
            policy.eval()
        except Exception as e:
            print(f"  {label:<22}  FAILED to load: {e}")
            continue

        for mode in modes:
            det = (mode == "det")
            res = evaluate_policy(policy, factory, n_eval=args.n_eval,
                                  deterministic=det, seed_offset=args.seed_offset)
            display_name = f"{label}" if len(modes) == 1 else f"{label}-{mode}"
            rows.append((display_name, res["nodes_mean"], res["nodes_std"],
                         res["solved_frac"], "learned"))
            print(f"  {display_name:<22}  nodes={res['nodes_mean']:>5.1f} ± {res['nodes_std']:<5.1f}  "
                  f"solved={100*res['solved_frac']:>5.1f}%")

    # Sort rows by mean nodes ascending so the table prints best-first
    rows.sort(key=lambda r: r[1])

    print("\n=== Ranked results (best -> worst) ===")
    print(f"{'Approach':<24}  {'Nodes (mean ± std)':<22}  Solved")
    print("-" * 60)
    for label, m, s, frac, kind in rows:
        marker = "*" if kind == "learned" else " "
        print(f"{marker} {label:<22}  {m:>5.1f} ± {s:<5.1f}            {100*frac:>5.1f}%")

    # Plot
    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = [r[0] for r in rows]
    means = [r[1] for r in rows]
    stds = [r[2] for r in rows]
    colors = ["tab:purple" if r[4] == "learned" else "tab:gray" for r in rows]
    ax.bar(labels, means, yerr=stds, color=colors, capsize=4, alpha=0.85)
    ax.set_ylabel("Nodes explored to optimum")
    ax.set_title(f"All-approaches benchmark on SetCover({args.n_e}e x {args.n_s}s d={args.density}), n={args.n_eval}")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    out = "plots/benchmark_all.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"\nSaved chart to {out}")

    # JSON dump
    json_out = "checkpoints/benchmark_all.json"
    with open(json_out, "w") as f:
        json.dump([
            {"label": r[0], "nodes_mean": r[1], "nodes_std": r[2],
             "solved_frac": r[3], "kind": r[4]}
            for r in rows
        ], f, indent=2)
    print(f"Saved JSON to {json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
