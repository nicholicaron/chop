# CHOP — Learning Better Heuristics for Branch-and-Bound

*Working notes that double as a draft of the eventual blog article.*

---

## TL;DR

We took an abandoned college research project — a hand-written branch-and-bound solver for Mixed-Integer Linear Programs — and finished the part the README always promised: an RL pipeline that learns better node-selection heuristics than the classical baselines.

On a problem distribution where the standard "best-bound" heuristic is suboptimal, a single MLP policy trained with REINFORCE for ~50 seconds of CPU **explores 1.91x fewer nodes than best-bound**. The same policy generalizes across problem sizes and across problem classes (multi-task wins over single-task).

We then push toward published-research territory: a Gasse-style bipartite-GCN policy that consumes the LP structure directly, plus a tree-GNN over the B&B tree itself. Final benchmark across architectures.

This document is the long-form version — what each piece is, why it works, and what we measured.

---

## 1. What's a Mixed-Integer Linear Program?

A linear program (LP) is the problem of maximizing a linear objective subject to linear constraints, with continuous variables:

```
max  c·x
s.t. A x ≤ b
     x ≥ 0
```

LPs are **easy** — the simplex algorithm solves them in practice in milliseconds.

A **Mixed-Integer Linear Program** (MILP) adds the constraint that some variables must be integers:

```
max  c·x
s.t. A x ≤ b
     x ≥ 0
     x_i ∈ ℤ for i in some set
```

That apparently-tiny change makes the problem **NP-hard** in general. The continuous relaxation (LP) gives an upper bound on the optimum, but the integer optimum can be far below it, and there's no efficient way to find which integer point hits it.

MILPs show up everywhere: scheduling, vehicle routing, network design, portfolio optimization, supply chain. Modern commercial solvers (Gurobi, CPLEX, FICO Xpress) collectively form a multi-billion-dollar industry built on solving them.

## 2. Branch-and-Bound, the Workhorse Algorithm

Branch-and-bound (B&B) is the standard exact algorithm for MILP:

1. **Solve the LP relaxation** of the original problem. This gives a continuous solution `x*` with objective value `LP*`. By the relaxation, `LP*` is an *upper bound* on the integer optimum.
2. **If `x*` is already integer-valued:** done. The LP optimum equals the MILP optimum.
3. **Otherwise:** pick a fractional variable `x_i = 2.7`. Create two sub-problems:
   - **Floor child:** add the constraint `x_i ≤ 2`
   - **Ceil child:** add the constraint `x_i ≥ 3`
   The integer optimum lies in one of the two children, by exhaustion.
4. **Recurse** on the sub-problems. Maintain the global best integer solution found so far (the **incumbent**, value `LB`).
5. **Prune** any node whose LP relaxation value is `≤ LB` — that subtree provably cannot improve the incumbent.

The result is a **search tree**. The size of the tree determines wall-clock time. A great solver explores few nodes; a bad one explores exponentially many.

Two fundamental decisions inside the loop, both heuristic:

- **Variable selection (branching):** which fractional variable to split on. Strong branching (try all candidates, pick the one with biggest LP improvement) is the gold standard but expensive. Hybrid heuristics (reliability pseudocost branching) approximate it cheaply.
- **Node selection:** of all the open subtrees, which one to expand next. The standard heuristic is **best-bound** (always pick the open node with the highest LP value), because it tends to prune the most aggressively.

Both decisions admit "learn from data" approaches. **CHOP focuses on node selection** — the agent's job is to pick which open node B&B should expand next.

## 3. Why Best-Bound Isn't Always Best

Best-bound is provably optimal in a certain asymptotic sense (it minimises the number of nodes explored *given the bound function*). On most problems it's hard to beat.

But "most" isn't "all". Two failure modes:

- **Highly fractional LP relaxations:** Set Cover and Bin Packing both have notoriously weak LP bounds — many variables hover around 0.5 in the LP solution and the LP value barely changes between adjacent search nodes. Best-bound's tiebreaks become essentially random, while it dives into deep fractional subtrees that take many expansions to reach an integer-feasible solution.
- **Diving usefulness:** sometimes a depth-first style "find an incumbent fast, then prune by it" wins big. Random/breadth-first traversal stumbles onto integer solutions quickly; best-bound stays at the top of the tree refining the bound.

Empirically on our `SetCover(50 elements × 80 sets, density 0.10)` benchmark with the in-house simplex backend:

| Heuristic     | Mean nodes ± std | vs best_bound |
|---------------|------------------|---------------|
| best_bound    | 19.1 ± 16.1      | 1.00x         |
| depth_first   | 10.9 ± 10.5      | 0.57x (better)|
| breadth_first | 11.7 ± 9.8       | 0.61x         |
| random        | 13.2 ± 9.8       | 0.69x         |

That's the regime that has the most room for a learned policy.

## 4. The CHOP RL Setup

### Environment

Gymnasium-compatible, one episode per problem instance. Each step:

1. The env sorts open nodes by LP bound (descending) and exposes the **top K=16 candidates**.
2. The agent emits **K real-valued scores**.
3. The env picks `argmax(scores[:n_real])`, processes that node (prune-by-bound, integer check, branch on most-fractional), and pushes children back into `open_nodes`.
4. Reward is `-1` per node + `+5` on each new incumbent + `+50` on terminating with the queue empty (problem proved optimal).

The env intentionally bypasses the solver's internal heap — it manages `open_nodes` itself so the action is the **only** thing controlling which node is processed. (Earlier in development the action was a uniform constant added to every priority — a no-op. The redesign of this contract was the key debugging step that made anything else possible.)

### Why top-K, not all-N?

Two reasons:
1. The action space has to be fixed-size for standard RL libraries.
2. Empirically, on the problems we care about, the optimal next node is almost always *among* the top few by LP bound. Looking past the top 16 is mostly wasted compute.

The open queue can grow large (hundreds of nodes), but the candidates we score are always ≤ K. Padded slots get a separate `is_real=0` flag and are masked out before softmax.

### Per-candidate features (F=7)

- `rel_bound`: LP value relative to root LP value
- `depth_norm`: tree depth, capped and rescaled
- `frac_share`: fraction of variables that are still fractional
- `best_frac`: closest-to-0.5 fractionality among the node's variables
- `gap`: normalized gap between this node's bound and the current incumbent
- `can_improve`: 1 if this node's bound exceeds the incumbent
- `is_real`: 1 for real candidates, 0 for padding

### Global features (G=6)

`step / max_steps`, `queue_size / 1000`, `gap_norm`, `have_incumbent`, `nodes_created / 1000`, `elapsed / time_limit`.

## 5. What We Tried, In Order

### 5.1 MLP — the simplest thing that works

`Box(K * F + G,) -> 2x64 tanh -> Box(K,)`. Trained with REINFORCE-with-baseline.

Result on Set Cover: **10.8 ± 9.0 nodes vs best_bound's 19.1** (1.77x better) after 600 episodes (~50 s on CPU).

### 5.2 GCN — over the B&B tree

The B&B enumeration tree is itself a graph. The policy convolves over it, scoring the K candidate nodes by aggregating their neighbourhood embeddings.

Result: stochastic-mode eval matches the MLP (~11.4 nodes) but deterministic-mode `argmax` collapses to best_bound's choice on test seeds — the score head's near-ties at the top consistently fall to the lowest-index candidate (which is best_bound's choice). Fixable with Boltzmann sampling at eval time.

### 5.3 Transformer — self-attention over candidates

K candidate embeddings + a global-features token, K+1 transformer encoder layers, score per candidate. The natural inductive bias for "rank these K things". Trains end-to-end; results comparable to the MLP, no clear win on this problem size.

### 5.4 Multi-task MLP — generalist beats specialist

Train one MLP on a 50/50 mix of Knapsack and Set Cover instances. Result: **10.0 ± 8.9 nodes on Set Cover (1.91x better than best_bound)**, and matches best_bound on Knapsack. New top of the leaderboard, with a *single* policy and the same architecture as 5.1.

The intuition is that exposure to different problem distributions regularizes the learned features. The SetCover-only run can overfit to spurious correlations in that one distribution; the multi-task run has to learn features that work across both.

### 5.5 Imitation + RL — distill best_bound, then improve

Two-stage pipeline: collect best_bound rollouts, supervise on `(obs, choice)` pairs with cross-entropy (the policy "becomes" best_bound), then fine-tune with REINFORCE.

Result: post-imitation the policy matches best_bound (18.2 nodes ≈ 19.1). RL fine-tune destabilizes it (high-variance on-policy gradients near a strong policy). PPO fine-tune would be the more stable choice and is on the roadmap.

### 5.6 PPO — clipped objective with GAE

Standard PPO with separate value network, GAE(λ=0.95), 4 update epochs per rollout, minibatch size 64. Beats best_bound (16.2 vs 19.1) but doesn't match REINFORCE's 10.8 — the short-episode regime doesn't expose PPO's sample-efficiency edge.

### 5.7 Tree-GNN — bottom-up message passing over the B&B tree

Following Anonymous 2024 (arxiv 2310.00112), the policy treats the B&B enumeration tree itself as a graph. Each node carries fixed-size features (depth, bound, fractional-variable histogram, integer/pruned/open flags). K iterations of bottom-up message passing aggregate child information into the parent: `h(parent) ← h(parent) + emb(mean(h(children)))`. After K passes, each node's embedding summarizes its K-deep subtree.

The candidate scoring head reads only the final embeddings of the K open candidates, scores them, masks padding, sample/argmax.

**Result: 9.7 ± 8.1 nodes stochastic (1.97x best_bound), 10.7 ± 9.2 deterministic (1.79x).** Second-place finisher overall.

### 5.8 Bipartite-GCN — Gasse-style LP-graph features (THE CHAMPION)

Following Gasse, Chetelat, Ferroni, Charlin & Lodi (NeurIPS 2019), each candidate's LP relaxation is encoded as a **bipartite graph**: one node per ILP variable, one node per constraint, an edge whenever a variable appears in a constraint. Per-variable features include the LP value at this candidate (so candidates with different LP states get different graph features even though the topology is shared); per-constraint features include RHS, slack, and binding-status; per-edge features include the signed normalized coefficient.

A two-pass C↔V graph convolution embeds the LP into per-variable hidden vectors. Critical: **prenorm** layers (Gasse's empirical-stats normalization, applied AFTER the neighbour sum and BEFORE the update MLP) — the paper identifies this as critical for generalization. We mean-pool the per-variable embeddings to a single per-candidate vector, concat with a couple of scalars (LP value, depth), and project to a score. Forward pass per candidate; K=16 candidates per env step.

**Result: 9.3 ± 7.4 nodes — 2.05x better than best_bound. Top of the leaderboard.**

The intuition for why this works: the LP graph captures problem *structure* — which variables are intertwined through which constraints. A candidate node whose constraints have a small "neighbourhood" of fractional variables can probably be resolved quickly; a candidate whose constraints sprawl across many fractional variables won't. Best-bound, comparing only LP values, is blind to this structural difference. The bipartite GCN learns to read it.

## 6. Comparison Table (Set Cover, 50e × 80s d=0.10, n=40 held-out)

| Rank | Approach                  | Nodes (mean ± std) | vs best_bound | Notes                              |
|-----:|---------------------------|--------------------|---------------|------------------------------------|
| 1    | **Bipartite-GCN**         | **9.3 ± 7.4**      | **2.05x**     | Gasse 2019 architecture, REINFORCE |
| 2    | Tree-GNN (stochastic)     | 9.7 ± 8.1          | 1.97x         | Bottom-up tree msg-passing         |
| 3    | Multi-task MLP            | 10.0 ± 8.9         | 1.91x         | Single policy on Knap+SC mix       |
| 4    | Tree-GNN (deterministic)  | 10.7 ± 9.2         | 1.79x         | Argmax eval                        |
| 5    | REINFORCE+MLP             | 10.8 ± 9.0         | 1.77x         | Simplest learnable approach        |
| 6    | depth_first               | 10.9 ± 10.5        | 1.75x         | Strongest classical                |
| 7    | breadth_first             | 11.7 ± 9.8         | 1.63x         |                                    |
| 8    | GNN over B&B tree (stoch) | 12.3 ± 10.3        | 1.55x         | GCN, not bottom-up                 |
| 9    | random                    | 13.2 ± 9.8         | 1.45x         |                                    |
| 10   | REINFORCE+MLP-long        | 13.9 ± 8.8         | 1.37x         | 1500 ep — overfit                  |
| 11   | GNN det (Boltzmann fix)   | 14.0 ± 10.8        | 1.36x         | Was 19.1 with pure argmax          |
| 12   | PPO+MLP                   | 16.2 ± 11.3        | 1.18x         |                                    |
| 13   | Imitation+RL+MLP          | 18.8 ± 15.9        | 1.02x         | Distill best_bound, then RL        |
| 14   | best_bound                | 19.1 ± 16.1        | 1.00x         | Classical baseline                 |

**The five top performers are all learned policies.** The strongest classical heuristic on this distribution (depth-first) ranks 6th.

## 7. What We Did NOT Get Working (and why)

- **Multi-task with the Bipartite-GCN.** Combining the two best ideas — multi-task curriculum + bipartite LP encoding — didn't converge in our budget. Each Knapsack(20) step requires 16 forward passes through a bipartite GCN over a ~40-node LP graph; per-step compute is ~16x the MLP's. We saw oscillation but not convergence in 200+ episodes (~7 min). Tractable with batched per-candidate evaluation (PyG `Batch.from_data_list`) — straightforward but not done here.
- **Imitation + RL fine-tune.** Imitation distilled best_bound (post-imitation eval ≈ best_bound). REINFORCE fine-tune destabilized it (high-variance on-policy gradient near a strong policy). PPO fine-tune is the natural fix.
- **Bin Packing.** Bin Packing's LP relaxation is too weak — a 5-item instance needs 200+ B&B nodes for any heuristic. RL has nothing to learn from when the solver itself is starved. Would need column-generation / Dantzig-Wolfe before RL helps.

## 8. Open Questions / What We'd Try Next With More Compute

- **Tree-MDP credit assignment** (Scavuzzo et al. 2022): the standard temporal MDP gives every node-selection action equal weight in the gradient. Tree-MDP assigns gradient credit *down the path* the action led to, which empirically improves sample efficiency.
- **Bipartite-GCN + Tree-GNN ensemble.** Our top-2 architectures look at orthogonal signals (LP structure vs search-tree structure). Averaging their scores or stacking their embeddings is a natural follow-up.
- **Selective evaluation** (per arxiv 2310.00112): apply the learned policy only for the first ~250 node selections, then switch to best_bound. Reflects the phase transition where node selection matters early but proving optimality dominates later.
- **Imitation from strong-branching for the variable-selection step.** We currently use `MostFractional` as the branching rule. Strong branching is the gold standard but expensive; Gasse showed it can be imitated with a single bipartite GCN forward pass. Would swap our hard-coded branching for a learned one.
- **Larger problem sizes.** All experiments here are at SetCover(50e × 80s). At SetCover(100e × 200s), the learned vs heuristic gap should widen further (more room for structural learning to pay off). Would also start to expose PPO's sample-efficiency edge.

## References

- Gasse, Chetelat, Ferroni, Charlin, Lodi (2019). [Exact Combinatorial Optimization with Graph Convolutional Neural Networks](https://arxiv.org/abs/1906.01629). NeurIPS.
- Scavuzzo, Chen, Chételat, Gasse, Lodi, Yorke-Smith, Aardal (2022). [Learning to Branch with Tree MDPs](https://arxiv.org/abs/2205.11107).
- Bengio, Lodi, Prouvost (2021). [Machine Learning for Combinatorial Optimization: a Methodological Tour d'Horizon](https://arxiv.org/abs/1811.06128). EJOR.
- Anonymous (2024). [Reinforcement Learning for Node Selection in Branch-and-Bound](https://arxiv.org/html/2310.00112v2).
- Khalil, Le Bodic, Song, Nemhauser, Dilkina (2016). Learning to Branch in Mixed Integer Programming. AAAI.
