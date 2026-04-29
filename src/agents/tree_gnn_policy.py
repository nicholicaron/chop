"""
Tree-structured GNN policy following the architecture from
"Reinforcement Learning for Node Selection in Branch-and-Bound"
(arxiv 2310.00112v2, Anonymous 2024).

Architecture:
  1. Embed per-node features to hidden dim with a small MLP (skip + LayerNorm).
  2. K iterations of bottom-up message passing:
        h(n) <- h(n) + emb( mean( h(children of n) ) )
     Each iteration extends a node's receptive field by one level of
     descendants. After K passes, each node's embedding summarizes its
     K-deep subtree.
  3. Score each open candidate by projecting its final embedding to a scalar.

The intuition: in B&B, the value of expanding node X depends not just on X's
own LP bound, but on the structure of X's subtree (where the open frontier
is, what's been pruned, etc.). Message passing over the tree exposes that
structure to the policy.

We adopt the paper's "constant-size features" trick: the per-node feature
vector includes a histogram (10 buckets) of fractional-variable parts, so
the feature dim doesn't grow with problem size.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.policy import action_for_choice
from src.environments.branch_and_bound_env import BranchAndBoundEnv, DEFAULT_K


TREE_F_PER_NODE = 19  # extended from the 9-d graph_observation features


class TreeNodeEmbedding(nn.Module):
    """Initial embedding: feature vector -> hidden dim with skip + norm."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.proj_in(x))
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + F.leaky_relu(layer(h)))
        return h


class TreeBottomUpStep(nn.Module):
    """One iteration of bottom-up message passing.

    For each parent node, the aggregated message is the *mean* of its
    children's current embeddings. The parent's embedding gets updated by
    h <- h + emb(mean_children). Leaves (no children) are unchanged.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.update = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, h: torch.Tensor, edge_index_p_to_c: torch.Tensor) -> torch.Tensor:
        """edge_index_p_to_c: (2, E) row 0 = parent idx, row 1 = child idx."""
        if edge_index_p_to_c.numel() == 0:
            return h
        parent_idx = edge_index_p_to_c[0]
        child_idx = edge_index_p_to_c[1]
        # Mean child embedding per parent
        child_h = h.index_select(0, child_idx)
        # Sum then divide by per-parent count
        sum_per_parent = torch.zeros_like(h).index_add(0, parent_idx, child_h)
        ones = torch.ones(child_idx.shape[0], device=h.device)
        count_per_parent = torch.zeros(h.shape[0], device=h.device).index_add(0, parent_idx, ones)
        denom = count_per_parent.clamp(min=1.0).unsqueeze(-1)
        mean_per_parent = sum_per_parent / denom
        # Only parents (count>0) get an update; leaves left alone
        update_mask = (count_per_parent > 0).unsqueeze(-1).float()
        delta = self.update(mean_per_parent) * update_mask
        return h + delta


class TreeGNNNodeSelectionPolicy(nn.Module):
    """Tree GNN over the B&B enumeration tree, scoring K candidates."""

    def __init__(self, k: int = DEFAULT_K, hidden: int = 64, n_iters: int = 3):
        super().__init__()
        self.k = k
        self.hidden = hidden
        self.n_iters = n_iters

        self.embed = TreeNodeEmbedding(TREE_F_PER_NODE, hidden)
        self.steps = nn.ModuleList([TreeBottomUpStep(hidden) for _ in range(n_iters)])
        self.score_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, 1),
        )

    def _build_features_and_edges(self, env: BranchAndBoundEnv):
        """Pull rich per-node features and parent->child edges from the
        env's NetworkX enumeration tree."""
        tree = env.solver.enumeration_tree
        all_ids = list(tree.nodes())
        id_to_idx = {nid: i for i, nid in enumerate(all_ids)}

        root_v = env.solver.root_relaxation_value or 1.0
        root_v_safe = root_v if root_v != 0 else 1.0
        incumbent = env.solver.global_lower_bound

        open_ids = {n.id for n in env.open_nodes}
        candidate_ids = []  # K candidate ids, in env's sort order
        env.open_nodes.sort(key=lambda n: -n.value)
        for cand in env.open_nodes[: self.k]:
            candidate_ids.append(cand.id)

        # Variable-stats lookup keyed by node id (only for open nodes)
        node_lookup = {n.id: n for n in env.open_nodes}

        features = []
        for nid in all_ids:
            data = tree.nodes[nid]

            depth = data.get("depth", 0)
            raw_bound = data.get("relaxed_obj_value")
            if raw_bound is None or not np.isfinite(raw_bound):
                lb = data.get("local_upper_bound", 0.0)
                raw_bound = lb if np.isfinite(lb) else 0.0
            rel_bound = float(raw_bound) / root_v_safe
            num_int = data.get("num_int", 0)
            num_frac = data.get("num_frac", 0)
            color = data.get("color", "")
            is_open = 1.0 if nid in open_ids else 0.0
            is_pruned = 1.0 if color in ("orange", "red") else 0.0
            is_integer = 1.0 if color in ("green", "lightblue") else 0.0
            is_root = 1.0 if color == "blue" else 0.0
            is_candidate = 1.0 if nid in candidate_ids else 0.0
            # Gap to incumbent
            if np.isfinite(incumbent) and rel_bound != 0:
                gap = (raw_bound - incumbent) / max(abs(raw_bound), 1.0)
            else:
                gap = 0.0
            can_improve = 1.0 if (not np.isfinite(incumbent) or raw_bound > incumbent) else 0.0

            # Histogram of fractional parts (10 buckets) for open nodes only
            hist = np.zeros(10, dtype=np.float32)
            if nid in node_lookup and node_lookup[nid].relaxed_soln is not None:
                frac_parts = np.abs(node_lookup[nid].relaxed_soln - np.round(node_lookup[nid].relaxed_soln))
                frac_parts = np.clip(frac_parts, 0.0, 0.5)  # symmetry: 0.5 is most fractional
                bin_idx = np.minimum((frac_parts * 20).astype(int), 9)
                for b in bin_idx:
                    hist[b] += 1.0
                hist = hist / max(1, len(frac_parts))

            features.append([
                depth / 50.0,
                rel_bound,
                num_int / 50.0,
                num_frac / 50.0,
                is_open,
                is_pruned,
                is_integer,
                is_root,
                is_candidate,
                gap,
                can_improve,
                hist[0], hist[1], hist[2], hist[3], hist[4],
                hist[5], hist[6], hist[7],
            ])

        x = torch.as_tensor(features, dtype=torch.float32)

        # Build parent->child edges
        edges = []
        for u, v in tree.edges():
            edges.append((id_to_idx[u], id_to_idx[v]))
        if edges:
            edge_index_p_to_c = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index_p_to_c = torch.zeros((2, 0), dtype=torch.long)

        candidate_indices = torch.tensor(
            [id_to_idx[cid] for cid in candidate_ids], dtype=torch.long
        )

        return x, edge_index_p_to_c, candidate_indices

    def act(self, env: BranchAndBoundEnv, deterministic: bool = False):
        x, edge_index_p_to_c, candidate_indices = self._build_features_and_edges(env)
        n_real = int(candidate_indices.numel())

        if n_real == 0:
            return action_for_choice(0, self.k), torch.tensor(0.0), torch.tensor(0.0)

        h = self.embed(x)
        for step in self.steps:
            h = step(h, edge_index_p_to_c)

        cand_h = h.index_select(0, candidate_indices)
        scores = self.score_head(cand_h).squeeze(-1)  # (n_real,)

        full_logits = torch.full((self.k,), float("-inf"), dtype=scores.dtype, device=scores.device)
        full_logits[:n_real] = scores

        probs = F.softmax(full_logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        if deterministic:
            choice = int(torch.argmax(probs).item())
            log_prob = torch.log(probs[choice] + 1e-12)
        else:
            sampled = dist.sample()
            choice = int(sampled.item())
            log_prob = dist.log_prob(sampled)
        entropy = dist.entropy()

        return action_for_choice(choice, self.k), log_prob, entropy
