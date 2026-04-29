"""
Hybrid policy: Bipartite-GCN over the LP + Tree-GNN over the search tree,
trained JOINTLY end-to-end with a shared score head.

This is the "smart ensemble" version of what we tried with EnsemblePolicy
(which combined two independently-trained policies and didn't help). The
hypothesis there was that the two architectures should specialize on
complementary signals; here we let them do that explicitly during training.

Per candidate:
  bp_vec = mean_pool(BipartiteGCN(this candidate's LP))      # LP structure
  tr_vec = TreeGNN(B&B tree)[this candidate's tree node]      # search context

Per env step:
  for each of K candidates:
      h = concat([bp_vec, tr_vec, scalar features])
      score = score_head(h)
  softmax over K, mask padding, sample / argmax

Training: same REINFORCE pipeline. The shared score head learns to weigh
LP-structure vs search-context vs raw scalars.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.bipartite_gnn_policy import BipartiteGCNEncoder
from src.agents.policy import action_for_choice
from src.agents.tree_gnn_policy import TREE_F_PER_NODE, TreeBottomUpStep, TreeNodeEmbedding
from src.environments.branch_and_bound_env import BranchAndBoundEnv, DEFAULT_K


class HybridGNNPolicy(nn.Module):
    """Joint Bipartite-GCN + Tree-GNN with a shared score head."""

    def __init__(
        self,
        k: int = DEFAULT_K,
        var_dim: int = 9,
        con_dim: int = 5,
        edge_dim: int = 1,
        hidden: int = 64,
        n_tree_iters: int = 3,
        cand_scalar_dim: int = 2,
    ):
        super().__init__()
        self.k = k
        self.hidden = hidden

        # LP-side encoder (per candidate)
        self.bp_encoder = BipartiteGCNEncoder(var_dim, con_dim, edge_dim, hidden)

        # Tree-side encoder (one pass over the whole tree)
        self.tree_embed = TreeNodeEmbedding(TREE_F_PER_NODE, hidden)
        self.tree_steps = nn.ModuleList(
            [TreeBottomUpStep(hidden) for _ in range(n_tree_iters)]
        )

        # Score head over concatenated [bp_vec, tr_vec, scalars]
        self.score_head = nn.Sequential(
            nn.Linear(hidden * 2 + cand_scalar_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    # ----- Tree side -----

    def _tree_features_and_edges(self, env: BranchAndBoundEnv):
        """Lift the env's enumeration tree into per-node features + parent->child edges."""
        tree = env.solver.enumeration_tree
        all_ids = list(tree.nodes())
        id_to_idx = {nid: i for i, nid in enumerate(all_ids)}

        root_v = env.solver.root_relaxation_value or 1.0
        root_v_safe = root_v if root_v != 0 else 1.0
        incumbent = env.solver.global_lower_bound

        env.open_nodes.sort(key=lambda n: -n.value)
        candidate_ids = [c.id for c in env.open_nodes[: self.k]]
        open_ids = {n.id for n in env.open_nodes}
        candidate_id_set = set(candidate_ids)
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
            is_candidate = 1.0 if nid in candidate_id_set else 0.0
            if np.isfinite(incumbent) and rel_bound != 0:
                gap = (raw_bound - incumbent) / max(abs(raw_bound), 1.0)
            else:
                gap = 0.0
            can_improve = 1.0 if (not np.isfinite(incumbent) or raw_bound > incumbent) else 0.0

            hist = np.zeros(10, dtype=np.float32)
            if nid in node_lookup and node_lookup[nid].relaxed_soln is not None:
                fp = np.abs(node_lookup[nid].relaxed_soln - np.round(node_lookup[nid].relaxed_soln))
                fp = np.clip(fp, 0.0, 0.5)
                bin_idx = np.minimum((fp * 20).astype(int), 9)
                for b in bin_idx:
                    hist[b] += 1.0
                hist = hist / max(1, len(fp))

            features.append([
                depth / 50.0, rel_bound, num_int / 50.0, num_frac / 50.0,
                is_open, is_pruned, is_integer, is_root, is_candidate,
                gap, can_improve,
                hist[0], hist[1], hist[2], hist[3], hist[4],
                hist[5], hist[6], hist[7],
            ])
        x = torch.as_tensor(features, dtype=torch.float32)

        edges = [(id_to_idx[u], id_to_idx[v]) for u, v in tree.edges()]
        if edges:
            edge_index_p_to_c = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index_p_to_c = torch.zeros((2, 0), dtype=torch.long)

        candidate_indices = torch.tensor(
            [id_to_idx[cid] for cid in candidate_ids], dtype=torch.long
        )
        return x, edge_index_p_to_c, candidate_indices

    def _tree_encode(self, env: BranchAndBoundEnv) -> Tuple[torch.Tensor, torch.Tensor]:
        x, e, ci = self._tree_features_and_edges(env)
        h = self.tree_embed(x)
        for step in self.tree_steps:
            h = step(h, e)
        return h, ci

    # ----- LP side -----

    def _lp_encode_candidates(self, env: BranchAndBoundEnv):
        """Returns (vecs (N_real, hidden), scalars (N_real, 2), mask (K,))."""
        graphs, mask = env.bipartite_observation()
        vecs = []
        scalars = []
        for i, g in enumerate(graphs):
            if not bool(mask[i].item()):
                continue
            v, _ = self.bp_encoder(g.x_var, g.x_con, g.edge_index, g.edge_attr)
            if v.shape[0] == 0:
                pooled = torch.zeros(self.hidden)
            else:
                pooled = v.mean(dim=0)
            vecs.append(pooled)
            s = torch.tensor(
                [float(g.cand_value.item()), float(g.cand_depth.item())],
                dtype=pooled.dtype,
            )
            s = s / (s.abs() + 1.0)
            scalars.append(s)
        if vecs:
            return torch.stack(vecs, dim=0), torch.stack(scalars, dim=0), mask
        empty = torch.zeros((0, self.hidden))
        empty_s = torch.zeros((0, 2))
        return empty, empty_s, mask

    # ----- Combined act() -----

    def act(self, env: BranchAndBoundEnv, deterministic: bool = False):
        # LP side: per-candidate vectors (and we get the same mask back)
        bp_vecs, scalars, mask = self._lp_encode_candidates(env)
        n_real = int(mask.sum().item())
        if n_real == 0:
            return action_for_choice(0, self.k), torch.tensor(0.0), torch.tensor(0.0)

        # Tree side: encode the entire tree, then pick out candidate embeddings
        tree_h, candidate_indices = self._tree_encode(env)
        tr_vecs = tree_h.index_select(0, candidate_indices)         # (n_real, hidden)

        # Concat per-candidate: [bp, tr, scalars] -> score
        h = torch.cat([bp_vecs, tr_vecs, scalars], dim=-1)          # (n_real, 2H + 2)
        scores = self.score_head(h).squeeze(-1)                     # (n_real,)

        full_logits = torch.full((self.k,), float("-inf"), dtype=scores.dtype, device=scores.device)
        full_logits[:n_real] = scores

        probs = F.softmax(full_logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        if deterministic:
            temp = 0.05
            cool = F.softmax(full_logits / temp, dim=-1)
            sampled = torch.distributions.Categorical(probs=cool).sample()
            choice = int(sampled.item())
            log_prob = torch.log(probs[choice] + 1e-12)
        else:
            sampled = dist.sample()
            choice = int(sampled.item())
            log_prob = dist.log_prob(sampled)
        entropy = dist.entropy()
        return action_for_choice(choice, self.k), log_prob, entropy
