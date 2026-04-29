"""
Bipartite-graph GCN policy for CHOP node selection.

Architecture closely follows Gasse, Chetelat, Ferroni, Charlin & Lodi (2019),
"Exact Combinatorial Optimization with Graph Convolutional Neural Networks"
(NeurIPS), https://arxiv.org/abs/1906.01629 -- adapted from their *branching*
task to our *node-selection* task.

Per env step:
  1. Take the K best-LP-bound candidate open nodes.
  2. For each candidate, build its current LP as a variable-constraint
     bipartite graph (env.bipartite_observation()).
  3. Embed variable features V (per-cand, per-var) and constraint features C
     to a common hidden width.
  4. Two interleaved half-convolutions:
       C <- f_C(C, sum over neighbours of g_C(C, V, E))
       V <- f_V(V, sum over neighbours of g_V(C, V, E))
     where f_*, g_* are 2-layer MLPs with ReLU.
  5. Mean-pool the per-candidate variable embeddings to a single vector,
     concat per-candidate scalars (LP value, depth), and project to a score.
  6. Mask out padded slots, softmax, sample (or argmax) -- exactly matching
     the action contract used by the MLP / GNN / Transformer policies.

Critical Gasse trick: prenorm layer (empirical mean/std applied AFTER the
neighbour sum, BEFORE the update MLP). Per the paper this stabilises
training and -- more importantly for us -- improves generalization to
larger problem instances. Implemented here as a learned BatchNorm1d-style
running-stat normalizer over the aggregated messages.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.policy import action_for_choice
from src.environments.branch_and_bound_env import BranchAndBoundEnv, DEFAULT_K


def _two_layer_mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class BipartiteHalfConv(nn.Module):
    """One pass of C->V or V->C bipartite message passing (Gasse Eq. 4).

    Source-side embeddings + edge features feed g; aggregated messages feed
    a prenorm layer; concatenated with the destination embedding then run
    through the update MLP f.
    """

    def __init__(self, src_dim: int, dst_dim: int, edge_dim: int, hidden: int):
        super().__init__()
        self.g = _two_layer_mlp(src_dim + dst_dim + edge_dim, hidden, hidden)
        self.f = nn.Sequential(
            nn.Linear(dst_dim + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        # Prenorm: BatchNorm1d over aggregated messages. Init means
        # close-to-identity but with running stats it self-calibrates.
        self.prenorm = nn.BatchNorm1d(hidden)

    def forward(
        self,
        src: torch.Tensor,        # (N_src, src_dim)
        dst: torch.Tensor,        # (N_dst, dst_dim)
        edge_index: torch.Tensor, # (2, E) row 0 = src idx, row 1 = dst idx
        edge_attr: torch.Tensor,  # (E, edge_dim)
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            # No edges -> nothing to aggregate; fall back to a zero update
            zeros = torch.zeros(dst.shape[0], self.g[-1].out_features,
                                 device=dst.device, dtype=dst.dtype)
            agg = zeros
        else:
            src_idx = edge_index[0]
            dst_idx = edge_index[1]
            src_e = src.index_select(0, src_idx)
            dst_e = dst.index_select(0, dst_idx)
            inputs = torch.cat([src_e, dst_e, edge_attr], dim=-1)
            messages = self.g(inputs)              # (E, hidden)
            agg = torch.zeros(
                dst.shape[0], messages.shape[1],
                device=messages.device, dtype=messages.dtype,
            )
            agg = agg.index_add(0, dst_idx, messages)

        # Prenorm: only meaningful with >1 sample in batch
        if agg.shape[0] > 1:
            agg = self.prenorm(agg)

        out = self.f(torch.cat([dst, agg], dim=-1))
        return out


class BipartiteGCNEncoder(nn.Module):
    """Embeds a single bipartite LP into per-variable and per-constraint
    hidden vectors via initial linear embedding + one C->V->C pass.
    """

    def __init__(self, var_dim: int, con_dim: int, edge_dim: int, hidden: int = 64):
        super().__init__()
        self.var_init = nn.Linear(var_dim, hidden)
        self.con_init = nn.Linear(con_dim, hidden)

        # First pass: aggregate V into C
        self.v_to_c = BipartiteHalfConv(
            src_dim=hidden, dst_dim=hidden, edge_dim=edge_dim, hidden=hidden,
        )
        # Second pass: aggregate (now-updated) C into V
        self.c_to_v = BipartiteHalfConv(
            src_dim=hidden, dst_dim=hidden, edge_dim=edge_dim, hidden=hidden,
        )

    def forward(
        self,
        x_var: torch.Tensor,
        x_con: torch.Tensor,
        edge_index: torch.Tensor,  # (2, E): row 0 = constraint idx, row 1 = variable idx
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        v = F.relu(self.var_init(x_var))
        c = F.relu(self.con_init(x_con))

        # Pass 1: V -> C
        # The bipartite_observation method stores edge_index with row 0 = constraint,
        # row 1 = variable. To send V messages to C we need src=variable (row 1),
        # dst=constraint (row 0).
        e_v_to_c = torch.stack([edge_index[1], edge_index[0]], dim=0)
        c = self.v_to_c(v, c, e_v_to_c, edge_attr)

        # Pass 2: C -> V (now using updated c)
        e_c_to_v = torch.stack([edge_index[0], edge_index[1]], dim=0)
        v = self.c_to_v(c, v, e_c_to_v, edge_attr)

        return v, c


class BipartiteGCNNodeSelectionPolicy(nn.Module):
    """One bipartite GCN encoder, applied per candidate, then pooled and
    scored.

    For PPO replay convenience the policy exposes both ``act(env)`` (which
    pulls observations from the env directly) and ``forward_from_graphs``
    (which takes pre-extracted graph lists; useful inside trainers that
    snapshot the rollout for off-policy updates).
    """

    def __init__(
        self,
        k: int = DEFAULT_K,
        var_dim: int = 9,
        con_dim: int = 5,
        edge_dim: int = 1,
        hidden: int = 64,
        cand_scalar_dim: int = 2,  # cand_value, cand_depth
    ):
        super().__init__()
        self.k = k
        self.hidden = hidden
        self.encoder = BipartiteGCNEncoder(var_dim, con_dim, edge_dim, hidden)

        # Score head: per-candidate vector of [pooled_var; cand_scalars] -> scalar
        self.score_head = nn.Sequential(
            nn.Linear(hidden + cand_scalar_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _encode_one_candidate(self, data) -> torch.Tensor:
        """Run one forward pass of the encoder + pool + score head -> scalar."""
        v, _ = self.encoder(data.x_var, data.x_con, data.edge_index, data.edge_attr)
        if v.shape[0] == 0:
            pooled = torch.zeros(1, self.hidden, device=v.device, dtype=v.dtype)
        else:
            pooled = v.mean(dim=0, keepdim=True)
        scalars = torch.tensor(
            [[float(data.cand_value.item()), float(data.cand_depth.item())]],
            dtype=pooled.dtype, device=pooled.device,
        )
        # Normalize scalars to keep magnitudes comparable to the pooled vector.
        scalars = scalars / (scalars.abs() + 1.0)
        x = torch.cat([pooled, scalars], dim=-1)
        score = self.score_head(x).squeeze()
        return score

    def forward_from_graphs(
        self, graphs: List, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits (K,), mask (K,) bool)."""
        scores = torch.full((self.k,), float("-inf"))
        for i, g in enumerate(graphs):
            if not bool(mask[i].item()):
                continue
            scores[i] = self._encode_one_candidate(g)
        return scores, mask

    def act(self, env: BranchAndBoundEnv, deterministic: bool = False):
        graphs, mask = env.bipartite_observation()
        n_real = int(mask.sum().item())

        if n_real == 0:
            return action_for_choice(0, self.k), torch.tensor(0.0), torch.tensor(0.0)

        logits, _ = self.forward_from_graphs(graphs, mask)
        probs = F.softmax(logits, dim=-1)
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
