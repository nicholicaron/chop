"""
Graph-neural-network policy for CHOP node selection.

Reads the entire B&B enumeration tree as a graph (via env.graph_observation),
propagates features with GCN convolutions, and scores the top-K candidate
nodes. Output matches the env's Box(K,) action contract (1.0 in the chosen
slot, -1.0 elsewhere) and the same (action, log_prob, entropy) tuple that
the MLP policy returns, so ReinforceTrainer is policy-agnostic.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.policy import action_for_choice
from src.environments.branch_and_bound_env import BranchAndBoundEnv, DEFAULT_K


class GNNNodeSelectionPolicy(nn.Module):
    """GCN over the B&B tree, scoring the top-K candidate nodes."""

    def __init__(self, k: int = DEFAULT_K, in_dim: int = 9, hidden: int = 64, n_conv: int = 2):
        super().__init__()
        from torch_geometric.nn import GCNConv  # lazy: keeps import-time cheap

        self.k = k
        self.in_dim = in_dim

        layers = []
        d_in = in_dim
        for _ in range(n_conv):
            layers.append(GCNConv(d_in, hidden))
            d_in = hidden
        self.convs = nn.ModuleList(layers)

        self.score_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def _gnn_forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
        return h

    def act(self, env: BranchAndBoundEnv, deterministic: bool = False) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        data, candidate_indices = env.graph_observation()
        n_real = int(candidate_indices.numel())

        if n_real == 0:
            return action_for_choice(0, self.k), torch.tensor(0.0), torch.tensor(0.0)

        h = self._gnn_forward(data.x, data.edge_index)
        cand_h = h.index_select(0, candidate_indices)
        scores = self.score_head(cand_h).squeeze(-1)  # (n_real,)

        # Pad up to K with -inf so the categorical never samples a missing slot
        full_logits = torch.full((self.k,), float("-inf"), dtype=scores.dtype, device=scores.device)
        full_logits[:n_real] = scores

        probs = F.softmax(full_logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        if deterministic:
            # Boltzmann sampling at low temperature instead of pure argmax,
            # so near-tied scores don't always tiebreak to the same index
            # (the bug where the GNN's eval collapsed to best_bound's choice).
            temperature = 0.05
            cool_logits = full_logits / temperature
            cool_probs = F.softmax(cool_logits, dim=-1)
            cool_dist = torch.distributions.Categorical(probs=cool_probs)
            sampled = cool_dist.sample()
            choice = int(sampled.item())
            log_prob = torch.log(probs[choice] + 1e-12)
        else:
            sampled = dist.sample()
            choice = int(sampled.item())
            log_prob = dist.log_prob(sampled)
        entropy = dist.entropy()

        return action_for_choice(choice, self.k), log_prob, entropy

    @torch.no_grad()
    def evaluate(self, env: BranchAndBoundEnv) -> np.ndarray:
        """Pure-inference helper: returns the K-length action vector deterministically."""
        action, _, _ = self.act(env, deterministic=True)
        return action
