"""
Bipartite-GCN encoder followed by cross-candidate self-attention.

Motivation: the Gasse-style bipartite GCN encodes each candidate's LP
*independently*. The score head then projects each candidate's embedding
to a scalar without ever seeing the other candidates.

But ranking is fundamentally a comparative task — to score candidate i
well, you should know what candidates j != i look like. Self-attention
across the K candidate embeddings lets each candidate's score condition
on the others. This is the standard "set ranker" formulation
(Pasupat et al., Vinyals et al. pointer networks), grafted onto Gasse's
LP encoder.

The hope: catches cases where two candidates have similar LP bounds but
very different structural promise — the attention layer learns "candidate
i looks better than candidate j given everything else on the table."

Architecture:
    BipartiteGCN per candidate -> mean-pool -> per-candidate vector
    Stack K candidate vectors -> [global token | candidate vectors]
    N transformer encoder layers (self-attention + FFN), padding-aware mask
    Per-candidate linear -> score
    Softmax over K with mask -> sample / argmax
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.bipartite_gnn_policy import BipartiteGCNEncoder
from src.agents.policy import action_for_choice
from src.environments.branch_and_bound_env import BranchAndBoundEnv, DEFAULT_K


class BipartiteAttentionPolicy(nn.Module):
    """Bipartite-GCN per candidate, then cross-candidate self-attention."""

    def __init__(
        self,
        k: int = DEFAULT_K,
        var_dim: int = 9,
        con_dim: int = 5,
        edge_dim: int = 1,
        hidden: int = 64,
        n_attn_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.k = k
        self.hidden = hidden

        self.encoder = BipartiteGCNEncoder(var_dim, con_dim, edge_dim, hidden)

        # Per-candidate scalar embedding (LP value, depth)
        self.scalar_embed = nn.Linear(2, hidden)

        # Self-attention across candidates
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.attn = nn.TransformerEncoder(encoder_layer, num_layers=n_attn_layers)

        self.score_head = nn.Linear(hidden, 1)

    def _encode_one_candidate(self, data) -> torch.Tensor:
        """Run the bipartite encoder + mean-pool + add scalar embedding -> per-candidate vector."""
        v, _ = self.encoder(data.x_var, data.x_con, data.edge_index, data.edge_attr)
        if v.shape[0] == 0:
            pooled = torch.zeros(self.hidden, dtype=torch.float32)
        else:
            pooled = v.mean(dim=0)
        scalars = torch.tensor(
            [float(data.cand_value.item()), float(data.cand_depth.item())],
            dtype=pooled.dtype, device=pooled.device,
        )
        scalars = scalars / (scalars.abs() + 1.0)
        scalar_h = self.scalar_embed(scalars)
        return pooled + scalar_h

    def _candidate_vectors(self, env: BranchAndBoundEnv):
        """Returns (K, hidden) tensor of per-candidate vectors plus a (K,) bool mask.

        Padded slots get zero vectors and mask=False; the transformer's
        src_key_padding_mask makes attention skip them.
        """
        graphs, mask = env.bipartite_observation()
        vecs = []
        for i, g in enumerate(graphs):
            if bool(mask[i].item()):
                vecs.append(self._encode_one_candidate(g))
            else:
                vecs.append(torch.zeros(self.hidden, dtype=torch.float32))
        return torch.stack(vecs, dim=0), mask

    def forward(self, env: BranchAndBoundEnv) -> Tuple[torch.Tensor, torch.Tensor]:
        cand_vecs, mask = self._candidate_vectors(env)            # (K, H), (K,)
        # batch dim
        tokens = cand_vecs.unsqueeze(0)                            # (1, K, H)
        # padding mask: True = ignore
        key_padding_mask = (~mask).unsqueeze(0)                    # (1, K)
        encoded = self.attn(tokens, src_key_padding_mask=key_padding_mask)  # (1, K, H)
        scores = self.score_head(encoded).squeeze(-1).squeeze(0)   # (K,)
        return scores, mask

    def act(self, env: BranchAndBoundEnv, deterministic: bool = False):
        scores, mask = self.forward(env)
        n_real = int(mask.sum().item())
        if n_real == 0:
            return action_for_choice(0, self.k), torch.tensor(0.0), torch.tensor(0.0)

        masked_logits = scores.masked_fill(~mask, float("-inf"))
        probs = F.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        if deterministic:
            # Low-temperature Boltzmann to avoid pure-argmax tiebreak collapse
            temp = 0.05
            cool = F.softmax(masked_logits / temp, dim=-1)
            sampled = torch.distributions.Categorical(probs=cool).sample()
            choice = int(sampled.item())
            log_prob = torch.log(probs[choice] + 1e-12)
        else:
            sampled = dist.sample()
            choice = int(sampled.item())
            log_prob = dist.log_prob(sampled)
        entropy = dist.entropy()
        return action_for_choice(choice, self.k), log_prob, entropy
