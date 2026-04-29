"""
Transformer / self-attention policy for CHOP node selection.

Architecture rationale: the agent's job is to *rank* K candidate nodes.
Self-attention over the candidate set is a natural inductive bias --
each candidate's score can depend on relative comparisons with all the
others, not just on its own features. This is the standard "set ranker"
formulation used in pointer-network and learning-to-rank literature.

Input:  K x F_PER_NODE per-candidate features + F_GLOBAL global features
Forward:
  1. Embed each candidate to hidden dim (linear)
  2. Concatenate global features as a "[CLS]-style" extra token
  3. N transformer encoder layers (self-attention + FFN)
  4. Project each candidate token to a scalar score
Mask: padding-aware (is_real flag in features) so attention never attends to
fake slots and the final softmax never samples one.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.policy import action_for_choice
from src.environments.branch_and_bound_env import (
    BranchAndBoundEnv,
    DEFAULT_K,
    F_GLOBAL,
    F_PER_NODE,
)


class TransformerNodeSelectionPolicy(nn.Module):
    """Self-attention over the K candidate nodes with a pooled global token."""

    def __init__(
        self,
        k: int = DEFAULT_K,
        hidden: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.k = k
        self.f_per_node = F_PER_NODE
        self.f_global = F_GLOBAL
        self.hidden = hidden

        self.cand_embed = nn.Linear(F_PER_NODE, hidden)
        self.global_embed = nn.Linear(F_GLOBAL, hidden)

        # Standard transformer encoder. batch_first=True so input is (B, K+1, H).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.score_head = nn.Linear(hidden, 1)

    def _split_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """obs: (B, K*F_PER_NODE + F_GLOBAL) -> (cand_feats, global_feats, mask)."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        cand_part = obs[:, : self.k * self.f_per_node].view(-1, self.k, self.f_per_node)
        global_part = obs[:, self.k * self.f_per_node:]
        mask = cand_part[:, :, -1] > 0.5  # (B, K) -- True for real candidates
        return cand_part, global_part, mask

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits (B, K), mask (B, K) bool)."""
        cand_feats, global_feats, mask = self._split_obs(obs)

        cand_emb = self.cand_embed(cand_feats)        # (B, K, H)
        global_emb = self.global_embed(global_feats).unsqueeze(1)  # (B, 1, H)
        tokens = torch.cat([global_emb, cand_emb], dim=1)  # (B, K+1, H)

        # Build src_key_padding_mask: True at positions to be ignored.
        # Global token (position 0) is always real.
        global_token_mask = torch.zeros(mask.shape[0], 1, dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([global_token_mask, ~mask], dim=1)  # (B, K+1)

        encoded = self.encoder(tokens, src_key_padding_mask=full_mask)  # (B, K+1, H)

        # Drop the global token; per-candidate logits from the rest.
        cand_encoded = encoded[:, 1:, :]
        logits = self.score_head(cand_encoded).squeeze(-1)  # (B, K)
        return logits, mask

    def act(self, env: BranchAndBoundEnv, deterministic: bool = False) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        obs = env._observation()
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, mask = self.forward(obs_t)
        masked_logits = logits.masked_fill(~mask, float("-inf"))

        if not mask.any():
            return action_for_choice(0, self.k), torch.tensor(0.0), torch.tensor(0.0)

        probs = F.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        if deterministic:
            choice = int(torch.argmax(probs, dim=-1).item())
            log_prob = torch.log(probs[0, choice] + 1e-12)
        else:
            sampled = dist.sample()
            choice = int(sampled.item())
            log_prob = dist.log_prob(sampled).squeeze(0)
        entropy = dist.entropy().squeeze(0)

        return action_for_choice(choice, self.k), log_prob, entropy

    def sample_action(self, obs_np: np.ndarray, deterministic: bool = False):
        """Mirrors NodeSelectionPolicy.sample_action so utils.eval treats it the same."""
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        logits, mask = self.forward(obs_t)
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        if not mask.any():
            return action_for_choice(0, self.k), torch.tensor(0.0), torch.tensor(0.0)
        probs = F.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        if deterministic:
            choice = int(torch.argmax(probs, dim=-1).item())
            log_prob = torch.log(probs[0, choice] + 1e-12)
        else:
            sampled = dist.sample()
            choice = int(sampled.item())
            log_prob = dist.log_prob(sampled).squeeze(0)
        return action_for_choice(choice, self.k), log_prob, dist.entropy().squeeze(0)
