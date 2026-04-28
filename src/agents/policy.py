"""
Node-selection policy for the CHOP RL environment.

The policy reads a flat observation (K * F_PER_NODE per-node features +
F_GLOBAL global features) and emits K logits over the candidate nodes.
Padded slots are masked with -inf before softmax so they're never sampled.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.environments.branch_and_bound_env import F_GLOBAL, F_PER_NODE


def action_for_choice(choice_idx: int, k: int) -> np.ndarray:
    """Convert a discrete choice in [0, K) to the env's score-vector action format."""
    a = np.full(k, -1.0, dtype=np.float32)
    a[choice_idx] = 1.0
    return a


class NodeSelectionPolicy(nn.Module):
    """MLP that scores each of K candidate nodes."""

    def __init__(self, k: int = 16, hidden: int = 64):
        super().__init__()
        self.k = k
        self.f_per_node = F_PER_NODE
        self.f_global = F_GLOBAL
        in_dim = k * F_PER_NODE + F_GLOBAL

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.head = nn.Linear(hidden, k)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (B, K) raw scores
            mask:   (B, K) bool, True = real node, False = padded slot
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        h = self.trunk(obs)
        logits = self.head(h)

        # is_real flag is the last per-node feature, located at index
        # i * F_PER_NODE + (F_PER_NODE - 1) for slot i
        node_block = obs[:, : self.k * self.f_per_node].view(-1, self.k, self.f_per_node)
        is_real = node_block[:, :, -1]
        mask = is_real > 0.5
        return logits, mask

    def sample_action(self, obs_np: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Sample a discrete choice over candidate nodes.

        Returns:
            action_vector: (K,) numpy float32 -- pluggable into env.step()
            log_prob:      scalar tensor for REINFORCE
            entropy:       scalar tensor (for optional regularization)
        """
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        logits, mask = self.forward(obs_t)
        # Mask invalid slots
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        # If everything is masked (no open nodes), fall back to uniform over slot 0
        # (env will treat the step as a no-op since open_nodes is empty)
        if not mask.any():
            choice = 0
            log_prob = torch.tensor(0.0)
            entropy = torch.tensor(0.0)
        else:
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

        action = action_for_choice(choice, self.k)
        return action, log_prob, entropy
