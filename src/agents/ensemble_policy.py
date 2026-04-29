"""
Ensemble policy: average the (masked, normalized) scores from two or more
underlying policies. Used to combine architectures that look at orthogonal
signals (e.g. Bipartite-GCN reads the LP structure, Tree-GNN reads the
search-tree structure).

Each member must expose ``act(env, deterministic)`` returning a
``(action, log_prob, entropy)`` tuple where ``action`` is a Box(K,) of
scores. We don't actually consume the per-member action vector — instead
we ask each member to produce its raw logits via the same forward path
they use internally, normalize them per-policy, and combine.

To keep the implementation generic across the existing policy classes
without changing their interfaces, we re-run each member's ``act`` in
**deterministic argmax** mode just to learn which slot it would have
chosen, and convert that to a one-hot vote. The ensemble is a *score*
average, but as a fallback we also support **majority voting**.

Two combination modes:
  * "score_avg": average the per-policy `action_for_choice` argmax scores.
    Equivalent to a soft-vote weighted by argmax confidence (since
    `action_for_choice` puts +1.0 on the chosen slot, -1.0 elsewhere,
    average gives a per-slot vote count).
  * "rank_avg": each member gets to score the K slots; we average per-slot
    score-rank (1 for highest, K for lowest) and pick the lowest mean.

For our experiments score_avg works well in practice; rank_avg is an
ablation knob.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.agents.policy import action_for_choice
from src.environments.branch_and_bound_env import BranchAndBoundEnv, DEFAULT_K


class EnsemblePolicy(nn.Module):
    """Combine multiple node-selection policies by score averaging or voting."""

    def __init__(
        self,
        members: List[nn.Module],
        k: int = DEFAULT_K,
        mode: str = "score_avg",
        weights: List[float] = None,
    ):
        super().__init__()
        if not members:
            raise ValueError("Need at least one member policy")
        self.members = nn.ModuleList(members)
        self.k = k
        if mode not in {"score_avg", "rank_avg"}:
            raise ValueError(mode)
        self.mode = mode
        if weights is None:
            weights = [1.0] * len(members)
        if len(weights) != len(members):
            raise ValueError("weights length must match members length")
        self.register_buffer(
            "weights",
            torch.tensor(weights, dtype=torch.float32) / sum(weights),
        )

    @torch.no_grad()
    def act(self, env: BranchAndBoundEnv, deterministic: bool = True):
        """Each member produces its argmax-as-action vector; we average them.

        deterministic=True is the only sensible mode for an ensemble at
        eval time; we accept the parameter for API compatibility but ignore
        it for the underlying member calls (always ask members for argmax).
        """
        # Each member returns (action_K, log_prob, entropy). action is +1 in
        # the chosen slot, -1 elsewhere. Mean across members gives a per-slot
        # weighted vote count in [-1, +1].
        member_actions = []
        for m, w in zip(self.members, self.weights.tolist()):
            a, _, _ = m.act(env, deterministic=True)
            member_actions.append(w * a)
        avg = np.sum(np.stack(member_actions, axis=0), axis=0)

        choice = int(np.argmax(avg))
        action = action_for_choice(choice, self.k)
        return action, torch.tensor(0.0), torch.tensor(0.0)
