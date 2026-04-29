"""
Imitation-learning warm-start for CHOP node-selection policies.

Idea: instead of waiting for REINFORCE to discover the BestBound heuristic
through trial and error, distill BestBound (or any HeuristicAgent) directly
into the policy with supervised cross-entropy on (observation, choice) pairs
collected from heuristic rollouts. The resulting policy is "BestBound on
day one" -- and a subsequent RL fine-tune can spend its budget improving on
top of that prior rather than rediscovering it.

Works with any policy that exposes ``forward(obs_tensor) -> (logits, mask)``.
That's the MLP and the Transformer; the GNN would need its own collector.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.environments.branch_and_bound_env import (
    BranchAndBoundEnv,
    DEFAULT_K,
    HeuristicAgent,
)


@dataclass
class ImitationConfig:
    n_rollout_episodes: int = 200    # how many heuristic rollouts to record
    epochs: int = 20                  # supervised epochs over the buffer
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 0
    log_every: int = 5
    weight_decay: float = 1e-5


@dataclass
class ImitationStats:
    epoch: List[int] = field(default_factory=list)
    loss: List[float] = field(default_factory=list)
    accuracy: List[float] = field(default_factory=list)
    elapsed_s: List[float] = field(default_factory=list)


def collect_heuristic_rollouts(
    env_factory: Callable[[int], BranchAndBoundEnv],
    expert_mode: str,
    n_episodes: int,
    k_nodes: int,
    seed: int,
):
    """Run a heuristic agent for N episodes; return arrays of (obs, choice)."""
    obs_list: List[np.ndarray] = []
    choice_list: List[int] = []

    for ep in range(n_episodes):
        env = env_factory(seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        agent = HeuristicAgent(mode=expert_mode, k=k_nodes)
        agent.reset(seed=seed + ep)
        done, truncated = False, False
        while not (done or truncated):
            action = agent.act(obs)
            chosen = int(np.argmax(action))
            obs_list.append(obs.copy())
            choice_list.append(chosen)
            obs, _, done, truncated, _ = env.step(action)

    return np.stack(obs_list), np.array(choice_list, dtype=np.int64)


class ImitationLearner:
    """Supervised cross-entropy distillation of a HeuristicAgent into a policy."""

    def __init__(
        self,
        policy: torch.nn.Module,
        env_factory: Callable[[int], BranchAndBoundEnv],
        config: Optional[ImitationConfig] = None,
        expert_mode: str = "best_bound",
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.env_factory = env_factory
        self.cfg = config or ImitationConfig()
        self.expert_mode = expert_mode
        self.device = device
        self.optimizer = optim.Adam(
            policy.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        self.stats = ImitationStats()

    def train(self) -> ImitationStats:
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        start = time.time()

        # Collect expert rollouts once
        obs_arr, choice_arr = collect_heuristic_rollouts(
            self.env_factory, self.expert_mode, self.cfg.n_rollout_episodes,
            self.policy.k, self.cfg.seed,
        )
        n = obs_arr.shape[0]
        print(f"Collected {n} (obs, choice) pairs from {self.cfg.n_rollout_episodes} "
              f"{self.expert_mode} rollouts (took {time.time()-start:.1f}s)")

        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
        choice_t = torch.as_tensor(choice_arr, dtype=torch.long, device=self.device)

        idx = np.arange(n)
        for epoch in range(self.cfg.epochs):
            np.random.shuffle(idx)
            losses, accs = [], []
            for start_i in range(0, n, self.cfg.batch_size):
                bi = torch.as_tensor(idx[start_i:start_i + self.cfg.batch_size],
                                     dtype=torch.long, device=self.device)
                b_obs = obs_t.index_select(0, bi)
                b_choice = choice_t.index_select(0, bi)

                logits, mask = self.policy(b_obs)
                masked_logits = logits.masked_fill(~mask, float("-inf"))
                loss = F.cross_entropy(masked_logits, b_choice)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                preds = masked_logits.argmax(dim=-1)
                acc = (preds == b_choice).float().mean().item()
                losses.append(float(loss.item()))
                accs.append(acc)

            self.stats.epoch.append(epoch)
            self.stats.loss.append(float(np.mean(losses)))
            self.stats.accuracy.append(float(np.mean(accs)))
            self.stats.elapsed_s.append(time.time() - start)

            if (epoch + 1) % self.cfg.log_every == 0 or epoch == 0:
                print(f"epoch {epoch+1:>3d} | loss {np.mean(losses):>6.4f} | "
                      f"acc {100*np.mean(accs):>5.1f}% | elapsed {time.time()-start:>5.1f}s")

        return self.stats

    @torch.no_grad()
    def evaluate(self, n_eval: int, deterministic: bool = True) -> dict:
        nodes, completed = [], []
        for ep in range(n_eval):
            env = self.env_factory(30_000 + ep)
            env.reset(seed=30_000 + ep)
            done, truncated = False, False
            while not (done or truncated):
                action, _, _ = self.policy.act(env, deterministic=deterministic)
                _, _, done, truncated, info = env.step(action)
            nodes.append(info["nodes_explored"])
            completed.append(done)
        return {
            "nodes_mean": float(np.mean(nodes)),
            "nodes_std": float(np.std(nodes)),
            "solved_frac": float(np.mean(completed)),
        }
