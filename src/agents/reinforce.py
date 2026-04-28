"""
REINFORCE trainer for the CHOP node-selection policy.

REINFORCE-with-baseline: episode return drives the policy gradient; the
moving-average return acts as a variance-reduction baseline. Includes a
small entropy bonus to keep early exploration alive.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.optim as optim

from src.environments.branch_and_bound_env import BranchAndBoundEnv


@dataclass
class TrainConfig:
    n_episodes: int = 400
    lr: float = 5e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01
    baseline_decay: float = 0.95
    log_every: int = 20
    grad_clip: float = 1.0
    seed: int = 0


@dataclass
class TrainStats:
    episode: List[int] = field(default_factory=list)
    nodes_explored: List[int] = field(default_factory=list)
    return_total: List[float] = field(default_factory=list)
    obj_value: List[float] = field(default_factory=list)
    completed: List[bool] = field(default_factory=list)
    elapsed_s: List[float] = field(default_factory=list)
    pg_loss: List[float] = field(default_factory=list)


class ReinforceTrainer:
    """Policy-agnostic REINFORCE trainer.

    The policy must expose ``act(env, deterministic=False)`` returning a
    ``(action, log_prob, entropy)`` tuple. Both the MLP and GNN policies
    in src/agents/ follow this contract.
    """

    def __init__(
        self,
        policy: torch.nn.Module,
        env_factory: Callable[[int], BranchAndBoundEnv],
        config: Optional[TrainConfig] = None,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.env_factory = env_factory
        self.cfg = config or TrainConfig()
        self.device = device
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.cfg.lr)
        self.baseline = 0.0
        self.stats = TrainStats()

    def _run_episode(self, episode_idx: int) -> dict:
        env = self.env_factory(self.cfg.seed + episode_idx)
        env.reset(seed=self.cfg.seed + episode_idx)

        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        rewards: List[float] = []

        done, truncated = False, False
        while not (done or truncated):
            action, log_prob, entropy = self.policy.act(env, deterministic=False)
            _, reward, done, truncated, info = env.step(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
            rewards.append(reward)

        # Discounted returns (gamma close to 1 → ~undiscounted)
        returns = np.zeros(len(rewards), dtype=np.float64)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + self.cfg.gamma * running
            returns[t] = running

        return {
            "log_probs": log_probs,
            "entropies": entropies,
            "rewards": rewards,
            "returns": returns,
            "info": info,
            "completed": done,
        }

    def _update(self, episode: dict) -> float:
        returns = torch.as_tensor(episode["returns"], dtype=torch.float32, device=self.device)
        # Update baseline (EMA of episode return) before using it
        ep_return = float(returns[0].item()) if len(returns) else 0.0
        self.baseline = self.cfg.baseline_decay * self.baseline + (1 - self.cfg.baseline_decay) * ep_return
        advantages = returns - self.baseline

        log_probs = torch.stack(episode["log_probs"])
        entropies = torch.stack(episode["entropies"])

        pg_loss = -(log_probs * advantages.detach()).mean()
        ent_loss = -self.cfg.entropy_coef * entropies.mean()
        loss = pg_loss + ent_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        return float(pg_loss.item())

    def train(self) -> TrainStats:
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        start = time.time()

        for ep in range(self.cfg.n_episodes):
            episode = self._run_episode(ep)
            pg_loss = self._update(episode) if episode["log_probs"] else 0.0

            info = episode["info"]
            self.stats.episode.append(ep)
            self.stats.nodes_explored.append(info["nodes_explored"])
            self.stats.return_total.append(float(np.sum(episode["rewards"])))
            self.stats.obj_value.append(info["current_best_obj"])
            self.stats.completed.append(bool(episode["completed"]))
            self.stats.elapsed_s.append(time.time() - start)
            self.stats.pg_loss.append(pg_loss)

            if (ep + 1) % self.cfg.log_every == 0 or ep == 0:
                window = slice(max(0, ep + 1 - self.cfg.log_every), ep + 1)
                avg_nodes = np.mean(self.stats.nodes_explored[window])
                avg_ret = np.mean(self.stats.return_total[window])
                solved = np.mean(self.stats.completed[window]) * 100
                print(
                    f"ep {ep+1:>4d} | nodes {avg_nodes:>6.1f} | return {avg_ret:>+8.1f} | "
                    f"solved {solved:>5.1f}% | baseline {self.baseline:>+7.1f} | "
                    f"elapsed {time.time()-start:>5.1f}s"
                )
        return self.stats

    @torch.no_grad()
    def evaluate(self, n_eval: int, deterministic: bool = True) -> dict:
        nodes, objs, completed = [], [], []
        for ep in range(n_eval):
            env = self.env_factory(10_000 + ep)
            env.reset(seed=10_000 + ep)
            done, truncated = False, False
            while not (done or truncated):
                action, _, _ = self.policy.act(env, deterministic=deterministic)
                _, _, done, truncated, info = env.step(action)
            nodes.append(info["nodes_explored"])
            objs.append(info["current_best_obj"])
            completed.append(done)
        return {
            "nodes_mean": float(np.mean(nodes)),
            "nodes_std": float(np.std(nodes)),
            "obj_mean": float(np.mean(objs)),
            "solved_frac": float(np.mean(completed)),
            "n_eval": n_eval,
        }
