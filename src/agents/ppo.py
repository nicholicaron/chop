"""
PPO trainer for the CHOP node-selection MLP policy.

Differences vs. ReinforceTrainer:
  * Collects N episodes of rollouts before updating (minibatching across
    them in K epochs per update).
  * Uses GAE(lambda) for advantage estimation.
  * Adds a separate value network (small MLP) trained with MSE.
  * Clipped surrogate loss + entropy bonus.

For now this is MLP-only; the GNN variant would need parallel
graph_observation replay and is left to follow-up work.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.agents.policy import NodeSelectionPolicy, action_for_choice
from src.environments.branch_and_bound_env import (
    BranchAndBoundEnv,
    DEFAULT_K,
    F_GLOBAL,
    F_PER_NODE,
)


@dataclass
class PPOConfig:
    n_iterations: int = 50      # how many rollout / update cycles
    episodes_per_iter: int = 10  # rollout episodes per iteration
    lr_policy: float = 3e-4
    lr_value: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    update_epochs: int = 4
    minibatch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip: float = 0.5
    log_every: int = 1
    seed: int = 0


@dataclass
class PPOStats:
    iteration: List[int] = field(default_factory=list)
    nodes_explored_mean: List[float] = field(default_factory=list)
    return_mean: List[float] = field(default_factory=list)
    completed_frac: List[float] = field(default_factory=list)
    pg_loss: List[float] = field(default_factory=list)
    value_loss: List[float] = field(default_factory=list)
    entropy: List[float] = field(default_factory=list)
    elapsed_s: List[float] = field(default_factory=list)


class ValueNet(nn.Module):
    """Small MLP value network over the env's flat observation."""

    def __init__(self, k: int = DEFAULT_K, hidden: int = 64):
        super().__init__()
        in_dim = k * F_PER_NODE + F_GLOBAL
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.net(obs).squeeze(-1)


class PPOTrainer:
    """Single-env PPO with GAE for the MLP node-selection policy."""

    def __init__(
        self,
        policy: NodeSelectionPolicy,
        env_factory: Callable[[int], BranchAndBoundEnv],
        config: Optional[PPOConfig] = None,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.value_net = ValueNet(k=policy.k, hidden=64).to(device)
        self.env_factory = env_factory
        self.cfg = config or PPOConfig()
        self.device = device

        self.optim_p = optim.Adam(self.policy.parameters(), lr=self.cfg.lr_policy)
        self.optim_v = optim.Adam(self.value_net.parameters(), lr=self.cfg.lr_value)
        self.stats = PPOStats()

    # ----- Rollouts -----

    def _collect_rollouts(self, iter_idx: int) -> dict:
        all_obs: List[np.ndarray] = []
        all_actions: List[int] = []  # discrete choice
        all_log_probs: List[float] = []
        all_values: List[float] = []
        all_rewards: List[float] = []
        all_dones: List[bool] = []
        ep_lens: List[int] = []
        ep_returns: List[float] = []
        ep_completed: List[bool] = []

        for ep in range(self.cfg.episodes_per_iter):
            seed = self.cfg.seed + iter_idx * self.cfg.episodes_per_iter + ep
            env = self.env_factory(seed)
            obs, _ = env.reset(seed=seed)

            done, truncated = False, False
            ep_reward = 0.0
            ep_steps = 0
            while not (done or truncated):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    logits, mask = self.policy(obs_t)

                    # If the queue is empty there's nothing to act on — the env
                    # will just emit a terminal transition. Skip storage so we
                    # don't poison the update with all-(-inf) softmax rows.
                    if not mask.any():
                        choice = 0
                        action_vec = action_for_choice(choice, self.policy.k)
                        _, reward, done, truncated, info = env.step(action_vec)
                        ep_reward += reward
                        ep_steps += 1
                        continue

                    masked_logits = logits.masked_fill(~mask, float("-inf"))
                    probs = F.softmax(masked_logits, dim=-1)
                    dist = torch.distributions.Categorical(probs=probs)
                    choice_t = dist.sample()
                    choice = int(choice_t.item())
                    log_prob = float(dist.log_prob(choice_t).item())
                    value = float(self.value_net(obs_t).item())

                action_vec = action_for_choice(choice, self.policy.k)
                next_obs, reward, done, truncated, info = env.step(action_vec)

                all_obs.append(obs.copy())
                all_actions.append(choice)
                all_log_probs.append(log_prob)
                all_values.append(value)
                all_rewards.append(reward)
                all_dones.append(done or truncated)
                ep_reward += reward
                ep_steps += 1
                obs = next_obs

            ep_lens.append(ep_steps)
            ep_returns.append(ep_reward)
            ep_completed.append(done)

        return {
            "obs": np.stack(all_obs),
            "actions": np.array(all_actions, dtype=np.int64),
            "log_probs": np.array(all_log_probs, dtype=np.float32),
            "values": np.array(all_values, dtype=np.float32),
            "rewards": np.array(all_rewards, dtype=np.float32),
            "dones": np.array(all_dones, dtype=np.bool_),
            "ep_lens": ep_lens,
            "ep_returns": ep_returns,
            "ep_completed": ep_completed,
        }

    # ----- Advantage / target computation (GAE) -----

    def _compute_gae(self, rollout: dict) -> tuple[np.ndarray, np.ndarray]:
        rewards = rollout["rewards"]
        values = rollout["values"]
        dones = rollout["dones"]
        gamma, lam = self.cfg.gamma, self.cfg.gae_lambda

        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        last_adv = 0.0
        for t in reversed(range(T)):
            next_value = 0.0 if t == T - 1 or dones[t] else values[t + 1]
            non_terminal = 0.0 if dones[t] else 1.0
            delta = rewards[t] + gamma * next_value * non_terminal - values[t]
            last_adv = delta + gamma * lam * non_terminal * last_adv
            adv[t] = last_adv

        returns = adv + values
        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, returns

    # ----- Update -----

    def _update(self, rollout: dict, advantages: np.ndarray, returns: np.ndarray) -> dict:
        obs = torch.as_tensor(rollout["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(rollout["actions"], dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(rollout["log_probs"], dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        n = obs.shape[0]
        idx = np.arange(n)
        pg_losses, v_losses, entropies = [], [], []

        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.cfg.minibatch_size):
                batch_idx = idx[start:start + self.cfg.minibatch_size]
                bi = torch.as_tensor(batch_idx, dtype=torch.long, device=self.device)

                b_obs = obs.index_select(0, bi)
                b_actions = actions.index_select(0, bi)
                b_old_lp = old_log_probs.index_select(0, bi)
                b_adv = adv.index_select(0, bi)
                b_ret = ret.index_select(0, bi)

                logits, mask = self.policy(b_obs)
                masked_logits = logits.masked_fill(~mask, float("-inf"))
                probs = F.softmax(masked_logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs + 1e-12)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * b_adv
                pg_loss = -torch.min(surr1, surr2).mean()

                values_pred = self.value_net(b_obs)
                v_loss = F.mse_loss(values_pred, b_ret)

                loss = pg_loss + self.cfg.value_coef * v_loss - self.cfg.entropy_coef * entropy

                self.optim_p.zero_grad()
                self.optim_v.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.cfg.grad_clip)
                self.optim_p.step()
                self.optim_v.step()

                pg_losses.append(float(pg_loss.item()))
                v_losses.append(float(v_loss.item()))
                entropies.append(float(entropy.item()))

        return {
            "pg_loss": float(np.mean(pg_losses)),
            "value_loss": float(np.mean(v_losses)),
            "entropy": float(np.mean(entropies)),
        }

    # ----- Public API -----

    def train(self) -> PPOStats:
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        start = time.time()

        for it in range(self.cfg.n_iterations):
            rollout = self._collect_rollouts(it)
            advantages, returns = self._compute_gae(rollout)
            update_metrics = self._update(rollout, advantages, returns)

            mean_nodes = float(np.mean(rollout["ep_lens"]))
            mean_return = float(np.mean(rollout["ep_returns"]))
            solved_frac = float(np.mean(rollout["ep_completed"]))

            self.stats.iteration.append(it)
            self.stats.nodes_explored_mean.append(mean_nodes)
            self.stats.return_mean.append(mean_return)
            self.stats.completed_frac.append(solved_frac)
            self.stats.pg_loss.append(update_metrics["pg_loss"])
            self.stats.value_loss.append(update_metrics["value_loss"])
            self.stats.entropy.append(update_metrics["entropy"])
            self.stats.elapsed_s.append(time.time() - start)

            if (it + 1) % self.cfg.log_every == 0 or it == 0:
                print(
                    f"iter {it+1:>3d} | nodes {mean_nodes:>6.1f} | return {mean_return:>+8.1f} | "
                    f"solved {100*solved_frac:>5.1f}% | pg {update_metrics['pg_loss']:>+6.3f} | "
                    f"v {update_metrics['value_loss']:>6.3f} | H {update_metrics['entropy']:>5.3f} | "
                    f"elapsed {time.time()-start:>5.1f}s"
                )
        return self.stats

    @torch.no_grad()
    def evaluate(self, n_eval: int, deterministic: bool = True) -> dict:
        nodes, objs, completed = [], [], []
        for ep in range(n_eval):
            env = self.env_factory(20_000 + ep)
            env.reset(seed=20_000 + ep)
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
