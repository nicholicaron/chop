# Learnable Agents

This package ships two learnable node-selection policies plus a REINFORCE trainer that drives either of them.

## Policies

### `NodeSelectionPolicy` ([`policy.py`](policy.py))

A small MLP (default 2x64 with `tanh`) that consumes the env's flat observation
(`Box(K * F_PER_NODE + F_GLOBAL,)`) and emits K logits over the candidate
nodes. Padded slots are masked with `-inf` so the categorical never samples a
fake candidate.

```python
from src.agents.policy import NodeSelectionPolicy
policy = NodeSelectionPolicy(k=16, hidden=64)
action, log_prob, entropy = policy.act(env, deterministic=False)
```

### `GNNNodeSelectionPolicy` ([`gnn_policy.py`](gnn_policy.py))

GCN convolutions over the full B&B enumeration tree (via
`env.graph_observation()`), then an MLP head scores the K candidate
embeddings. Logically equivalent action contract to the MLP — same
`(action, log_prob, entropy)` return — so the trainer is policy-agnostic.

```python
from src.agents.gnn_policy import GNNNodeSelectionPolicy
policy = GNNNodeSelectionPolicy(k=16, hidden=64, n_conv=2)
action, log_prob, entropy = policy.act(env, deterministic=False)
```

## Action contract

Both policies return a `(K,)` numpy `float32` action vector with `+1.0` in
the chosen slot and `-1.0` elsewhere. The env interprets this via
`argmax(scores[:n_real])` so any encoding that puts the highest score on the
intended slot works.

## Training

Two trainers are available; both expect the same `act(env)` policy contract.

### `ReinforceTrainer` ([`reinforce.py`](reinforce.py))

Policy-gradient with EMA baseline. Simplest, lowest-overhead option.

* **Returns**: discounted with `gamma=0.99` (effectively undiscounted on short episodes)
* **Baseline**: EMA of episode returns with `baseline_decay=0.95`, used for variance reduction
* **Entropy bonus**: `entropy_coef * entropy.mean()` added to the loss to keep early exploration alive
* **Gradient clip**: `grad_clip=1.0`
* **Works with**: MLP and GNN policies

```python
from src.agents.reinforce import ReinforceTrainer, TrainConfig
from src.utils.eval import make_env_factory

env_factory = make_env_factory(make_problem, k_nodes=16, max_steps=400)
trainer = ReinforceTrainer(
    policy=policy,
    env_factory=env_factory,
    config=TrainConfig(n_episodes=600, lr=5e-4, entropy_coef=0.01),
)
stats = trainer.train()
eval_results = trainer.evaluate(n_eval=30, deterministic=True)
```

### `PPOTrainer` ([`ppo.py`](ppo.py))

Clipped-objective PPO with GAE(lambda), separate value network, and minibatch
updates. More overhead per iteration but more sample-efficient on longer-horizon
or higher-variance problems.

* **Rollouts**: collects `episodes_per_iter` complete episodes per iteration
* **Advantages**: GAE with `gamma=0.99`, `lambda=0.95`; normalized per batch
* **Update**: `update_epochs` passes through the rollout buffer with `minibatch_size`
* **Loss**: `clipped surrogate + value_coef * MSE - entropy_coef * H`, `clip_eps=0.2`
* **Currently MLP-only** — generalizing to the GNN requires storing the graph
  observation per step and replaying it through the GNN, which is straightforward
  but not yet implemented.

```python
from src.agents.ppo import PPOTrainer, PPOConfig

trainer = PPOTrainer(
    policy=policy,  # NodeSelectionPolicy
    env_factory=env_factory,
    config=PPOConfig(n_iterations=60, episodes_per_iter=10),
)
stats = trainer.train()
```

## Adding a new policy

Subclass `nn.Module` and implement:

```python
def act(self, env, deterministic: bool = False) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Return (action_vector_K, log_prob_scalar, entropy_scalar)."""
```

That's the entire contract — the trainer passes the env in so the policy
can fish out whichever observation flavor it wants (`env._observation()`,
`env.graph_observation()`, custom features).

## Empirical findings

See the top-level [README — Results](../../README.md#results) for the
headline numbers. Short version:

| Setup | Algorithm | Held-out nodes (mean ± std) | vs. BestBound |
|-------|-----------|------------------------------|---------------|
| Knapsack(25, medium) | REINFORCE + MLP | 66.7 ± 52.9 | matches |
| SetCover(50e×80s d=0.1) | REINFORCE + MLP | **11.3 ± 9.7** | **1.68x better** |
| SetCover(50e×80s d=0.1) | PPO + MLP | 16.4 ± 11.5 | 1.16x better |
| SetCover(50e×80s d=0.1) | REINFORCE + GNN | 19.0 ± 15.0 (collapses to BestBound under det. eval) | matches |

The GNN trains end-to-end with REINFORCE but its deterministic eval has so far collapsed to BestBound's behavior on Set Cover, even though stochastic per-episode performance during training is competitive (~10-15 nodes). Likely fixable via more entropy regularization and longer training; see roadmap.

PPO trains faster wall-clock per iteration but on these short-episode problems doesn't beat REINFORCE — its sample-efficiency edge will start to matter on longer episodes (n_items >= 50).
