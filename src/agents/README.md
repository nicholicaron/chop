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

`ReinforceTrainer` ([`reinforce.py`](reinforce.py)) is the policy-agnostic
trainer. It calls `policy.act(env)` each step, accumulates log-probs / rewards
across the episode, and applies REINFORCE-with-baseline:

* **Returns**: discounted with `gamma=0.99` (effectively undiscounted on short episodes)
* **Baseline**: EMA of episode returns with `baseline_decay=0.95`, used for variance reduction
* **Entropy bonus**: `entropy_coef * entropy.mean()` added to the loss to keep early exploration alive
* **Gradient clip**: `grad_clip=1.0`

```python
from src.agents.reinforce import ReinforceTrainer, TrainConfig
from src.utils.eval import make_env_factory

def make_problem(rng):
    return ...  # build a problem instance

env_factory = make_env_factory(make_problem, k_nodes=16, max_steps=400)
trainer = ReinforceTrainer(
    policy=policy,
    env_factory=env_factory,
    config=TrainConfig(n_episodes=600, lr=5e-4, entropy_coef=0.01),
)
stats = trainer.train()
eval_results = trainer.evaluate(n_eval=30, deterministic=True)
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

* **MLP on Knapsack(25, medium)** — recovers BestBound from scratch in ~50 s, generalizes to n_items 15-30
* **MLP on SetCover(50e×80s d=0.1)** — beats BestBound by ~40% (11.3 vs 19.0 nodes)
* **GNN on Knapsack** — trains end-to-end; deterministic eval matches BestBound on small instances. Hasn't decisively beaten the MLP yet — see `--policy gnn` flag in `examples/train_*.py` to experiment.
