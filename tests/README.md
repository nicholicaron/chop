# Tests

End-to-end smoke tests that exercise the full CHOP stack — simplex → B&B → RL env → policy training → eval. Designed to catch the kind of broken-import / silent-no-op regressions that previously made the project unrunnable.

## Running

```sh
pytest tests/                  # all tests, ~5 s
pytest tests/test_smoke.py -v  # verbose
pytest tests/test_smoke.py::test_rl_env_action_actually_changes_search  # single test
```

## What's covered

| Test | What it verifies |
|------|------------------|
| `test_simplex_solves_2d_lp` | Simplex backend imports correctly and finds the optimum on a 2D LP. |
| `test_branch_and_bound_solves_knapsack` | B&B + simplex + Knapsack `to_ilp()` produce a valid integer-feasible solution with the correct objective. |
| `test_rl_env_action_actually_changes_search` | Different heuristic agents produce **different** node counts. This is the test that would have caught the original "action is a no-op" bug. |
| `test_reinforce_runs_and_decreases_loss_baseline` | The MLP policy + REINFORCE trainer complete a 10-episode training run without erroring; gradients flow (loss varies). |
| `test_policy_eval_smoke` | Trained MLP policy can be evaluated deterministically and reaches optimum. |
| `test_gnn_policy_runs_episode` | GNN policy survives a full episode without NaN'ing the masked softmax. |
| `test_gnn_trains_with_reinforce` | Same trainer drives both MLP and GNN through `act(env)` interface. |
| `test_ppo_trains_mlp_policy` | PPO collects rollouts, computes GAE, and applies clipped-surrogate updates without erroring. |

All tests run on CPU in under 5 seconds.
