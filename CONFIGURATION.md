# Configuration

SPICE config is a flat set of named YAML specs under [src/spice/conf](src/spice/conf).
The resolver uses PyYAML plus Pydantic models; there is no Hydra or OmegaConf layer.

## Surfaces

`surface/*.yaml` is durable benchmark context. A surface names stable specs and explicit workflow-section refs:

```yaml
chain: ethereum
dataset: icdcs_2026
features: core_fee_dynamics
problem: current_row_nominal
dataset_builder: fixed_sequence_temporal
model: lstm
prediction: icdcs_2026
objective: profit_poisson_replay_2h_mean

acquisition:
  provider: publicnode
training:
  id: default
  split: default
tuning:
  id: default
  space: lstm_fixed_context
evaluation:
  id: poisson_replay_2h_mean
```

The default runnable surface is `current_row_fee_dynamics`. `evaluation.delay_seconds` is usually omitted; evaluation workflows default it from `problem.max_delay_seconds`. `model`, `features`, `objective`, `evaluation`, `tuning_space`, `delay_seconds`, `study`, `variant`, and `trial_count` may be supplied by benchmark cases or CLI overrides when a surface leaves variation to the run request.

## Workflow Refs

Workflow sections inside a surface point to named refs:

- `training/default.yaml`
- `split/default.yaml`
- `tuning/default.yaml`

Acquisition endpoints and rate controls live on the selected provider, for example `provider/publicnode.yaml` or `provider/tenderly.yaml`.

Add a new named ref only when behavior differs. Small run variation should normally live in benchmark cases or CLI flags.

## CLI Fields

Workflow commands use `--surface`. `--preset` is intentionally unsupported.

Shared model-workflow selectors include `--chain`, `--problem`, `--features`, `--objective`, `--evaluation`, `--model`, `--tuning-space`, `--training`, `--split`, `--tuning`, and `--study`. `train` and `evaluate` accept `--variant`; `evaluate` accepts `--delay-seconds`; `tune` accepts `--trial-count`. `acquire` accepts `--chain`, `--problem`, `--features`, `--provider`, `--dry-run`, and `--storage-root`.

Resolution order:

1. load the surface
2. load referenced typed specs
3. apply CLI or benchmark case overrides
4. validate the final workflow config
5. derive storage identity from the resolved config

Storage identity does not include the surface or benchmark name. It changes when resolved semantic payloads such as model, objective, features, problem, training, split, tuning, or tuning space change.

## Benchmarks

`benchmark/*.yaml` defines experiment matrices and workflow DAGs. Each case has a scalar
`base`, standard `dimensions`, and one or more ordered `steps`. Global dimensions apply
to all steps; step dimensions apply only to that step. Local `after` dependencies become
Slurm dependencies during remote submission.

```yaml
cases:
  - id: lookback_window_sweep
    base:
      surface: current_row_fee_dynamics
      training: default
      split: default
      study: lookback_window_sweep
    dimensions:
      data:
        - set: {chain: ethereum}
        - set: {chain: polygon}
        - set: {chain: avalanche}
      models:
        - set:
            model: lstm
            tuning_space: lstm_large_capacity
        - set:
            model: transformer
            tuning_space: transformer_large_capacity
        - set:
            model: transformer_lstm
            tuning_space: transformer_lstm_large_capacity
      problems:
        - grid:
            base: current_row_nominal
            fields:
              lookback_seconds: [600, 900, 1200]
    steps:
      - id: train_baseline
        workflow: train
        set:
          objective: profit_poisson_replay_2h_mean
          evaluation: poisson_replay_2h_mean
          variant: baseline
      - id: evaluate_baseline
        workflow: evaluate
        after: [train_baseline]
        set:
          objective: profit_poisson_replay_2h_mean
          evaluation: poisson_replay_2h_mean
          variant: baseline
```

Planning validates every row before printing anything:

```bash
spice benchmark plan lookback_window_sweep
```

Remote submission writes local run state under `outputs/benchmarks/runs/<name>/<timestamp>/`
and uses the configured remote target:

```bash
spice benchmark submit lookback_window_sweep
```

Collection pulls remote studies/artifacts through storage sync and prints JSONL status.
Use `--write` only when all expected evaluation rows are complete; missing rows abort the
ledger write.

```bash
spice benchmark collect lookback_window_sweep
spice benchmark collect lookback_window_sweep --write
```
