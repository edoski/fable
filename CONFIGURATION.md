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
objective: profit_poisson_replay

acquisition:
  provider: publicnode
training:
  id: default
  split: default
tuning:
  id: default
  space: lstm_fixed_context
evaluation:
  id: poisson_replay
```

The default runnable surface is `current_row_fee_dynamics`. `evaluation.delay_seconds` is usually omitted; evaluation workflows default it from `problem.max_delay_seconds`. Benchmark cases and CLI overrides may supply Workflow Selection fields when a surface leaves variation open; the canonical field vocabulary lives in `config.selections`.

Available evaluator ids are `poisson_replay` and `full_temporal_replay`. The matching evaluation objectives are `profit_poisson_replay` and `profit_full_temporal_replay`. Train and tune workflows require evaluation objectives to match the selected evaluator and metric direction; evaluate workflows may intentionally use a different diagnostic evaluator for an already-trained artifact.

## Workflow Refs

Workflow sections inside a surface point to named refs:

- `training/default.yaml`
- `split/default.yaml`
- `tuning/default.yaml`

Acquisition endpoints and rate controls live on the selected provider, for example `provider/publicnode.yaml` or `provider/tenderly.yaml`.

Add a new named ref only when behavior differs. Small run variation should normally live in benchmark cases or CLI flags.

## CLI Fields

Workflow commands use `--surface`. `--preset` is intentionally unsupported.

Shared train/tune selection options include `--chain`, `--problem`, `--features`, `--objective`, `--evaluation`, `--model`, `--tuning-space`, `--training`, `--split`, `--tuning`, and `--study`. `train` accepts `--variant`; `evaluate` accepts `--artifact-id`, `--dataset-id`, `--evaluation`, `--delay-seconds`, and `--batch-size`. `tune` accepts `--trial-count`. `acquire` accepts `--chain`, `--problem`, `--features`, `--provider`, `--dry-run`, and `--storage-root`.

Resolution order:

1. load the surface
2. build workflow-shaped refs by applying CLI or benchmark case overrides
3. load referenced typed specs
4. validate the final workflow config
5. derive storage identity from the resolved config

Storage identity does not include the surface or benchmark name. It changes when resolved semantic payloads such as model, objective, evaluator config for evaluation objectives, features, problem, training, split, tuning, or tuning space change.

## Benchmarks

`benchmark/*.yaml` defines experiment matrices and workflow DAGs. Each case has a scalar
`base`, standard `dimensions`, and one or more ordered `steps`. Global dimensions apply
to all steps; step dimensions apply only to that step. Local `after` dependencies become
Slurm dependencies during remote submission.

Benchmark specs load as raw config groups, then `benchmarks.schema` validates them before
materialization applies benchmark root policy and calls workflow config resolution.

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
          objective: profit_poisson_replay
          evaluation: poisson_replay
          variant: baseline
      - id: evaluate_baseline
        workflow: evaluate
        after: [train_baseline]
        artifact_from: train_baseline
        set:
          evaluation: poisson_replay
```

Planning validates every row before printing anything:

```bash
spice benchmark plan lookback_window_sweep --target disi_l40
```

Remote submission reads the persisted run directory and submits exactly that plan:

```bash
spice benchmark submit outputs/benchmarks/runs/lookback_window_sweep/<timestamp>
```

Collection pulls required remote artifacts, requires every expected evaluate result to
match its submitted execution provenance, then writes `collection.json` and upserts
`results.sqlite` only after full success:

```bash
spice benchmark collect outputs/benchmarks/runs/lookback_window_sweep/<timestamp>
spice benchmark index export --output benchmarks/exports/figure_3_model_comparison.csv
spice benchmark index rebuild
spice benchmark index list --benchmark lookback_window_sweep
```
