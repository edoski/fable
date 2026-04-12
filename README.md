# SPICE

Temporal-module baseline for SPICE-style fee-timing experiments.

This repository is intentionally scoped to the temporal module only:

- acquire canonical EVM block datasets
- build fixed-horizon temporal datasets
- train baseline sequence models
- run evaluation-day temporal simulations
- tune model hyperparameters

It does not implement the broader SPICE spatial/oracle/reputation system.

## Stack

- `Hydra` for runtime configuration and composition
- `DVC` for reproducible stages, artifact tracking, and future remote execution
- `MLflow` for run tracking, params, metrics, and artifacts
- `Lightning` for training orchestration
- `Optuna` for hyperparameter optimization
- `web3.py` for HTTP and IPC RPC transport
- `Pandera` + `Polars` for dataset validation and parquet/table work
- `scikit-learn` for scaling
- `NumPy` + `PyTorch` for dataset math, modeling, inference, and simulation

There is no legacy compatibility layer. The repository does not expose `spice.api`,
the old `spice` Typer CLI, snapshot registries, or the old custom config loader.

## Layout

```text
src/spice/
  acquisition/
  conf/
  core/
  data/
  modeling/
  workflows/
tests/
dvc.yaml
params.yaml
```

Consolidated runtime boundaries:

- acquisition window planning, canonical block-field extraction, dataset validation, and metadata shaping live under `src/spice/acquisition/`
- workflow lifecycle and small shared workflow helpers live in `src/spice/workflows/_shared.py`
- persisted training execution is centralized in `src/spice/modeling/execution.py`

Key runtime paths:

- history datasets: `artifacts/datasets/<chain>/<dataset_id>/history/...`
- evaluation datasets: `artifacts/datasets/<chain>/<dataset_id>/evaluation/...`
- dataset metadata: `artifacts/datasets/<chain>/<dataset_id>/.spice/metadata.json`
- model artifacts: `artifacts/models/<chain>/<dataset_id>/<family>/<delay>s/<variant>/<study_id>/...`
- tuning outputs: `artifacts/models/<chain>/<dataset_id>/<family>/<delay>s/tuned/<study_id>/tuning/...`
- MLflow store: `.mlflow/`

## Setup

```bash
.venv/bin/pip install -e .
```

If you use the `direct` provider, export chain RPC URLs:

- `ETHEREUM_RPC_URL`
- `POLYGON_RPC_URL`
- `AVALANCHE_RPC_URL`

If you use the `alchemy` provider, export:

- `ALCHEMY_API_KEY`

## Running

Use DVC as the primary surface:

```bash
.venv/bin/dvc repro acquire
.venv/bin/dvc repro tune
.venv/bin/dvc repro train
.venv/bin/dvc repro simulate
.venv/bin/dvc exp run --no-hydra -S artifact.variant=tuned -S study.id=fee-sweep-a train
.venv/bin/dvc repro
```

[params.yaml](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/params.yaml) is the thin public experiment spec. Both direct entrypoints and DVC load the same file, pin the requested task, apply explicit CLI overrides, and validate the result through the same typed config layer. `train` and `simulate` select artifacts through `artifact.variant=baseline|tuned`. Tuned lineage selection uses `study.id`.

On macOS, DVC stage commands run through `./bin/spice-awake`, which attaches `caffeinate` automatically when it is available so long `dvc repro ...` runs do not idle-sleep the machine mid-stage. Direct `spice-*` entrypoints do not use that wrapper automatically.

You can also run the workflow entrypoints directly:

```bash
.venv/bin/spice-acquire presets.chain=ethereum presets.provider=publicnode
.venv/bin/spice-train presets.model=lstm training.device=cpu
.venv/bin/spice-train artifact.variant=tuned study.id=fee-sweep-a
.venv/bin/spice-simulate presets.model=lstm training.device=cpu
.venv/bin/spice-tune presets.model=lstm tuning.trial_count=20
```

The public experiment surface in
[params.yaml](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/params.yaml)
is intentionally thin:

- `presets.chain`
- `presets.provider`
- `presets.model`
- `dataset.id`
- `dataset.span.start_date`
- `dataset.span.end_date`
- `evaluation.duration_days`
- `dataset.temporal.max_delay_seconds`
- `dataset.temporal.lookback_seconds`
- `dataset.sampling.sample_count`
- `acquisition.history_sample_budget` optional

`dataset.sampling.sample_count` is the training and tuning sample count.
`acquisition.history_sample_budget` is an optional acquisition-only override.
When it is omitted, acquisition uses the same value as `sample_count`.
`model:` contains only overrides for the selected `presets.model` family.

Artifact provenance is configured through:

- `artifact.variant`
- `study.id`

History acquisition is block-planned, not time-estimated. The acquisition
workflow resolves the evaluation start block, counts backward by the exact
required history block count, and records the actual first fetched block
timestamp in dataset metadata. Provider defaults are unchanged, and local IPC
nodes still fit through the same provider endpoint abstraction.

## Configuration

Hydra preset fragments and runtime defaults live under [src/spice/conf](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/conf):

- `chain/`
- `model/`
- `provider/`
- `rpc_profile/`

`params.yaml` stores only the selectors and first-order experiment values. Hydra preset files remain the source of truth for resolved chain/provider details, runtime defaults, RPC tuning, and family-specific model defaults. Use selector overrides such as `presets.chain=polygon`, `presets.provider=direct`, or `presets.model=transformer`. Runtime validation happens in [config.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/core/config.py). Pydantic models enforce structural invariants, including transformer head divisibility, family-specific model fields, closed tuning search-space fields, and acquire-only provider endpoint availability.

The tuning contract is explicit and closed. `tuning.search_space` is nested by subsystem, not dotted-path keyed:

```yaml
tuning:
  direction: maximize
  objective_metric: validation_profit_over_baseline
  search_space:
    training:
      learning_rate: [0.0001, 0.0003, 0.001]
      weight_decay: [0.0, 0.01, 0.05]
    model:
      hidden_size: [64, 128, 256]
      dropout: [0.0, 0.1, 0.2]
```

Supported objective metrics are:

- `validation_loss`
- `validation_accuracy`
- `validation_cost_over_optimum`
- `validation_profit_over_baseline`

Tuning outputs are structured JSON artifacts under `paths.tuning_root`:

- `study.json`: typed study summary including the nested search space and best-trial payload
- `trials.json`: typed per-trial records
- `best_params.json`: nested `params.training` / `params.model` payload consumed by `train` when `artifact.variant=tuned`

## Verification

```bash
.venv/bin/ruff check src/spice tests
.venv/bin/pyright
.venv/bin/pytest -q
```
