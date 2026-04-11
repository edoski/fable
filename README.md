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
- model artifacts: `artifacts/models/<chain>/<dataset_id>/<family>/<delay>s/...`
- simulation reports: `artifacts/models/<chain>/<dataset_id>/<family>/<delay>s/simulation_report.json`
- tuning outputs: `artifacts/models/<chain>/<dataset_id>/<family>/<delay>s/tuning/...`
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
.venv/bin/dvc repro
```

Hydra defaults live under [src/spice/conf](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/conf). DVC Hydra composition is enabled, and [params.yaml](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/params.yaml) is the composed baseline consumed by the DVC stages.

DVC stages call `spice-dvc`, a thin runner that loads `params.yaml`, pins the requested stage task, and dispatches the same workflow `run(...)` functions used by the direct entrypoints. Direct `spice-train` keeps `tuning.apply_best_params=false`; the DVC `train` stage applies the tuned best params because it depends on the model-local `best_params.json`.

On macOS, DVC stage commands run through `./bin/spice-awake`, which attaches `caffeinate` automatically when it is available so long `dvc repro ...` runs do not idle-sleep the machine mid-stage. Direct `spice-*` entrypoints do not use that wrapper automatically.

You can also run the workflow entrypoints directly:

```bash
.venv/bin/spice-acquire chain=ethereum provider=publicnode
.venv/bin/spice-train chain=ethereum model=lstm training.device=cpu
.venv/bin/spice-simulate chain=ethereum model=lstm training.device=cpu
.venv/bin/spice-tune chain=ethereum model=lstm tuning.trial_count=20
```

The default dataset window is configured explicitly through `dataset.*` in
[params.yaml](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/params.yaml)
and Hydra defaults:

- `dataset.id`
- `dataset.window.start_date`
- `dataset.window.end_date`
- `dataset.temporal.max_delay_seconds`
- `dataset.temporal.lookback_seconds`
- `dataset.sampling.anchor_count`
- `dataset.sampling.history_anchor_count`

`dataset.sampling.anchor_count` is the training/tuning sample count.
`dataset.sampling.history_anchor_count` is optional. When unset, it follows
`anchor_count`, but you can raise it to keep a larger reusable history window
for acquisition.

History acquisition is block-planned, not time-estimated. The acquisition
workflow resolves the evaluation start block, counts backward by the exact
required history block count, and records the actual first fetched block
timestamp in dataset metadata. Provider defaults are unchanged, and local IPC
nodes still fit through the same provider endpoint abstraction.

## Configuration

Hydra config groups live under [src/spice/conf](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/conf):

- `acquisition/`
- `chain/`
- `dataset/`
- `model/`
- `provider/`
- `paths/`
- `runtime/`
- `simulation/`
- `split/`
- `tracking/`
- `training/`
- `tuning/`

Runtime validation happens in [config.py](/Users/edo/Documents/Obsidian/the-vault/university/Thesis/spice/src/spice/core/config.py). Hydra YAML owns defaults; Pydantic models enforce structural invariants, including transformer head divisibility and provider endpoint availability.

## Verification

```bash
.venv/bin/ruff check src/spice tests
.venv/bin/pyright
.venv/bin/pytest -q
```
