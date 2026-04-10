# SPICE

Lean temporal-module baseline for SPICE-style fee-timing experiments.

This repository is intentionally scoped to the temporal module only:

- pull raw EVM block data
- enrich missing block fields
- build fixed-horizon temporal datasets
- train baseline sequence models
- run evaluation-day temporal simulations

It does not implement the broader SPICE spatial/oracle/reputation system.

## Stack

- `Typer` + `Rich` for CLI and progress output
- `Pydantic v2` + `pydantic-settings` for config, settings, manifests, and reports
- `Polars` for Parquet IO, validation scans, enrichment staging, and feature prep
- `HTTPX` for JSON-RPC transport
- `scikit-learn` for weighted feature scaling
- `NumPy` + `PyTorch` for dataset math, modeling, training, inference, and simulation

## Package Layout

The installable namespace is `spice`, so the source layout stays `src/spice/...`.
The package itself is now split into four shallow subpackages plus two top-level entrypoints:

- `src/spice/core`: config, settings, constants, and console/reporting primitives
- `src/spice/acquisition`: raw pulls, RPC, enrichment, validation, and provenance
- `src/spice/data`: Parquet IO, feature engineering, dataset geometry, and scaling
- `src/spice/modeling`: models, training, inference, simulation, artifacts, and reports
- `src/spice/api.py`: supported high-level Python API
- `src/spice/cli.py`: supported CLI surface

There are no dual paths for old/new formats. Runtime block datasets are Parquet-only.

## Configuration

Experiment configuration is loaded from YAML through `spice.core.config.ExperimentConfig`.
Environment-backed RPC settings are loaded through `spice.core.settings.RuntimeSettings`.

Supported environment variables:

- `RPC_PROVIDER`
- `ETHEREUM_RPC_URL`
- `POLYGON_RPC_URL`
- `AVALANCHE_RPC_URL`
- `ALCHEMY_API_KEY`

## CLI

The installed command is `spice`.

Examples:

- `spice blocks plan configs/baseline.yaml`
- `spice blocks pull configs/pilots/ethereum-36s.yaml ethereum history --rpc-provider publicnode --no-dry-run`
- `spice blocks enrich configs/pilots/ethereum-36s.yaml ethereum <raw-dir> <enriched-dir>`
- `spice train configs/pilots/ethereum-36s.yaml <history-dir> <artifact-dir> ethereum lstm 36 --device cpu`
- `spice simulate configs/pilots/ethereum-36s.yaml <artifact-dir> <history-dir> <evaluation-dir> --device cpu`

The CLI uses Rich progress for pull, enrich, and train stages and then prints concise stable summary lines at the end.

## Python API

`spice.api` is the only supported Python API surface.

```python
from pathlib import Path

from spice.api import (
    load_artifact,
    load_config,
    run_simulation_workflow,
    run_training_workflow,
)

config = load_config(Path("configs/pilots/ethereum-36s.yaml"))

train_report = run_training_workflow(
    config,
    Path("artifacts/pilots/ethereum-36s/enriched/ethereum/history"),
    Path("artifacts/pilots/ethereum-36s/runs/ethereum/lstm-36s"),
    "ethereum",
    "lstm",
    36,
    device="cpu",
)

artifact = load_artifact(Path("artifacts/pilots/ethereum-36s/runs/ethereum/lstm-36s"))

simulation_report = run_simulation_workflow(
    config,
    Path("artifacts/pilots/ethereum-36s/runs/ethereum/lstm-36s"),
    Path("artifacts/pilots/ethereum-36s/enriched/ethereum/history"),
    Path("artifacts/pilots/ethereum-36s/enriched/ethereum/evaluation"),
    device="cpu",
)
```

## Artifacts

Training writes:

- `artifact.json`
- `model.pt`
- `train_report.json`
- `simulation_report.json`

Dataset provenance is stored under `.spice/source.json` inside dataset directories.

## Verification

Run all checks inside the project virtual environment:

- `.venv/bin/ruff check src/spice tests`
- `.venv/bin/pyright src/spice tests`
- `.venv/bin/pytest -q`
