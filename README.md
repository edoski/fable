# SPICE

SPICE is a temporal fee-timing pipeline for EVM chains. It acquires canonical block corpora, builds timestamp-native feature tables, tunes models, trains artifacts, and runs evaluation-day simulations.

## Stack

- `Typer` for the root CLI
- `Pydantic` + `PyYAML` for explicit config loading
- `SQLAlchemy Core` for SPICE-owned structured state
- `sf-hamilton` for feature/dataflow execution
- `Lightning` + `PyTorch` for training
- `Optuna` for tuning and study persistence
- `web3.py` for RPC access
- `Polars` + `Pandera` for block-table validation and corpus IO

## Setup

```bash
brew install uv
uv sync --extra dev
source .venv/bin/activate
```

`uv` manages the repo-local `.venv/`. If you do not want to activate it, prefix commands with `uv run`.

Provider credentials:

- `direct`: export `ETHEREUM_RPC_URL`, `POLYGON_RPC_URL`, `AVALANCHE_RPC_URL`
- `alchemy`: export `ALCHEMY_API_KEY`

## CLI

Everything runs through one command with explicit flags:

```bash
spice acquire --preset icdcs_2026
spice acquire --preset icdcs_2026 --chain avalanche --provider publicnode
spice train --preset icdcs_2026 --model lstm --feature-set icdcs_2026
spice tune --preset icdcs_2026 --model lstm --feature-set icdcs_2026 --trial-count 20
spice simulate --preset icdcs_2026 --variant baseline
spice config list provider
spice config show dataset icdcs_2026
spice config create chain my_chain --set runtime.chain_id=123 --set runtime.uses_poa_extra_data=false
spice config update provider direct --set chains.my_chain.endpoint.env_var=MY_CHAIN_RPC_URL
spice config delete preset old_preset
spice show dataset
spice show artifact --chain avalanche --dataset icdcs_2026 --model lstm --task icdcs_2026 --variant baseline
spice show study --chain avalanche --dataset icdcs_2026 --model lstm --task icdcs_2026 --study default
spice show study --chain avalanche --dataset icdcs_2026 --model lstm --task icdcs_2026 --study default --detail config
spice delete artifact --chain avalanche --dataset icdcs_2026 --model lstm --task icdcs_2026 --variant baseline
```

Override files stay plain YAML:

```bash
spice train --preset icdcs_2026 --config local/train.yaml
```

## Config

Config loading lives in [src/spice/config](src/spice/config).

Named specs live under [src/spice/conf](src/spice/conf):

- `preset/`: convenience bundles of named selectors
- `dataset/`: evaluation-date selectors
- `chain/`, `provider/`: chain and RPC specs
- `task/`, `execution/`: delay budgets and sampling contracts in real seconds
- `model/`, `feature_set/`: modeling choices
- `training/`, `split/`, `simulation/`, `acquisition/`, `tuning/`, `tuning_space/`: workflow profiles

Core spec authoring goes through `spice config`:

- `spice config list <group>`
- `spice config show <group> <name>`
- `spice config create <group> <name> --set path=value ...`
- `spice config update <group> <name> --set path=value ... --unset path ...`
- `spice config delete <group> <name> [--force]`

Rules:

- Presets are optional. They are not the canonical schema.
- `spice config` writes canonical YAML into `src/spice/conf/<group>/<name>.yaml`.
- `task.lookback_seconds` and `task.max_supported_delay_seconds` are real wall-clock contracts.
- Feature history is derived from the selected feature graph in seconds.
- `acquire` expands raw history until the selected task and feature graph produce enough valid anchor samples.
- `train` and `simulate` validate that the selected feature graph matches the trained artifact.

## Temporal Semantics

Public interfaces stay seconds-native:

- `lookback_seconds`
- `max_supported_delay_seconds`
- `requested_delay_seconds`

Internal semantics are timestamp-native:

- context = blocks with timestamps inside the real lookback window
- valid future candidates = blocks inside the real delay window
- labels = cheapest valid future block under that real deadline

SPICE does not convert seconds into nominal block counts anymore.

## Output Layout

- catalog: `outputs/.spice/catalog.sqlite`
- history corpus: `outputs/corpora/<chain>/<corpus_id>/history/...`
- evaluation corpus: `outputs/corpora/<chain>/<corpus_id>/evaluation/...`
- corpus state: `outputs/corpora/<chain>/<corpus_id>/.spice/state.sqlite`
- tuned study state: `outputs/studies/<chain>/<study_id>/.spice/state.sqlite`
- model artifacts: `outputs/artifacts/<chain>/<artifact_id>/...`
- artifact state: `outputs/artifacts/<chain>/<artifact_id>/.spice/state.sqlite`

`outputs/` is the default root. Override it only when you want isolation somewhere else.

Users query by selectors such as `--dataset`, `--study`, `--model`, `--task`, and `--variant`.
`dataset` is the public selector word. Internally the raw block collection is a `corpus`.
Storage ids are deterministic internal ids. The catalog maps selectors to roots.
Structured state is SQLite-only. SPICE does not persist generated JSON metadata or report files.
Re-running `spice tune` with the same study resumes that study up to the requested total `--trial-count`.

## Current Model Boundary

Canonical internal truth is:

- raw block corpus
- timestamp-native feature table
- ragged timestamp-native samples

Current sequence families share one semantic input representation:

- `lstm`
- `transformer`
- `transformer_lstm`

That shared representation is compiled into a masked/padded batch only at the model boundary.
Future model families can register a different input representation without changing corpus storage, task semantics, or workflow interfaces.

## Verification

```bash
uv run ruff check src tests
uv run pyright
uv run pytest -q
```
