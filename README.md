# SPICE

SPICE is a temporal fee-timing pipeline for EVM chains. It acquires canonical block corpora, builds resolved feature tables through explicit feature families, tunes models, trains artifacts, and evaluates stored artifacts under real delay budgets.

## Stack

- `Typer` for the CLI
- `Pydantic` + `PyYAML` for config loading
- `SQLAlchemy Core` for SPICE-owned state
- `Polars` for corpus IO and validation
- `PyTorch` for modeling
- `Optuna` for tuning
- `web3.py` for RPC access

## Setup

```bash
brew install uv
uv sync --extra dev
source .venv/bin/activate
```

`uv` manages the repo-local `.venv/`. If you do not want to activate it, prefix commands with `uv run`.

## CLI

Local workflow commands:

```bash
spice acquire --preset icdcs_2026
spice train --preset icdcs_2026 --model lstm --feature-set icdcs_2026
spice tune --preset icdcs_2026 --model lstm --feature-set icdcs_2026 --trial-count 20
spice evaluate --preset icdcs_2026 --variant baseline
```

Remote workflow and query commands:

```bash
spice remote train --preset icdcs_2026 --execution default
spice remote tune --preset icdcs_2026 --trial-count 20
spice remote evaluate --preset icdcs_2026 --variant baseline
spice remote config list provider
spice remote show artifact --chain avalanche --dataset icdcs_2026 --model lstm --problem icdcs_2026 --variant baseline
spice remote refresh catalog
```

Local config, inspection, transfer, and deletion commands:

```bash
spice config list provider
spice config show dataset icdcs_2026
spice config edit problem icdcs_2026
spice show dataset
spice show artifact --chain avalanche --dataset icdcs_2026 --model lstm --problem icdcs_2026 --variant baseline
spice delete artifact --chain avalanche --dataset icdcs_2026 --model lstm --problem icdcs_2026 --variant baseline
spice push dataset --chain avalanche --dataset icdcs_2026
spice pull artifact --chain avalanche --dataset icdcs_2026 --model lstm --problem icdcs_2026 --variant baseline
spice refresh catalog
```

## Config

Config loading lives in [src/spice/config](src/spice/config).

Named specs live under [src/spice/conf](src/spice/conf):

- `preset/`: workflow bundles
- `dataset/`: evaluation-date selectors
- `chain/`, `provider/`, `execution/`: runtime and remote execution specs
- `problem/`: delay budgets and sampling contracts
- `model/`, `feature_set/`, `prediction/`: core modeling seams
- `dataset_builder/`: dataset preparation seam
- `tuning_space/`: tuning search spaces

Rules:

- presets are the main workflow entrypoint
- explicit CLI selectors override preset selections
- `problem.compiler.id` selects the temporal compiler
- `feature_set.family.id` selects the feature family
- `prediction.family.id` selects the prediction family
- `dataset_builder.id` selects the dataset preparation path
- `acquire` expands history until the selected problem and feature graph yield enough valid samples
- `train` and `evaluate` validate that the selected feature graph matches the stored artifact

## Core Seams

Strong domain seams stay separate:

- feature family
- temporal compiler
- prediction family
- evaluator
- model family
- representation

Current shipped ids:

- feature families: `block_native`, `time_native`
- compilers: `timestamp_native`, `estimated_block`
- prediction families: `candidate_offset_selection`, `min_block_fee_multitask`
- evaluators: `poisson_replay`, `paper_fullset`
- input normalization: `row_standard`, `window_weighted_standard`
- representation: `sequence_inputs`

## Output Layout

- catalog: `outputs/.spice/catalog.sqlite`
- history corpus: `outputs/corpora/<chain>/<corpus_id>/history/...`
- evaluation corpus: `outputs/corpora/<chain>/<corpus_id>/evaluation/...`
- corpus state: `outputs/corpora/<chain>/<corpus_id>/.spice/state.sqlite`
- tuned study state: `outputs/studies/<chain>/<study_id>/.spice/state.sqlite`
- model artifacts: `outputs/artifacts/<chain>/<artifact_id>/...`
- artifact state: `outputs/artifacts/<chain>/<artifact_id>/.spice/state.sqlite`

Users query by selectors such as `--dataset`, `--study`, `--model`, `--problem`, and `--variant`.

## Verification

```bash
uv run ruff check src tests
uv run pyright
uv run pytest -q
```
