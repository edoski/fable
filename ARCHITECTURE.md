# Architecture Guide

## Overview

Runtime commands:

1. `spice acquire`
2. `spice tune`
3. `spice train`
4. `spice simulate`

One CLI. One config system. One compiler boundary for temporal semantics.
One feature-family boundary for feature semantics.
One prediction-family boundary for downstream semantics.

## Core Model

SPICE has one architectural hierarchy:

1. canonical domain truth
2. model-input compilation
3. prediction semantics
4. model family

Canonical domain truth is:

- raw block corpus
- resolved feature table
- compiled problem store

Model-input compilation is a separate layer:

- current sequence families use the shared `sequence_inputs` representation
- future families may register a different representation if they need genuinely different input semantics

Prediction semantics is another separate layer:

- one active `prediction`
- one compiled prediction family contract
- one family-owned target contract
- one family-owned output contract
- one family-owned loss, metrics, decode, and simulation bundle
- one primary validation and tuning metric selected by that family

Model family stays below that boundary:

- `lstm`
- `transformer`
- `transformer_lstm`

This keeps domain semantics stable while allowing future model growth.
It also keeps prediction changes orthogonal to corpus storage and model-input representation.

## Config Flow

Config loading lives in [src/spice/config](src/spice/config).

Flow:

1. The loader reads named specs from [src/spice/conf](src/spice/conf).
2. `spice config edit` authors repo-local YAML specs directly under that tree.
3. `--preset` optionally selects a bundle of named defaults.
4. Explicit CLI selector and runtime flags override preset values.
5. Pydantic validates the final request model.
6. `PathLayout` derives deterministic storage ids and roots from `storage.root`.

Selector rules:

- `dataset.name` and `study.name` are human selectors.
- `corpus_id`, `study_id`, and `artifact_id` are deterministic storage ids.
- Runtime commands work from selectors. Users do not need paths.
- Reusing a study selector resumes the same stored study definition. Drift is rejected.

Public temporal contract:

- `dataset.evaluation_date` defines the fixed one-day UTC evaluation window.
- `problem.lookback_seconds` defines real context span.
- `problem.sample_count` defines training and tuning anchor count.
- `problem.max_delay_seconds` defines artifact capability.
- `problem.compiler.id` selects the temporal compiler.
- `delay_seconds` defines the runtime deadline inside that capability.

No config field encodes nominal block time.
`prediction.id` selects the named prediction config.
`prediction.family.id` selects the registered prediction family.

## Temporal Semantics

SPICE is seconds-native outside and compiler-driven inside.

Current compiler boundary:

- `timestamp_native`: context and candidates come from real timestamp windows
- `estimated_block`: seconds are lowered into corpus-calibrated row geometry

Both compilers lower into the same `CompiledProblemStore`:

- `anchor_rows`
- `context_start_rows`
- `candidate_end_rows`
- `max_candidate_slots`

Acquisition, training, inference, and simulation consume compiled contracts and compiled stores only. They do not branch on compiler id.

## Prediction Semantics

Prediction execution lives in [src/spice/prediction](src/spice/prediction).

Current shipped families:

- `candidate_offset_selection`
  - one-head candidate-offset selection
  - primary metric: `profit_over_baseline`
- `min_block_fee_multitask`
  - min-block classification plus min-fee regression
  - primary metric: `total_loss`

Family-owned responsibilities:

- target realization
- output head specification
- training loss
- epoch metrics
- best-epoch selection
- optimization value for tuning
- decode
- simulate

Simulation metrics stay economic and family-neutral:

- `profit_over_baseline`
- `cost_over_optimum`
- `baseline_cost_over_optimum`

## Feature Architecture

Feature execution lives in [src/spice/features](src/spice/features).

Rules:

- each feature family is a registered semantic system
- each feature is a small Hamilton node inside one family
- feature selection is config-driven
- feature formulas stay in Python
- prerequisites are derived from the selected graph as `history_seconds` and `warmup_rows`
- artifacts persist `feature_set_id`, `feature_family_id`, ordered `feature_names`, and `feature_graph_fingerprint`
- inference rebuilds the same graph and fails on mismatch

Current feature families:

- `block_native`
  - `elapsed_blocks`
  - rolling statistics over `10`, `50`, `200` blocks
  - `trend_slope_200`
  - wall-clock cyclical features
- `time_native`
  - `seconds_since_previous_block`
  - `elapsed_seconds`
  - rolling statistics over `60s`, `300s`, `600s`
  - `trend_slope_600s`
  - wall-clock cyclical features

Feature families do not select compilers. Compilers consume only the generic resolved feature table and prerequisites.

## Corpus and Samples

Raw block storage is a corpus. Public CLI still uses the selector word `dataset`.

Derived learning data is not stored as fixed block windows. Instead SPICE builds ragged samples:

- `anchor_row`
- `context_start_row`
- `candidate_end_row`

Padding is not domain truth. Padding exists only in the collate path for model execution.

## Model Boundary

Current sequence families share one semantic input representation because they solve the same sequence problem.

Shared representation semantics:

- `inputs`
- `input_mask`

Important distinction:

- `input_mask` is batch transport logic
- prediction-family targets are compiled separately after representation preparation
- optimum index, baseline fee, and realized fee are derived by the active prediction family

The compiler seam is keyed by input representation semantics, not model family name.

Current mapping:

- `lstm` -> `sequence_inputs`
- `transformer` -> `sequence_inputs`
- `transformer_lstm` -> `sequence_inputs`

Future examples:

- `time_grid`
- `graph`
- `point_process`

## CLI Shape

SPICE CLI commands fall into three categories:

1. workflow commands
2. config query and edit commands
3. state query and deletion commands

Workflow commands:

- `acquire`
- `tune`
- `train`
- `simulate`

These follow the full config-loading path and then delegate to workflow modules.

Query and deletion commands:

- `show dataset|study|artifact`
- `delete dataset|study|artifact`

These are selector-driven commands over existing state. They do not use workflow request models. `show` resolves catalog matches, loads typed state summaries, builds typed root descriptions, and renders console sections.

## Package Roles

### `config`

- [models.py](src/spice/config/models.py): typed specs, workflow request models, path layout, provider resolution
- [loader.py](src/spice/config/loader.py): preset + selector resolution into validated workflow configs
- [registry.py](src/spice/config/registry.py): config group registry, canonical YAML serialization, query/edit seeding

### `core`

- [console.py](src/spice/core/console.py): workflow reporting
- [files.py](src/spice/core/files.py): atomic file and directory promotion helpers
- [errors.py](src/spice/core/errors.py): operator-facing CLI and workflow error categories

### `storage`

- `engine.py`: SQLAlchemy engine creation, SQLite PRAGMAs, `RootKind`, root-kind bootstrap
- `schema.py`: SPICE-owned Core table definitions
- `catalog.py`: global selector-to-root catalog
- `corpus.py`: dataset manifest + acquire-run persistence
- `artifact.py`: manifest, training, and simulation table I/O
- `study_models.py`: study manifests, summaries, and trial DTOs
- `study_manifest.py`: study manifest creation, persistence, and validation
- `study_optuna.py`: Optuna-backed study access and tuned-param loading
- `study_render.py`: study summary rendering
- `inspect.py`: tiny selector-based root dispatcher for `spice show`
- `inspect_dataset.py`, `inspect_artifact.py`, `inspect_study.py`: root-specific typed descriptions and sections

### `acquisition`

- [rpc.py](src/spice/acquisition/rpc.py): exact timestamp window resolution, RPC pulling, adaptive batching

### `features`

- [engine.py](src/spice/features/engine.py): family-aware Hamilton driver, resolved feature tables, prerequisite derivation, fingerprinting
- `families/`: self-registering feature family specs and Hamilton modules

### `corpus`

- `contract.py`: canonical block schema
- `io.py`: parquet corpus discovery and loading
- `validation.py`: corpus validation
- `metadata.py`: typed corpus summary builders

### `temporal`

- `contracts.py`: compiled problem contracts shared across workflows
- `problem_store.py`: shared compiled problem-store IR and split helpers
- `compilers/`: self-registering temporal compiler specs

### `modeling`

- [representations.py](src/spice/modeling/representations.py): input-representation registry
- [pipeline.py](src/spice/modeling/pipeline.py): training and inference dataset preparation
- [models.py](src/spice/modeling/models.py): baseline temporal models
- [training.py](src/spice/modeling/training.py): trainer execution and metrics
- [persisted_training.py](src/spice/modeling/persisted_training.py): persisted training flow
- [artifacts.py](src/spice/modeling/artifacts.py): model + manifest persistence and feature validation
- [results.py](src/spice/modeling/results.py): typed training and simulation summary envelopes
- [result_codecs.py](src/spice/modeling/result_codecs.py): typed runtime-to-storage codecs
- [simulation.py](src/spice/modeling/simulation.py): Poisson-arrival simulation over evaluation examples

### `workflows`

- [acquire.py](src/spice/workflows/acquire.py): acquisition orchestration
- [tune.py](src/spice/workflows/tune.py): Optuna orchestration with `RDBStorage`
- [train.py](src/spice/workflows/train.py): artifact-producing training orchestration
- [simulate.py](src/spice/workflows/simulate.py): evaluation-day simulation orchestration
- [_shared.py](src/spice/workflows/_shared.py): shared runtime helpers

## Storage Layout

Corpora:

- `outputs/corpora/<chain>/<corpus_id>/history/...`
- `outputs/corpora/<chain>/<corpus_id>/evaluation/...`
- `outputs/corpora/<chain>/<corpus_id>/.spice/state.sqlite`

Artifacts:

- `outputs/artifacts/<chain>/<artifact_id>/model.pt`
- `outputs/artifacts/<chain>/<artifact_id>/.spice/state.sqlite`

Studies:

- `outputs/studies/<chain>/<study_id>/.spice/state.sqlite`

Notes:

- `outputs/.spice/catalog.sqlite` is the global lookup index
- `src/spice/conf` is the saved spec registry
- SPICE-owned structured state lives only in `.spice/state.sqlite`
- root identity is typed internally as `RootKind` with values `corpus`, `study`, `artifact`
- study roots persist a typed study manifest plus Optuna state
- studies and artifacts are separate roots
- `spice config list|show|edit` is the human-facing config path
- `spice show dataset|study|artifact` is the human-facing inspection path
- `spice delete dataset|study|artifact` is the cleanup path

Passive state naming stays strict:

- `Snapshot`: captured semantic or runtime state
- `Record`: persisted envelope or read-model row
- `Summary`: derived user-facing read model
