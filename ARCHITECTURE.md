# Architecture Guide

## Overview

Runtime commands:

1. `spice acquire`
2. `spice tune`
3. `spice train`
4. `spice simulate`

One CLI. One config system. One canonical temporal semantics model.

## Core Model

SPICE has one architectural hierarchy:

1. canonical domain truth
2. model-input compilation
3. model family

Canonical domain truth is:

- raw block corpus
- timestamp-native feature table
- ragged timestamp-native anchor samples

Model-input compilation is a separate layer:

- current sequence families use the shared `sequence_event` representation
- future families may register a different representation if they need genuinely different input semantics

Model family stays below that boundary:

- `lstm`
- `transformer`
- `transformer_lstm`

This keeps domain semantics stable while allowing future model growth.

## Config Flow

Config loading lives in [src/spice/config](src/spice/config).

Flow:

1. The loader reads named specs from [src/spice/conf](src/spice/conf).
2. `spice config` authors repo-local YAML specs directly under that tree.
3. `--preset` optionally selects a bundle of named defaults.
4. `--config PATH` overlays plain YAML on top of that preset.
5. Explicit CLI flags override both preset and file values.
6. Pydantic validates the final request model.
7. `PathLayout` derives deterministic storage ids and roots from `storage.root`.

Selector rules:

- `dataset.name` and `study.name` are human selectors.
- `corpus_id`, `study_id`, and `artifact_id` are deterministic storage ids.
- Runtime commands work from selectors. Users do not need paths.
- Reusing a study selector resumes the same stored study definition. Drift is rejected.

Public temporal contract:

- `dataset.evaluation_date` defines the fixed one-day UTC evaluation window.
- `task.lookback_seconds` defines real context span.
- `task.sample_count` defines training and tuning anchor count.
- `task.max_supported_delay_seconds` defines artifact capability.
- `execution.requested_delay_seconds` defines the runtime deadline inside that capability.

No config field encodes nominal block time.

## Temporal Semantics

SPICE is seconds-native outside and timestamp-native inside.

For anchor block `i` at timestamp `t_i`:

- context = blocks in `[t_i - lookback_seconds, t_i]`
- valid future candidates = blocks in `(t_i, t_i + delay_seconds]`
- label = cheapest valid future block inside that real deadline
- baseline = next block

This semantics is shared by acquisition sufficiency checks, training, inference, and simulation.

## Feature Architecture

Feature execution lives in [src/spice/features](src/spice/features).

Rules:

- each feature is a small Hamilton node
- feature selection is config-driven
- feature formulas stay in Python
- feature history is derived from the selected graph in seconds
- artifacts persist `feature_set_id`, ordered `feature_names`, and `feature_graph_fingerprint`
- inference rebuilds the same graph and fails on mismatch

Current feature family is time-native:

- event-time deltas such as `seconds_since_previous_block`
- elapsed time such as `elapsed_seconds`
- rolling statistics over `60s`, `300s`, `600s`
- trend windows over real time
- wall-clock cyclical features

## Corpus and Samples

Raw block storage is a corpus. Public CLI still uses the selector word `dataset`.

Derived learning data is not stored as fixed block windows. Instead SPICE builds ragged samples:

- `anchor_row`
- `context_start_row`
- `candidate_end_row`
- `class_label`

Padding is not domain truth. Padding exists only in the collate path for model execution.

## Model Boundary

Current sequence families share one semantic batch shape because they solve the same sequence-event task.

Shared batch semantics:

- `inputs`
- `input_mask`
- candidate fee tensor
- `candidate_mask`
- labels, fee targets, baselines

Important distinction:

- `input_mask` is batch transport logic
- `candidate_mask` is task semantics because valid future actions truly vary by sample

The compiler seam is keyed by input representation semantics, not model family name.

Current mapping:

- `lstm` -> `sequence_event`
- `transformer` -> `sequence_event`
- `transformer_lstm` -> `sequence_event`

Future examples:

- `time_grid`
- `graph`
- `point_process`

## Package Roles

### `config`

- [models.py](src/spice/config/models.py): typed specs, workflow request models, path layout, provider resolution
- [loader.py](src/spice/config/loader.py): named YAML loading, fixed-order merges, CLI/file override composition
- `registry.py`: config group registry, canonical YAML serialization, create/update/delete helpers

### `core`

- [console.py](src/spice/core/console.py): workflow reporting
- [files.py](src/spice/core/files.py): atomic file and directory promotion helpers

### `state`

- `engine.py`: SQLAlchemy engine creation, SQLite PRAGMAs, root-kind bootstrap
- `schema.py`: SPICE-owned Core table definitions
- `catalog.py`: global selector-to-root catalog
- `dataset.py`: corpus summary + acquire-run persistence
- `artifact.py`: manifest, training, and simulation persistence
- `study.py`: Optuna-backed study helpers and tuned-param loading
- `show.py`: selector-resolved inspection helpers

### `acquisition`

- [rpc.py](src/spice/acquisition/rpc.py): exact timestamp window resolution, RPC pulling, adaptive batching
- [datasets.py](src/spice/acquisition/datasets.py): history and evaluation corpus reuse
- [metadata.py](src/spice/acquisition/metadata.py): typed corpus summary builders

### `features`

- [engine.py](src/spice/features/engine.py): Hamilton driver, feature selection, history-seconds derivation, fingerprinting
- [base.py](src/spice/features/base.py): base event-time features
- [rolling.py](src/spice/features/rolling.py): time-window statistics
- [trend.py](src/spice/features/trend.py): time-window trend features

### `data`

- [block_contract.py](src/spice/data/block_contract.py): canonical block schema
- [io.py](src/spice/data/io.py): parquet corpus discovery and loading
- [datasets.py](src/spice/data/datasets.py): timestamp-native sample store and split helpers
- [normalization.py](src/spice/data/normalization.py): ragged-span scaler fitting and application
- [validation.py](src/spice/data/validation.py): corpus validation

### `planning`

- [geometry.py](src/spice/planning/geometry.py): seconds-based lookback and delay windows
- [contracts.py](src/spice/planning/contracts.py): resolved task contracts shared across workflows

### `modeling`

- [representations.py](src/spice/modeling/representations.py): input-representation registry
- [torch_datasets.py](src/spice/modeling/torch_datasets.py): shared `sequence_event` batch compiler
- [pipeline.py](src/spice/modeling/pipeline.py): training and inference dataset preparation
- [models.py](src/spice/modeling/models.py): baseline temporal models
- [training.py](src/spice/modeling/training.py): trainer execution and metrics
- [execution.py](src/spice/modeling/execution.py): persisted training flow
- [artifacts.py](src/spice/modeling/artifacts.py): model + manifest persistence and feature validation
- [reporting.py](src/spice/modeling/reporting.py): internal summary objects
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
- study roots persist a typed study manifest plus Optuna state
- studies and artifacts are separate roots
- `spice config ...` is the human-facing config authoring path
- `spice show dataset|study|artifact` is the human-facing inspection path
- `spice delete dataset|study|artifact` is the cleanup path
