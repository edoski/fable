# SPICE

SPICE is a temporal fee-timing research pipeline for EVM chains. It acquires canonical block data, builds feature tables, constructs temporal decision problems, trains neural models, tunes studies, stores artifacts, and evaluates fee-timing decisions under real delay budgets.

The project is organized around explicit seams: configs resolve into typed contracts, contracts compile concrete implementations, workflows orchestrate storage effects, and docs are split into generic architecture and concrete implementation guides.

## Start Here

Read in this order if you are new to the codebase and the domain:

```text
README.md
  -> ARCHITECTURE.md
  -> src/spice/ARCHITECTURE.md
  -> src/spice/conf/IMPLEMENTATIONS.md
  -> package ARCHITECTURE.md files
  -> package IMPLEMENTATIONS.md files
```

Use this rule:

| File | Purpose |
| --- | --- |
| `ARCHITECTURE.md` | Generic structure, boundaries, data flow, dependency direction. |
| `IMPLEMENTATIONS.md` | Current concrete engines, families, algorithms, YAML Surfaces/Config Groups, math, and failure modes. |

Architecture docs explain the shape of the system. Implementation docs explain what the current code actually runs.

## Learning Paths

### Beginner ML And Modeling Path

This path explains how raw block rows become model predictions:

```text
features
  -> temporal problems
  -> dataset builders
  -> model families
  -> prediction families
  -> objectives
  -> evaluation
```

Read:

1. [Feature implementations](src/spice/features/ARCHITECTURE.md)
2. [Temporal compiler implementations](src/spice/temporal/compilers/IMPLEMENTATIONS.md)
3. [Execution policy implementation](src/spice/temporal/execution_policy/IMPLEMENTATIONS.md)
4. [Input normalization implementations](src/spice/temporal/input_normalization/IMPLEMENTATIONS.md)
5. [Dataset builder implementations](src/spice/modeling/dataset_builders/IMPLEMENTATIONS.md)
6. [Model family implementations](src/spice/modeling/families/IMPLEMENTATIONS.md)
7. [Prediction family implementations](src/spice/prediction/families/IMPLEMENTATIONS.md)
8. [Objective implementations](src/spice/objectives/IMPLEMENTATIONS.md)
9. [Evaluator implementations](src/spice/evaluation/IMPLEMENTATIONS.md)

The key mental model:

```text
canonical block rows
  -> feature matrix
  -> temporal problem store
  -> sequence batches
  -> neural model
  -> decoded offsets
  -> evaluator metrics
```

### Data Acquisition And Storage Path

This path explains how chain data is downloaded, validated, committed, indexed, and transferred:

```text
RPC acquisition
  -> Corpus Assembly
  -> corpus validation
  -> storage roots
  -> catalog
  -> sync
```

Read:

1. [RPC acquisition implementations](src/spice/acquisition/rpc/IMPLEMENTATIONS.md)
2. [Corpus implementations](src/spice/corpus/IMPLEMENTATIONS.md)
3. [Storage implementations](src/spice/storage/IMPLEMENTATIONS.md)
4. [Catalog implementations](src/spice/storage/catalog/IMPLEMENTATIONS.md)

The key mental model:

```text
timestamp window
  -> block range
  -> RPC batches
  -> canonical parquet rows
  -> root state DB
  -> derived catalog row
```

### Workflow And Operations Path

This path explains how users run work:

```text
YAML Surfaces and Config Groups
  -> config resolution
  -> CLI command
  -> local workflow or remote submit
  -> storage effect
```

Read:

1. [Config implementations](src/spice/conf/IMPLEMENTATIONS.md)
2. [Config resolution implementations](src/spice/config/IMPLEMENTATIONS.md)
3. [Workflow implementations](src/spice/workflows/IMPLEMENTATIONS.md)
4. [Execution implementations](src/spice/execution/IMPLEMENTATIONS.md)
5. [CLI command implementations](src/spice/cli/commands/IMPLEMENTATIONS.md)

The key mental model:

```text
surface name + overrides
  -> resolved workflow config
  -> acquire/train/tune/evaluate
  -> corpus/study/artifact state
```

## Core Terms

| Term | Meaning |
| --- | --- |
| Corpus | Stored canonical block data for one chain/dataset/evaluation date. |
| History rows | Rows before the evaluation window, used for training and warmup. |
| Evaluation rows | Rows in the evaluation day, used for diagnostic replay. |
| Feature | Numeric observable derived from block rows. |
| Sample | One temporal decision example. |
| Anchor row | Row representing the decision time for a sample. |
| Context rows | Past rows the model may observe. |
| Candidate window | Future row interval the model may choose from. |
| Candidate offset | Integer action relative to the candidate-window start. |
| Execution policy | Rule that maps decoded offsets to actual outcome rows. |
| Decoded Result ABI | Typed prediction output contract consumed by evaluators. |
| DecodedOffsets | Current candidate-offset decoded result ABI. |
| Artifact | Persisted trained model plus exact manifest and runtime state. |
| Study | Persisted tuning state and Optuna trial database. |
| Evaluator | Runtime scorer that turns decoded predictions into metrics. |
| Surface | High-level YAML recipe combining chain, dataset, model, features, problem, objective, and evaluation. |

## Stack

| Tool | Use |
| --- | --- |
| Typer | CLI |
| Pydantic + PyYAML | Config models and YAML loading |
| SQLAlchemy Core | SPICE-owned SQLite state |
| Polars | Corpus IO and validation |
| PyTorch | Modeling |
| Optuna | Tuning |
| web3.py | RPC access |

## Setup

```bash
brew install uv
uv sync --extra dev
source .venv/bin/activate
```

`uv` manages the repo-local `.venv/`. Without activation, prefix commands with `uv run`.

Push to both remotes:

```bash
git push origin main
git push university main
```

## CLI Quickstart

Local acquisition:

```bash
spice acquire --surface current_row_fee_dynamics
```

Remote train/tune/evaluate submission:

```bash
spice train --surface current_row_fee_dynamics --dataset-id cor_9a73b1e88edb488afb1e
spice tune --surface current_row_fee_dynamics --dataset-id cor_9a73b1e88edb488afb1e --trial-count 20
spice evaluate --artifact-id art_... --dataset-id cor_9a73b1e88edb488afb1e --evaluation poisson_replay_2h
```

The CLI owns the default remote target, `disi_l40`. Train, tune, and evaluate submit remotely by default; Python workflow runners remain available for the remote runner and tests.

Config and storage inspection:

```bash
spice config list provider
spice config show dataset icdcs_2026
spice config edit problem current_row_nominal

spice show dataset
spice show artifact --artifact-id art_...
spice delete artifact --artifact-id art_...
spice refresh catalog
```

Transfer:

```bash
spice transfer push dataset --dataset-id cor_9a73b1e88edb488afb1e
spice transfer pull artifact --artifact-id art_...
```

## Current Concrete IDs

| Seam | Current ids |
| --- | --- |
| Features | `core_fee_dynamics` |
| Temporal compilers | `observed_time_window` |
| Execution policies | `strict_deadline_miss` |
| Input normalization | `row_standard`, `window_weighted_standard` |
| Dataset builders | `fixed_sequence_temporal` |
| Model families | `lstm`, `transformer`, `transformer_lstm` |
| Prediction families | `min_block_fee_multitask` |
| Evaluators | `poisson_replay_2h`, `full_temporal_replay` |
| Remote target | `disi_l40` |

## Output Layout

```text
outputs/
  .spice/catalog.sqlite
  corpora/<chain>/<corpus_id>/
    history/
    evaluation/
    .spice/state.sqlite
  studies/<chain>/<study_id>/
    .spice/state.sqlite
  artifacts/<chain>/<artifact_id>/
    model.pt
    .spice/state.sqlite
```

Root state DBs and manifests are source of truth. The catalog is derived and can be refreshed.

## Verification

```bash
uv run ruff check .
uv run pyright
uv run pytest -q
```

YAML specs can be validated through raw config group loading:

```bash
uv run python - <<'PY'
from spice.config.groups import load_named_group_payload, named_group_keys, list_group_names

count = 0
errors = []
for group in named_group_keys():
    for name in list_group_names(group):
        try:
            load_named_group_payload(name, group)
        except Exception as exc:
            errors.append((group, name, type(exc).__name__, str(exc)))
        count += 1
print(f"validated={count} errors={len(errors)}")
for error in errors:
    print(error)
PY
```
