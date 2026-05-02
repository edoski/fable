# Dataset Builders Architecture

## Purpose

`modeling.dataset_builders` turns canonical block frames, feature contracts, temporal problem contracts, and artifact runtime state into prepared datasets for training and inference. This is the Temporal Dataset Preparation Interface.

Tensorization matters because the same temporal problem can be represented as independent rows, fixed-sequence sequences, or another model input shape without changing feature semantics or evaluator behavior.

## Generic Flow

```text
blocks
  |
  v
builder-local frame preparation
  |
  v
feature_contract.build_table()
  |
  v
problem_contract.build_capability_store()  train
problem_contract.build_delay_store()       evaluate
  |
  v
input normalization fit/transform
  |
  v
PreparedTrainingDataset / PreparedInferenceDataset
```

Frame preparation is intentionally local to each builder. A generic helper with flags can hide policy differences such as sort key, deduplication, or whether an empty evaluation frame is allowed. Those choices affect examples and therefore model behavior.

## Registry Dispatch

```text
dataset builder payload
  -> dataset_builder.id
  -> local spec config_type
  -> concrete DatasetBuilderConfig

compile_dataset_builder_contract(config)
  -> local spec
  -> require_spec_config(config, spec.config_type)
  -> concrete compile_dataset_builder(config)
```

The registry does not serialize and revalidate a concrete config during compile. Coercion validates. Compile dispatch asserts the invariant.

Config-facing dataset-builder and builder-runtime-metadata envelope failures use `ConfigResolutionError`. Already typed configs are returned unchanged.

## Runtime Metadata

Runtime metadata exists because some builders learn non-model assumptions during training. Examples include sequence length and calibrated timing assumptions.

```text
training prepare
  -> builder runtime metadata
  -> Temporal Capability
  -> artifact manifest
  -> Artifact Inference Context validation
  -> evaluation prepare
  -> reconstruct same assumptions
```

Artifact Inference Context validates artifact and corpus compatibility, then passes trusted artifact facts into the dataset-builder contract. The contract coerces builder runtime metadata, passes the artifact Temporal Capability through, and normalizes evaluation-window timestamps before the concrete builder prepares inference data.

Builder runtime metadata is builder-local. Compiler runtime metadata is not hidden inside it; compiler metadata travels with Temporal Capability so artifact action width and compiler assumptions stay one typed value.

## Preparation Types

`preparation.py` contains the public prep specs and prepared dataset results. `base.py` contains registry/config dispatch and `CompiledDatasetBuilderContract`. Concrete builders depend on the preparation Interface instead of importing orchestration types from `pipeline.py`.

## Invariants

Builders must preserve:

```text
sample ordering
split assignment
candidate-window alignment
feature prerequisite validation
training/inference scaler ownership
runtime metadata round-trip
```

Sampling and split behavior are builder-owned invariants.

## Extension Points

Add a builder when model input representation changes. Keep it behind `DatasetBuilderConfig`, `CompiledDatasetBuilderContract`, preparation specs, and a local spec entry. Avoid workflow branches on builder ids.
