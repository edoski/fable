# Sequence Preparation Architecture

## Purpose

`modeling.dataset_builders` prepares canonical block frames for training and
inference. The name remains historical, but the public builder abstraction is gone:
there is one internal fixed-sequence path.

## Flow

```text
blocks
  |
  v
sort and deduplicate by block
  |
  v
feature_contract.build_table()
  |
  v
problem_contract.build_capability_store()  train
problem_contract.build_delay_store()       evaluate
  |
  v
derive fixed sequence length from training cadence
  |
  v
fit row-standard scaler on train rows
  |
  v
execution policy prepares selected sample facts
  |
  v
PreparedTrainingDataset / PreparedInferenceDataset
```

Training derives one `SequenceRuntimeMetadata` value from training samples:
sequence length, median cadence, and configured min/max bounds. The artifact
manifest persists that metadata so inference reconstructs the same context length.

Prepared training datasets own train, validation, and test selected samples. Each
role carries prepared temporal facts from the execution policy. Prepared inference
datasets own one selected action space. Runtime paths consume those prepared facts;
they do not rebuild split alignment from raw arrays.

## Invariants

Sequence preparation must preserve sample ordering, split assignment,
candidate-window alignment, feature-prerequisite validation, training-only scaler
fitting, and runtime metadata round-trip.
