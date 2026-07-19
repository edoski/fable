# Concrete Modeling Runtime

The modeling runtime builds fixed-sequence datasets, trains CUDA neural networks,
decodes predictions, runs evaluations, persists artifacts, and tunes
hyperparameters.

## Mental Model

```text
corpus rows
  -> feature matrix
  -> temporal problem store
  -> fixed sequence preparation
  -> Batch Plan
  -> prediction contract
  -> Lightning-hosted fit
  -> artifact manifest + model.pt
```

The artifact manifest records exact configs, semantic fingerprints, scaler,
sequence runtime metadata, temporal capability, and model metadata. Artifact root
state stores training and evaluation summaries. Prepared batch tensors are transient
runtime values.

## Training Context

Training compiles the feature, problem, prediction, and model-family contracts.
Fixed sequence preparation then owns sample selection, chronological split
assignment, sequence-length calibration, and scaler fitting.

## Batch Plan

The fixed sequence tensorizer builds:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `inputs` | `[batch, max_context, features]` | Padded feature windows. |
| `input_mask` | `[batch, max_context]` | True where a row is real input. |
| `action_mask` | `[batch, max_candidate_slots]` | Execution-policy action availability. |
| `sample_positions` | `[batch]` | Positions into the selected sample array. |

Windows are front-packed. Models use `take_last_valid` to read the final real
context position.

Batch Plan binds sequence inputs with prediction targets, groups samples by context
length, and returns streaming host `DataLoader`s. Training shuffles deterministically
at the batch level; validation, inference, and metric scoring keep stable order.

## Training Loop

Training is CUDA-only and uses Lightning only as the host for `fit`. The SPICE
Lightning module uses manual optimization with AdamW, `32-true` precision, gradient
clipping, SPICE prediction losses, and SPICE metric accumulators.

`TrainingFitPolicy` owns finite-metric checks, strict `min_delta`, one-based best
epoch, best CPU state, and patience stopping. Validation `total_loss` selects the
best state. The best state is restored before the training result returns.

## Inference

Inference allocates one decoded result buffer sized to selected samples. Each batch
writes decoded prediction values back by sample position.

```text
prepared Action Space
  -> streaming batches
  -> model outputs
  -> prediction contract decode
  -> decoded prediction result buffer
```

`DecodedOffsets` is the current candidate-offset decoded result ABI, but generic
modeling code depends only on the decoded-result id. Offset narrowing belongs in
the evaluator or serving code that requires offsets.

## Scoring Service

`score_evaluation()` validates evaluator compatibility, scores the model into a
decoded result, and calls `evaluator.run(store, execution_policy, decoded_result,
action_space)`. Prediction metric scoring uses the same streaming forward runtime
and prediction training state from the completed fit.

## Tuning

`spice.tuning.run_candidate` loads one Corpus, prepares the request's fit history,
runs one caller-supplied Method with its explicit deployment, and retains the
successful result. Candidate choice and Study publication remain separate operations.
