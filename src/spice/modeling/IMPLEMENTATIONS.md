# Concrete Modeling Runtime

The modeling runtime builds datasets, trains CUDA neural networks, decodes predictions, runs evaluation objectives, persists artifacts, and tunes hyperparameters.

## Mental Model

SPICE trains a model to choose a candidate offset for each temporal sample.

```text
corpus rows
  -> feature matrix
  -> temporal problem store
  -> dataset builder
  -> Representation contract
  -> Batch Plan
  -> prediction contract
  -> loss/metrics
  -> artifact manifest + model.pt
```

The artifact manifest records exact configs, semantic fingerprints, scaler, dataset-builder metadata, and model metadata. Artifact root state stores training and evaluation summaries. Prepared batch storage is transient runtime storage; artifact persistence stores manifest, scaler, builder runtime metadata, model weights, and summaries, not prepared batch tensors.

## Training Context

Training starts by compiling all contracts:

```text
features config         -> feature contract
problem spec            -> temporal problem contract
prediction config       -> prediction contract
objective config        -> objective contract
objective + evaluator contract -> objective runtime
dataset_builder config  -> dataset builder contract
input_normalization     -> scaler policy
representation          -> model-input representation contract
model config            -> model family
```

The compiled context is the source of truth for model input width, prediction output heads, action-space size, prediction target preparation contracts, evaluator config identity, objective metric direction, Objective Runtime metric production, and dataset-builder preparation context.

## Batch Plan

The sequence representation builds tensors:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `inputs` | `[batch, max_context, features]` | Padded feature windows. |
| `input_mask` | `[batch, max_context]` | True where a row is real input. |
| `action_mask` | `[batch, max_candidate_slots]` | Execution-policy action availability. |
| `sample_positions` | `[batch]` | Positions into the selected sample array. |

Windows are front-packed. Models use `take_last_valid` to read the final real context position.

Batch Plan binds representation batches with prediction targets, orders samples by batch signature, and chooses host or device-resident storage after runtime device-storage budget is known. `DeviceStorageBudget` names the phase of that budget: disabled host-only storage, coarse startup estimate, or measured residual capacity after a runtime probe. CUDA budget discovery belongs to runtime; Batch Plan consumes `ModelingRuntimePlan`, owns host-loader and device-storage policy, and passes only host-memory and batch-size facts into the Representation seam. Selected-sample facts are prepared before Batch Plan: training passes split-bound temporal facts, and inference/scoring passes a prepared Action Space.

## Training Loop

Training is CUDA-only. `runtime_planning.py` builds the executable runtime plan: seed setup, CUDA runtime context, backend determinism, precision, and model placement/compilation. `_runtime.py` owns coarse CUDA budget discovery and shared budget arithmetic.

`training_runtime.prepare_training_runtime()` prepares the model, optimizer, and batch plan from that runtime plan. `plan_training_runtime()` consumes the whole `ModelingRuntimePlan`, uses the private runtime probe helper to build the host warmup plan, measure the budget, and return the final measured runtime plan, then performs the training-specific warmup body: unshuffled host Batch Plan, temporary AdamW, one gradient-bearing probe step, model-state restore, and cache cleanup. Restore and cleanup run even if the probe fails. `TrainingResult` owns the semantic-immutable prediction training state reused for train, validation, returned training results, and split metrics. The final measured runtime plan is the runtime plan carried by the training batch plan, evaluation objectives, and persisted split-metric scoring.

`_epoch_execution` owns the mechanics inside a train or validation epoch. `_fit_policy` owns finite-metric behavior, objective history, strict `min_delta`, best-state tracking, progress payloads, and patience stopping. `training_runner.run_training_fit()` calls callbacks, keeps the best state in memory, restores it before returning, and assembles the public result.

```text
for epoch:
    train batches
    compute prediction loss
    clip gradients
    validate
    evaluate objective metric
    retain best state
    apply early stopping
```

Optimizer is AdamW. The runtime plan resolves precision to `32-true`, and the runtime only supports that precision. The best in-memory state is restored before final artifact persistence.

## Beginner Theory: Loss, Backpropagation, And Optimization

Training needs one scalar loss. The prediction family computes that loss from model outputs and target batches. Backpropagation differentiates the loss with respect to every trainable weight. The optimizer then changes the weights.

```text
outputs = model(inputs)
loss = prediction_family.loss(outputs, targets)
compute gradients from loss
clip gradients
optimizer.step()
```

AdamW is the current optimizer. Adam keeps moving averages of gradients and squared gradients so each parameter gets an adaptive step size. AdamW applies weight decay separately from those adaptive moments, which makes the regularization effect easier to reason about than coupling it into the gradient update.

Gradient clipping limits the total gradient norm before the optimizer step. This does not change the loss function. It prevents one unusually large batch from producing a very large update that destabilizes training.

Mixed precision is intentionally not part of the current modeling runtime. A future precision policy would need an explicit design pass because lower precision can change numeric behavior, not just speed.

Early stopping is a model-selection rule. If the objective metric stops improving by at least `min_delta` for `patience` epochs, training stops and the best state is restored. This keeps the persisted artifact tied to the best validation objective, not the final epoch by accident.

## Inference

Inference allocates one decoded result buffer sized to the selected samples. Each batch writes a generic decoded prediction result back by sample position.

```text
prepared Action Space
  -> batches
  -> model outputs
  -> prediction contract decode
  -> decoded prediction result buffer
```

`DecodedOffsets` is the current candidate-offset decoded result ABI, but generic modeling code depends only on the decoded-result id. Offset narrowing belongs inside the evaluator runner that requires offsets.

Forward-only inference and split-metric passes use `forward_runtime`: callers pass one `ModelingRuntimePlan`, then runtime asks `_runtime_probe.py` to build a host warmup Batch Plan with disabled device-storage budget, measure one no-grad eval forward batch, and return the final measured runtime plan. `forward_runtime` then builds the final Batch Plan with the measured budget and executes model forward. Probe outputs are discarded. Normal forward measurement does not clear CUDA cache; cache clearing is limited to Batch Plan OOM fallback and destructive training-probe cleanup.

## Scoring Service

`score_evaluation()` is the bridge from model to evaluator:

```text
validate evaluator accepts prediction contract
  -> score model into decoded result
  -> evaluator.run(store, execution_policy, decoded_result, action_space)
```

This keeps evaluation scoring independent from training-loop details while making device, precision, runtime context, determinism, seed, and compile policy explicit through `EvaluationScoringRuntimePlan`. Prediction metric scoring lives in the same module and consumes the final `ModelingRuntimePlan` instead of rebuilding runtime facts from training config.

## Artifact Persistence

A trained artifact contains:

| Item | Meaning |
| --- | --- |
| `model.pt` | PyTorch state. |
| Artifact manifest | Exact authored configs and semantic identity. |
| Scaler | Training-fitted input normalization. |
| Builder runtime metadata | Dataset-builder-specific inference metadata. |
| Training summary/epochs | Runtime training metrics. |
| Evaluation summaries | Stored diagnostic evaluation runs. |

Evaluate workflow reconstructs trained semantics from the artifact manifest, validates selected corpus compatibility, and applies the active evaluate config's evaluator, delay, batch, and root-id controls before running inference.

## Tuning

Tuning Execution uses Optuna. `modeling.tuning_execution` opens or resumes the storage-backed study, validates that the requested trial count can extend existing state, samples typed params, applies them to the base config, trains each trial through the non-persisted trial-training path, records sampled params and best epoch metadata, and returns the selected objective metric.

The lifecycle is: tuning space -> shared categorical sampler -> model-family adapter -> typed tuned params -> validated copy of the base model config. Training and problem parameters use the same shared sampler. Architecture-specific derivations stay in the family adapter.

Study state stores the study manifest and Optuna tables. Trial user attributes record sampled params and best epoch. The tune workflow supplies reporting callbacks; it does not own trial execution.

## Failure Modes

| Failure | Meaning |
| --- | --- |
| No CUDA | Current training implementation requires NVIDIA CUDA. |
| Model/prediction shape mismatch | Output heads do not match prediction contract. |
| Empty split | Dataset builder produced no samples for a split. |
| Objective metric missing | Prediction/evaluator metrics do not expose configured metric. |
| Artifact/corpus semantic mismatch | Selected artifact or corpus state is incompatible with manifest-first evaluation constraints. |
| Tuning identity mismatch | Existing study state belongs to a different study definition. |

## Extension Pattern

New modeling implementations should enter through contracts: model families own architectures, prediction families own output/loss/decode, dataset builders own sample construction, and workflows only orchestrate those pieces.

## Theory References

- AdamW: Loshchilov and Hutter, "Decoupled Weight Decay Regularization": https://arxiv.org/abs/1711.05101
- PyTorch AdamW behavior: https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW
- PyTorch gradient clipping: https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
