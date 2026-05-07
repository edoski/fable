# Prediction Architecture

## Purpose

`prediction` owns task semantics: output heads, targets, training loss, training metric production, decoding, and decoded-result contracts.

## Theory

A model architecture says how tensors are transformed. A prediction family says what those tensors mean. Separating them lets the same model family learn different prediction tasks and lets evaluators depend on decoded semantics instead of logits.

## Pattern

Prediction configs compile into contracts. Contracts prepare targets, define output heads, compute losses and training metrics, choose the primary training metric, and decode model outputs into a typed decoded result. Generic metric descriptors, metric sets, and window summaries live in `spice.metrics`.

`decoding.py` defines the generic decoded-result ABI. `DecodedOffsets` is offset-specific and lives in `decoded_offsets.py`. Family-owned offset masking stays inside `families/min_block_fee_multitask/outputs.py`. Generic code should depend on decoded-result ids, not assume every prediction is an offset task.

## Invariants

Prediction code may import prepared temporal facts because target batches are built from policy-owned Action Spaces and Temporal Outcome Facts. Core rendering must not import prediction. Candidate masking stays family-owned until multiple families need the same operation.

Prediction target preparation receives prepared temporal facts rather than an independent sample selection. Those facts pair the policy-owned Action Space with Temporal Outcome Facts for exactly one sample set. Training target batches use the policy-owned mask from the Action Space, while family-specific labels and auxiliary targets stay prediction-owned.

Prediction training state is semantic-immutable. `fit_training_state()` derives reusable facts from the same prepared temporal facts used by prediction target batches. Loss computation may cache device/dtype views on that state, but it must not mutate semantic tensors or depend on batch call order.

## Extension Points

Add a prediction family when target semantics change. Add shared helpers only when multiple families need the same mathematically generic operation.

## Module Map

```text
prediction/
  base.py       output specs
  contracts.py  compiled prediction contract
  decoding.py   generic decoded-result ABI and decode context
  decoded_offsets.py candidate-offset decoded result
  registry.py   prediction family dispatch
  families/     concrete target/loss/decode implementations
```

## Prediction Flow

```text
CompiledProblemStore
      |
      v
prepared Action Space
      |
      v
prepared temporal facts
      |
prediction target batch
      |
      v
model output heads
      |
      v
loss and training metrics
      |
      v
decoded prediction result
```

## Loss Versus Metric

The loss is what gradient descent optimizes. Metrics are what humans inspect. A family can optimize cross entropy while reporting accuracy, fee-related diagnostics, or other values. Evaluators are still separate because they score decoded decisions, not raw training loss.

## Decoded-Result Boundary

Evaluators declare accepted decoded-result ids. This avoids a hidden assumption that every model output is an offset classifier. A future prediction family can decode to another result type when evaluator contracts accept that result id.
