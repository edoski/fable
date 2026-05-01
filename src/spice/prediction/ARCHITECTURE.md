# Prediction Architecture

## Purpose

`prediction` owns task semantics: output heads, targets, training loss, epoch metrics, decoding, and decoded-result contracts.

## Theory

A model architecture says how tensors are transformed. A prediction family says what those tensors mean. Separating them lets the same model family learn different prediction tasks and lets evaluators depend on decoded semantics instead of logits.

## Pattern

Prediction configs compile into contracts. Contracts prepare targets, define output heads, compute losses and metrics, choose the primary metric, and decode model outputs into a typed decoded result.

`decoding.py` defines the generic decoded-result ABI. `DecodedOffsets` is offset-specific and lives in `decoded_offsets.py`. Generic code should depend on decoded-result ids, not assume every prediction is an offset task.

## Invariants

Prediction code may import temporal stores and execution policies because targets are built from temporal examples and policy-owned Action Spaces. Core rendering must not import prediction. Shared masking logic lives in `masking.py` so families apply action masks consistently.

Prediction training state is semantic-immutable. `fit_training_state()` returns reusable facts derived from the training split. Loss computation may cache device/dtype views on that state, but it must not mutate semantic tensors or depend on batch call order.

## Extension Points

Add a prediction family when target semantics change. Add shared helpers only when multiple families need the same mathematically generic operation.

## Module Map

```text
prediction/
  base.py       metric descriptors and output specs
  contracts.py  compiled prediction contract
  decoding.py   generic decoded-result ABI and decode context
  decoded_offsets.py candidate-offset decoded result and offset decode helper
  masking.py    shared candidate-logit masking helper
  registry.py   prediction family dispatch
  families/     concrete target/loss/decode implementations
```

## Prediction Flow

```text
CompiledProblemStore
      |
      v
target tensors and masks
      |
      v
model output heads
      |
      v
loss and epoch metrics
      |
      v
decoded prediction result
```

## Loss Versus Metric

The loss is what gradient descent optimizes. Metrics are what humans inspect. A family can optimize cross entropy while reporting accuracy, fee-related diagnostics, or other values. Evaluators are still separate because they score decoded decisions, not raw training loss.

## Decoded-Result Boundary

Evaluators declare accepted decoded-result ids. This avoids a hidden assumption that every model output is an offset classifier. A future prediction family can decode to another result type when evaluator contracts accept that result id.
