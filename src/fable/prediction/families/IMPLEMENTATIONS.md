# Concrete Prediction Families

Prediction families define the supervised task on top of a temporal problem. They own target preparation, output head specs, loss functions, metrics, and decoding.

## Mental Model

The model produces tensors. The prediction family gives those tensors meaning.

```text
prepared temporal facts
  -> prediction target batch

model outputs + target batch
  -> loss and metrics

model outputs + action mask
  -> DecodedOffsets
```

Current evaluators accept the candidate-offset decoded result ABI, so current prediction families decode to `DecodedOffsets`.

## Shared ABI

`DecodedOffsets` is a CPU int64 vector aligned by sample position. Inference allocates one buffer for the selected samples and each batch writes into it.

```text
batch sample positions: [5, 9, 12]
decoded offsets:        [0, 3, 1]
buffer[5]=0, buffer[9]=3, buffer[12]=1
```

## Action Masking

Prediction families mask invalid actions before softmax or argmax. Family masking code validates shape and requires at least one valid action per row. Shared helpers should wait until a second family needs the same operation.

Execution policies own action availability. `strict_deadline_miss` currently prepares a full action mask for every compiled candidate slot; short candidate windows still expose overflow slots, and the policy resolves those offsets to the post-window row.

## Family Comparison

| Family | Training loss | Outputs | Decode |
| --- | --- | --- | --- |
| `min_block_fee_multitask` | Weighted offset classification plus fee regression. | Offset logits and scalar min log fee. | Masked argmax offset. |

## Metrics

Prediction metrics are computed on validation/test batches during model training. Evaluation metrics are separate and run through evaluator contracts after decoding.

```text
prediction metrics: train/validation loss, offset accuracy, macro F1, log-fee regression diagnostics
evaluation metrics: temporal replay economics over decoded offsets
```

The `min_block_fee_multitask` training state stores semantic CPU tensors for class weights and fee normalization. Runtime loss calls may resolve cached device/dtype views, but those cached views must not change the semantic tensors. This lets Training Runtime Plan fit the state once and reuse it across probe, train, validation, returned result metadata, and split metrics.

## Failure Modes

| Failure | Meaning |
| --- | --- |
| No valid action in mask | The action space cannot be decoded. |
| Output shape mismatch | Model head width does not match max candidate slots. |
| Missing target field | Prediction family cannot compute loss. |
| Decoded length mismatch | Batch writes do not align with selected samples. |

## Extension Pattern

A new prediction family should define output specs, target preparation, loss, metrics, and decode together. If it decodes to a new result type, evaluators must declare that accepted decoded-result id.
