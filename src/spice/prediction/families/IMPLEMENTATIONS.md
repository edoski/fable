# Concrete Prediction Families

Prediction families define the supervised task on top of a temporal problem. They own target preparation, output head specs, loss functions, metrics, and decoding.

## Mental Model

The model produces tensors. The prediction family gives those tensors meaning.

```text
problem store + execution policy
  -> target batch

model outputs + target batch
  -> loss and metrics

model outputs + action mask
  -> DecodedOffsets
```

Current evaluators accept `DecodedOffsets`, so both current prediction families decode to candidate offsets.

## Shared ABI

`DecodedOffsets` is a CPU int64 vector aligned by sample position. Inference allocates one buffer for the selected samples and each batch writes into it.

```text
batch sample positions: [5, 9, 12]
decoded offsets:        [0, 3, 1]
buffer[5]=0, buffer[9]=3, buffer[12]=1
```

## Action Masking

Prediction code uses masking helpers before softmax or argmax. The helpers validate shape and require at least one valid action per row.

Current problem stores build all-true action masks. Short candidate windows still expose overflow slots, and `strict_deadline_miss` resolves those offsets to the post-window row.

## Family Comparison

| Family | Training objective | Outputs | Decode |
| --- | --- | --- | --- |
| `min_block_fee_multitask` | Weighted offset classification plus fee regression. | Offset logits and scalar min log fee. | Masked argmax offset. |

## Metrics

Prediction metrics are computed on validation/test batches during model training. Evaluation metrics are separate and run through evaluator contracts after decoding.

```text
prediction metrics: train/validation loss, accuracy, profit-like diagnostics
evaluation metrics: replay or rollout economics over decoded offsets
```

## Failure Modes

| Failure | Meaning |
| --- | --- |
| No valid action in mask | The action space cannot be decoded. |
| Output shape mismatch | Model head width does not match max candidate slots. |
| Missing target field | Prediction family cannot compute loss. |
| Decoded length mismatch | Batch writes do not align with selected samples. |

## Extension Pattern

A new prediction family should define output specs, target preparation, loss, metrics, and decode together. If it decodes to a new result type, evaluators must declare that accepted decoded-result id.
