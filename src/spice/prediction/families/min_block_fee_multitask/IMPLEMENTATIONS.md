# `min_block_fee_multitask`

This prediction family trains two related tasks at once: classify the minimum-fee offset and regress the minimum log fee. Inference uses the offset classifier.

## Mental Model

The family asks the model to answer:

1. Which offset has the minimum block fee?
2. What is that minimum log fee value?

```text
sequence representation
  -> offset logits
  -> min-log-fee scalar
```

The scalar head gives an auxiliary training signal. The decoded action still comes from offset logits.

## Output Heads

| Head | Shape | Meaning |
| --- | --- | --- |
| `min_block_offset_logits` | `[batch, max_candidate_slots]` | Class scores for optimum offset. |
| `min_block_log_fee` | `[batch, 1]` | Predicted minimum log fee. |

## Target Batch

Targets include:

| Field | Meaning |
| --- | --- |
| `candidate_mask` | Valid resolvable action slots. |
| `min_block_offsets` | Class label for optimum offset. |
| `min_block_log_fees` | Regression target for minimum log fee. |

The optimum offset is the in-window minimum. Overflow slots remain selectable at decode time but the target class points to the best in-window row.

## Training State

The family computes training statistics:

| Statistic | Purpose |
| --- | --- |
| Inverse-frequency class weights | Balances rare optimum offsets in cross-entropy. |
| Fee mean and std | Normalizes the scalar fee regression target. |

Fee std has a small epsilon to avoid division by zero.

## Loss

The total loss combines classification and regression:

```text
classification = weighted_cross_entropy(offset_logits, target_offset)
regression = smooth_l1(normalized_predicted_fee, normalized_target_fee)
total_loss = classification + 0.5 * regression
```

The classification head chooses the action. The regression head helps the shared representation learn fee level.

## Beginner Theory: Why Two Tasks Can Help

Multitask learning trains one shared representation with more than one supervision signal. Here, the offset task teaches "which candidate is best," while the scalar fee task teaches "what fee level is expected." These tasks are related: knowing the fee curve shape can help identify the minimum, and identifying the minimum can help learn which fee levels matter.

Cross entropy is the classification loss. It expects unnormalized logits and a target class index. Internally, the useful idea is:

```text
loss = -log(probability assigned to the correct class)
```

Class weights matter when optimum offsets are imbalanced. If offset `0` is common and later offsets are rare, an unweighted classifier can look good by favoring common classes. Inverse-frequency weights increase the penalty for missing rare target offsets.

SmoothL1 is the regression loss. It behaves quadratically near zero error and linearly for larger errors. That makes small mistakes smooth to optimize while reducing sensitivity to large outliers compared with pure squared error.

The fee regression target is normalized before SmoothL1:

```text
normalized_fee = (log_fee - train_mean) / train_std
```

This keeps the classification and regression components on more comparable numeric scales.

## Metrics

| Metric | Meaning |
| --- | --- |
| `total_loss` | Combined training objective. |
| `offset_accuracy` | Fraction of samples with correct optimum offset. |
| `macro_f1` | Macro F1 of predicted offset against optimum offset over supported target classes. |
| `classification_loss` | Weighted cross-entropy component. |
| `regression_loss` | SmoothL1 component. |
| `log_fee_mae` | Mean absolute error of the denormalized log-fee prediction. |
| `log_fee_mse` | Mean squared error of the denormalized log-fee prediction. |

Primary metric is `total_loss`, minimized.

## Decode

Decode ignores the scalar fee head:

```text
min_block_offset_logits + action_mask
  -> masked argmax
  -> DecodedOffsets
```

The evaluator receives only candidate offsets.

## Failure Modes

| Failure | Meaning |
| --- | --- |
| Missing output head | Model does not satisfy the multitask contract. |
| Invalid class target | Optimum offset outside head width. |
| No valid action | Masked argmax cannot choose. |
| Fee stats unavailable | Regression target cannot be normalized. |

## Theory References

- PyTorch cross entropy: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- PyTorch SmoothL1 loss: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.smooth_l1_loss.html
