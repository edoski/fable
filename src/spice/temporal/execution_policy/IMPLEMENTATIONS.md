# Concrete Execution Policy

Execution policies convert decoded model actions into concrete block rows. Current code has one policy: `strict_deadline_miss`.

## Mental Model

The model predicts an offset. The execution policy decides what that offset means economically.

```text
decoded offset
  -> requested row
  -> realized row
  -> realized fee
```

The policy also provides action masks plus baseline and optimum rows used by training targets and evaluation metrics.

## `strict_deadline_miss`

The policy models a deadline: the user wants to wait for a cheaper block, but missing the candidate window has a cost. If the selected offset goes past the candidate window, the transaction realizes at the post-window row.

```text
candidate window: [start, end)
post-window row: end

if start + offset < end:
    realized = start + offset
else:
    realized = post-window row
```

Negative offsets fail.

## Baseline And Optimum

For each sample:

| Row | Meaning |
| --- | --- |
| Baseline row | First candidate row. |
| Optimum row | Candidate row with minimum log base fee inside the candidate window. |
| Realized row | Row selected by decoded offset after overflow handling. |

Baseline is the no-wait comparison. Optimum is the best possible in-window choice with perfect future knowledge.

## Overflow Slots

The action mask marks all `max_candidate_slots` as selectable. When a sample has fewer real candidate rows than the maximum, the remaining offsets are overflow actions. Under this policy, those offsets realize to the post-window row.

```text
sample A candidates: 0 1 2 3
sample B candidates: 0 1
max slots:           0 1 2 3

sample B offset 2 -> post-window row
sample B offset 3 -> post-window row
```

This is intentional. A true action slot means "the policy can resolve this action," not "this offset is inside the candidate window."

## Target Construction

Prediction target batches include candidate fees, baseline index, optimum offset, optimum log fee, and action mask. For short candidate windows, overflow target fees are filled from the post-window row so every action slot has a fee consequence.

## Failure Modes

| Failure | Meaning |
| --- | --- |
| Negative decoded offset | Model output is outside action ABI. |
| Candidate window has no rows | Problem compiler produced invalid sample. |
| Overflow needs missing post-window row | Corpus/evaluation data cannot resolve deadline miss. |
| Offsets length mismatches sample count | Prediction/evaluation alignment failed. |

## Extension Pattern

A new execution policy should define baseline, optimum, overflow behavior, and target construction together. Prediction families and evaluators should consume the policy through the shared execution interface.
