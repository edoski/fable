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

The policy owns action masks, temporal outcome facts, and selected-action realization. Temporal outcome facts contain baseline rows, per-action fee consequences, reachable actions, and overflow behavior. `CompiledExecutionPolicyContract.prepare_action_space()` validates that the prepared Action Space matches the requested samples, store action width, and mask shape before modeling, prediction, or evaluation consumes it. `prepare_temporal_facts()` prepares the Action Space and temporal outcome facts once for a sample set; prediction families then build their own target batches from those facts.

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

## Overflow Action Slots

The action mask marks the full action width as selectable. When a sample has fewer real candidate-window rows than the maximum, the remaining offsets are overflow actions. Under this policy, those offsets realize to the post-window row.

```text
sample A candidates: 0 1 2 3
sample B candidates: 0 1
action slots:        0 1 2 3

sample B offset 2 -> post-window row
sample B offset 3 -> post-window row
```

This is intentional. A true action slot means "the policy can resolve this action," not "this offset is inside the candidate window."

## Temporal Outcome Facts

Temporal outcome facts include per-action rows and log fees, baseline rows, reachable action masks, and overflow masks. For short candidate windows, overflow fees are filled from the post-window row so every action slot has a policy consequence. Prediction target batch shape remains prediction-family-owned.

## Failure Modes

| Failure | Meaning |
| --- | --- |
| Negative decoded offset | Model output is outside action ABI. |
| Candidate window has no rows | Problem compiler produced invalid sample. |
| Overflow needs missing post-window row | Corpus/evaluation data cannot resolve deadline miss. |
| Offsets length mismatches sample count | Prediction/evaluation alignment failed. |

## Extension Pattern

A new execution policy should define Action Space preparation, baseline, optimum, overflow behavior, and temporal outcome facts together. Prediction families and evaluators should consume the policy through the shared execution interface.
