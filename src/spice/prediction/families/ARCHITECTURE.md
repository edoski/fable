# Prediction Families Architecture

## Purpose

`prediction.families` contains concrete prediction tasks behind the generic prediction contract.

## Theory

Each family chooses supervised prediction semantics. For classification, the target may be a candidate class. For multitask learning, the model may predict both a decision and auxiliary quantities. The family owns how temporal outcome facts become target batches, loss terms, and prediction metrics.

## Invariants

Families must validate action masks, keep target tensors aligned with sample indices, and return decoded results with declared ids. Evaluators should never read raw logits from a family.

## Extension Points

Create a new family for a new prediction target. Extract shared masking helpers only after multiple families need the same invalid-candidate behavior.

## Family Implementation Checklist

```text
config model
  -> compiled prediction contract
  -> target preparation
  -> output head specs
  -> loss function
  -> epoch metric calculation
  -> decoder
  -> primary metric id
```

## Action Masking

Temporal stores may have variable candidate counts. Families that score candidates must mask invalid slots before softmax, argmax, or loss operations:

```text
raw logits + action mask -> masked logits -> probabilities/decision
```

Masking behavior belongs to the family until it has a second caller.
