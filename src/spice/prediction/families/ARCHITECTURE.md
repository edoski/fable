# Prediction Families Architecture

## Purpose

`prediction.families` contains concrete prediction tasks behind the generic prediction contract.

## Theory

Each family chooses a learning objective. For classification, the target may be a candidate class. For multitask learning, the model may predict both a decision and auxiliary quantities. The family owns how these outputs become loss terms and metrics.

## Invariants

Families must validate action masks, keep target tensors aligned with sample indices, and return decoded results with declared ids. Evaluators should never read raw logits from a family.

## Extension Points

Create a new family for a new prediction target. Reuse shared masking helpers instead of reimplementing invalid-candidate behavior.

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

The shared helper keeps the invalid-slot behavior consistent across families.
