# Minimum Block Fee Multitask Architecture

## Purpose

This family predicts candidate-offset decisions with auxiliary fee-related signals.

## Theory

Multitask learning can improve representations by asking the model to learn related signals. The auxiliary task should support the main decision task without changing evaluator semantics.

## Invariants

The family decodes to `DecodedOffsets`. Auxiliary heads, losses, candidate masking, and offset metrics stay family-owned.

## Extension Points

Add auxiliary heads only when they are part of the family training loss. Add a new prediction family if downstream decoded behavior changes.

## Runtime Shape

```text
shared model representation
      |
      +--> offset classification head -> DecodedOffsets
      |
      +--> auxiliary fee head(s)       -> training signal only
```

## Multitask Rule

Auxiliary heads can influence training through additional loss terms. Evaluator contracts consume only the declared decoded-result semantics.
