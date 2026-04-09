# SPICE Temporal Baseline

Practical reproduction scaffold for the temporal module described in `ICDCS_2026.pdf`.

## Scope

- Build chain-local datasets from raw block data.
- Train baseline LSTM, Transformer, and Transformer-LSTM models.
- Evaluate classification accuracy, total loss, cost-over-optimum, and profit-over-baseline.
- Prepare a later handoff to chain-specific hyperparameter optimization.

## Recommended environment

- Python 3.11 or 3.12 for the most predictable PyTorch support.
- Apple Silicon is supported through the PyTorch MPS backend.
- Define `ALCHEMY_API_KEY` in a local `.env` before running data pulls.

## Planned workflow

1. Pull raw block data with `cryo`.
2. Enrich missing block fields if needed.
3. Build supervised datasets with fixed lookback and future windows.
4. Train the 27-model baseline matrix.
5. Run economic simulations on the evaluation day.
6. Start chain-specific HPO only after the baseline matrix is verified.

## Useful commands

- `python -m spice_temporal.cli plan-pull configs/baseline.yaml`
- `python -m spice_temporal.cli pull-blocks configs/baseline.yaml ethereum history`
