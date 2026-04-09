# SPICE Temporal Baseline

Practical reproduction scaffold for the temporal module described in `ICDCS_2026.pdf`.

## Scope

- Build chain-local datasets from raw block data.
- Train baseline LSTM, Transformer, and Transformer-LSTM models.
- Evaluate classification accuracy, total loss, cost-over-optimum, and profit-over-baseline.
- Prepare a later handoff to chain-specific hyperparameter optimization.

## Architecture

- `config.py`: typed experiment, training, and chain configuration.
- `contracts.py`: typed boundary contracts for raw rows, tensor batches, and model outputs.
- `env.py`: local `.env` loading and Alchemy URL resolution.
- `cryo.py`: cryo pull planning and execution.
- `io.py` and `enrich.py`: block-file loading plus `gas_limit` enrichment for cryo output.
- `features.py`, `datasets.py`, and `normalization.py`: feature engineering, supervised example construction, chronological splits, and train-only scaling.
- `models.py`, `torch_datasets.py`, `training.py`, and `evaluation.py`: PyTorch models, tensor adapters, training loop, and metrics.
- `pipeline.py`: one-file orchestration for a single training run.
- `simulation.py`: temporal economic simulation helpers for the evaluation phase.
- `cli.py`: Typer entrypoints for pull, enrich, and train commands.

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

## First pilot

Use a small Ethereum history slice to validate the full raw-data-to-training path before larger pulls:

1. `python -m spice_temporal.cli pull-blocks configs/baseline.yaml ethereum history --dry-run false`
2. `python -m spice_temporal.cli enrich-blocks configs/baseline.yaml ethereum <raw-path> <enriched-path>`
3. `python -m spice_temporal.cli train-single configs/baseline.yaml <enriched-file> ethereum lstm 12`

The first pilot target is `Ethereum + 12s + LSTM`, because it is the simplest and most stable baseline cell.

## Verification

- `ruff check src tests`
- `pyright`
- `PYTHONPATH=src pytest -q`

Typing is intentionally boundary-focused for now rather than repo-wide strict.

## Useful commands

- `python -m spice_temporal.cli show-config configs/baseline.yaml`
- `python -m spice_temporal.cli plan-pull configs/baseline.yaml`
- `python -m spice_temporal.cli pull-blocks configs/baseline.yaml ethereum history`
- `python -m spice_temporal.cli enrich-blocks configs/baseline.yaml ethereum <input> <output>`
- `python -m spice_temporal.cli train-single configs/baseline.yaml <enriched-file> ethereum lstm 12`
