"""Render held-out horizon economics and rolling-policy deltas."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import numpy as np
from figure_style import (
    DEFAULT_OUTPUT_DIRECTORY,
    add_family_legend,
    display_name,
    family_style,
    save_pdf,
    subplots,
)

from kairos.evaluation import ROLLING_HORIZONS, reduce_evaluation, reduce_rolling
from kairos.experiments import ExperimentKind, load_experiment_manifest

_METRICS = (
    ("base_fee_savings", "Base-fee\nsavings (%)"),
    ("p50_fee_inclusive_savings", "P50 fee-inclusive\nsavings (%)"),
    ("base_fee_optimality_gap", "Cost over\noptimum (%)"),
)


@dataclass(frozen=True)
class _HeldOutResults:
    chains: tuple[str, ...]
    families_by_chain: dict[str, tuple[str, ...]]
    roster: dict[tuple[str, str], dict[int, UUID]]
    metrics: dict[tuple[str, str, int], dict[str, float]]


def _load_results(storage_root: Path, experiment_id: UUID) -> _HeldOutResults:
    manifest = load_experiment_manifest(storage_root, ExperimentKind.HELD_OUT, experiment_id)
    chains: list[str] = []
    families_by_chain: dict[str, list[str]] = defaultdict(list)
    roster: dict[tuple[str, str], dict[int, UUID]] = defaultdict(dict)
    metrics = {}
    for cell, evaluation_id in manifest.items():
        try:
            chain, family, horizon_label = cell.split(".")
            horizon = int(horizon_label.removeprefix("K"))
        except ValueError as error:
            raise ValueError(f"invalid held-out cell: {cell}") from error
        if chain not in chains:
            chains.append(chain)
        if family not in families_by_chain[chain]:
            families_by_chain[chain].append(family)
        roster[chain, family][horizon] = evaluation_id
        metrics[chain, family, horizon] = reduce_evaluation(storage_root, evaluation_id).row(
            0, named=True
        )
    return _HeldOutResults(
        chains=tuple(chains),
        families_by_chain={chain: tuple(families) for chain, families in families_by_chain.items()},
        roster=dict(roster),
        metrics=metrics,
    )


def _render_horizons(results: _HeldOutResults, output_directory: Path) -> Path:
    figure, axes = subplots(len(results.chains), len(_METRICS), height=1.85 * len(results.chains))
    for row, chain in enumerate(results.chains):
        for column, (metric, label) in enumerate(_METRICS):
            axis = axes[row, column]
            for family in results.families_by_chain[chain]:
                horizons = sorted(results.roster[chain, family])
                color, marker = family_style(family)
                axis.plot(
                    horizons,
                    [
                        100.0 * results.metrics[chain, family, horizon][metric]
                        for horizon in horizons
                    ],
                    color=color,
                    marker=marker,
                    label=display_name(family),
                )
            if row == 0:
                axis.set_title(label)
            if row == len(results.chains) - 1:
                axis.set_xlabel("Horizon blocks K")
            if column == 0:
                axis.set_ylabel(display_name(chain))
    add_family_legend(figure, axes[0, 0])
    return save_pdf(figure, output_directory / "horizon-study.pdf")


def _render_rolling(storage_root: Path, results: _HeldOutResults, output_directory: Path) -> Path:
    required = set(ROLLING_HORIZONS)
    rolling_roster = {}
    for (chain, family), evaluations in results.roster.items():
        missing = sorted(required.difference(evaluations))
        if missing:
            raise ValueError(f"{chain}.{family} lacks rolling horizons {missing}")
        rolling_roster[f"{chain}.{family}"] = {
            horizon: evaluations[horizon] for horizon in ROLLING_HORIZONS
        }
    rows = {
        row["cell"]: row
        for row in reduce_rolling(storage_root, rolling_roster).iter_rows(named=True)
    }
    figure, axes = subplots(len(results.chains), len(_METRICS), height=1.75 * len(results.chains))
    for row_index, chain in enumerate(results.chains):
        families = results.families_by_chain[chain]
        positions = np.arange(len(families))
        for column, (metric, label) in enumerate(_METRICS):
            axis = axes[row_index, column]
            values = []
            colors = []
            for family in families:
                result = rows[f"{chain}.{family}"]
                values.append(100.0 * (result[f"rolling_{metric}"] - result[f"one_shot_{metric}"]))
                colors.append(family_style(family)[0])
            axis.bar(positions, values, color=colors, width=0.65)
            axis.axhline(0.0, color="#333333", linewidth=0.7)
            axis.set_xticks(
                positions, [display_name(family) for family in families], rotation=25, ha="right"
            )
            if row_index == 0:
                axis.set_title(label)
            if column == 0:
                axis.set_ylabel(f"{display_name(chain)}\nΔ rolling − one-shot (pp)")
    return save_pdf(figure, output_directory / "rolling-comparison.pdf")


def render(storage_root: Path, experiment_id: UUID, output_directory: Path) -> tuple[Path, Path]:
    results = _load_results(storage_root, experiment_id)
    horizons = _render_horizons(results, output_directory)
    rolling = _render_rolling(storage_root, results, output_directory)
    print(horizons)
    print(rolling)
    return horizons, rolling


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("storage_root", type=Path)
    parser.add_argument("experiment_id", type=UUID)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    arguments = parser.parse_args()
    render(arguments.storage_root, arguments.experiment_id, arguments.output_directory)


if __name__ == "__main__":
    main()
