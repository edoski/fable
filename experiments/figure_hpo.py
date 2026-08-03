"""Render validation HPO deltas from canonical Studies."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from uuid import UUID

from figure_style import (
    DEFAULT_OUTPUT_DIRECTORY,
    add_family_legend,
    display_name,
    family_style,
    save_pdf,
    subplots,
)

from kairos.experiments import ExperimentKind, load_experiment_manifest
from kairos.study import load_study


def render(storage_root: Path, experiment_id: UUID, output_directory: Path) -> Path:
    manifest = load_experiment_manifest(storage_root, ExperimentKind.HPO, experiment_id)
    objectives: dict[tuple[str, str], tuple[float, ...]] = {}
    winners: dict[tuple[str, str], int] = {}
    chains: list[str] = []
    families_by_chain: dict[str, list[str]] = defaultdict(list)

    for cell, study_id in manifest.items():
        try:
            chain, family = cell.split(".")
        except ValueError as error:
            raise ValueError(f"invalid HPO cell: {cell}") from error
        study = load_study(storage_root, study_id)
        if chain not in chains:
            chains.append(chain)
        if family not in families_by_chain[chain]:
            families_by_chain[chain].append(family)
        objectives[chain, family] = tuple(trial.objective for trial in study.trials)
        winners[chain, family], _ = study.best_result()

    figure, axes = subplots(1, len(chains), height=2.5)
    for column, chain in enumerate(chains):
        axis = axes[0, column]
        for family in families_by_chain[chain]:
            values = objectives[chain, family]
            candidates = range(1, len(values) + 1)
            baseline = values[0]
            deltas = [100.0 * (value - baseline) for value in values]
            color, marker = family_style(family)
            axis.plot(candidates, deltas, color=color, marker=marker, label=display_name(family))
            winner = winners[chain, family]
            axis.scatter(winner + 1, deltas[winner], color=color, s=32, facecolors="none", zorder=3)
        axis.axhline(0.0, color="#333333", linewidth=0.7)
        axis.set_title(display_name(chain))
        axis.set_xlabel("L9 candidate")
        if column == 0:
            axis.set_ylabel("Δ Cost over optimum (pp)")

    add_family_legend(figure, axes[0, 0])
    path = save_pdf(figure, output_directory / "hpo.pdf")
    print(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("storage_root", type=Path)
    parser.add_argument("experiment_id", type=UUID)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    arguments = parser.parse_args()
    render(arguments.storage_root, arguments.experiment_id, arguments.output_directory)


if __name__ == "__main__":
    main()
