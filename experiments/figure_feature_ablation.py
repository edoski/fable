"""Render validation feature-ablation deltas from canonical Studies."""

from __future__ import annotations

import argparse
from collections import defaultdict
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

from fable.experiments import ExperimentKind, load_experiment_manifest
from fable.study import load_study


def _objective(storage_root: Path, study_id: UUID) -> float:
    study = load_study(storage_root, study_id)
    if len(study.trials) != 1:
        raise ValueError("feature-ablation Studies must contain exactly one trial")
    return study.trials[0].objective


def _configuration_label(configuration: str) -> str:
    if configuration == "base_only":
        return "Base fee only"
    if configuration.startswith("without_"):
        return f"Without {display_name(configuration.removeprefix('without_')).lower()}"
    return display_name(configuration)


def render(storage_root: Path, experiment_id: UUID, output_directory: Path) -> Path:
    manifest = load_experiment_manifest(
        storage_root, ExperimentKind.FEATURE_ABLATION, experiment_id
    )
    objectives: dict[tuple[str, str, str], float] = {}
    chains: list[str] = []
    families_by_chain: dict[str, list[str]] = defaultdict(list)
    configurations_by_chain: dict[str, list[str]] = defaultdict(list)
    configurations: list[str] = []

    for cell, study_id in manifest.items():
        try:
            chain, family, configuration = cell.split(".")
        except ValueError as error:
            raise ValueError(f"invalid feature-ablation cell: {cell}") from error
        if chain not in chains:
            chains.append(chain)
        if family not in families_by_chain[chain]:
            families_by_chain[chain].append(family)
        if configuration != "full" and configuration not in configurations_by_chain[chain]:
            configurations_by_chain[chain].append(configuration)
        if configuration != "full" and configuration not in configurations:
            configurations.append(configuration)
        objectives[chain, family, configuration] = _objective(storage_root, study_id)

    for chain in chains:
        expected = ("full", *configurations_by_chain[chain])
        for family in families_by_chain[chain]:
            missing = [
                configuration
                for configuration in expected
                if (chain, family, configuration) not in objectives
            ]
            if missing:
                raise ValueError(f"{chain}.{family} lacks configurations {missing}")

    figure, axes = subplots(1, len(chains), height=4.6)
    for column, chain in enumerate(chains):
        axis = axes[0, column]
        families = families_by_chain[chain]
        positions = np.arange(len(configurations), dtype=float)
        width = 0.8 / len(families)
        for family_index, family in enumerate(families):
            baseline = objectives[chain, family, "full"]
            deltas = [
                100.0 * (objectives[chain, family, configuration] - baseline)
                if configuration in configurations_by_chain[chain]
                else np.nan
                for configuration in configurations
            ]
            color, _ = family_style(family)
            offset = (family_index - (len(families) - 1) / 2.0) * width
            axis.barh(
                positions + offset, deltas, height=width, color=color, label=display_name(family)
            )
        axis.axvline(0.0, color="#333333", linewidth=0.7)
        axis.set_title(display_name(chain))
        axis.set_yticks(positions, [_configuration_label(value) for value in configurations])
        axis.invert_yaxis()
        axis.set_xlabel("Δ Cost over optimum (pp)")
        if column > 0:
            axis.tick_params(axis="y", labelleft=False)

    add_family_legend(figure, axes[0, 0])
    path = save_pdf(figure, output_directory / "feature-ablation.pdf")
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
