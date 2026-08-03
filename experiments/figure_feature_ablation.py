"""Render validation feature-ablation deltas from canonical Studies."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from uuid import UUID

import matplotlib as mpl
import numpy as np
from figure_style import DEFAULT_OUTPUT_DIRECTORY, display_name, save_pdf, subplots
from matplotlib.colors import TwoSlopeNorm

from kairos.experiments import ExperimentKind, load_experiment_manifest
from kairos.study import load_study


def _objective(storage_root: Path, study_id: UUID) -> float:
    study = load_study(storage_root, study_id)
    if len(study.trials) != 1:
        raise ValueError("feature-ablation Studies must contain exactly one trial")
    return study.trials[0].objective


_FEATURE_LABELS = {
    "base_fee": "Base fee",
    "gas_utilization": "Gas utilization",
    "exact_forming_base_fee": "Forming base fee",
    "gas_limit": "Gas limit",
    "transaction_count": "Transaction count",
    "block_interval": "Block interval",
    "hour": "UTC hour",
    "day_of_week": "UTC day of week",
    "priority_fee_p50": "Priority-fee P50",
    "priority_fee_p90": "Priority-fee P90",
}


def _feature_label(configuration: str) -> str:
    feature = configuration.removeprefix("without_")
    return _FEATURE_LABELS.get(feature, display_name(feature))


def _family_label(family: str) -> str:
    return "Hybrid" if family == "transformer_lstm" else display_name(family)


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
        if configuration.startswith("without_") and configuration not in configurations:
            configurations.append(configuration)
        objectives[chain, family, configuration] = _objective(storage_root, study_id)

    configurations.sort(key=lambda value: value == "without_exact_forming_base_fee")

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

    deltas = np.full(
        (len(chains), len(configurations), max(map(len, families_by_chain.values()))), np.nan
    )
    for chain_index, chain in enumerate(chains):
        for family_index, family in enumerate(families_by_chain[chain]):
            baseline = objectives[chain, family, "full"]
            for configuration_index, configuration in enumerate(configurations):
                if configuration in configurations_by_chain[chain]:
                    deltas[chain_index, configuration_index, family_index] = 100.0 * (
                        objectives[chain, family, configuration] - baseline
                    )

    finite = deltas[np.isfinite(deltas)]
    if finite.size == 0:
        raise ValueError("feature-ablation manifest contains no omission cells")
    limit = float(np.max(np.abs(finite)))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    colormap = mpl.colormaps["RdBu_r"].copy()
    colormap.set_bad("#F2F2F2")

    figure, axes = subplots(1, len(chains), height=4.0)
    image = None
    for column, chain in enumerate(chains):
        axis = axes[0, column]
        families = families_by_chain[chain]
        image = axis.imshow(
            deltas[column, :, : len(families)], aspect="auto", cmap=colormap, norm=norm
        )
        axis.set_title(display_name(chain))
        axis.set_xticks(
            range(len(families)),
            [_family_label(value) for value in families],
            rotation=30,
            ha="right",
            rotation_mode="anchor",
        )
        axis.set_yticks(
            range(len(configurations)), [_feature_label(value) for value in configurations]
        )
        if column > 0:
            axis.tick_params(axis="y", labelleft=False)
        axis.grid(False)
        axis.set_xticks(np.arange(-0.5, len(families), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(configurations), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.8)
        axis.tick_params(which="minor", bottom=False, left=False)
        for configuration_index, configuration in enumerate(configurations):
            if configuration not in configurations_by_chain[chain]:
                for family_index in range(len(families)):
                    axis.text(
                        family_index,
                        configuration_index,
                        "N/A",
                        ha="center",
                        va="center",
                        color="#666666",
                        fontsize=6,
                    )

    assert image is not None
    colorbar = figure.colorbar(
        image, ax=axes.ravel().tolist(), orientation="horizontal", shrink=0.72, aspect=35, pad=0.08
    )
    colorbar.set_label(
        "Better (lower cost)  ←  change from full contract (pp)  →  Worse (higher cost)"
    )
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
