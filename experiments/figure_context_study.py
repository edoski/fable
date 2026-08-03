"""Render validation context sensitivity from canonical Studies."""

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

from fable.experiments import ExperimentKind, load_experiment_manifest
from fable.study import load_study


def render(storage_root: Path, experiment_id: UUID, output_directory: Path) -> Path:
    manifest = load_experiment_manifest(storage_root, ExperimentKind.C_STUDY, experiment_id)
    objectives: dict[tuple[str, str, int], float] = {}
    chains: list[str] = []
    families_by_chain: dict[str, list[str]] = defaultdict(list)
    contexts_by_chain: dict[str, set[int]] = defaultdict(set)

    for cell, study_id in manifest.items():
        try:
            chain, family, context_label = cell.split(".")
            context = int(context_label.removeprefix("C"))
        except ValueError as error:
            raise ValueError(f"invalid context-study cell: {cell}") from error
        study = load_study(storage_root, study_id)
        if len(study.trials) != 1:
            raise ValueError("context Studies must contain exactly one trial")
        if chain not in chains:
            chains.append(chain)
        if family not in families_by_chain[chain]:
            families_by_chain[chain].append(family)
        contexts_by_chain[chain].add(context)
        objectives[chain, family, context] = study.trials[0].objective

    figure, axes = subplots(len(chains), 1, height=5.2)
    for row, chain in enumerate(chains):
        axis = axes[row, 0]
        contexts = sorted(contexts_by_chain[chain])
        for family in families_by_chain[chain]:
            missing = [
                context for context in contexts if (chain, family, context) not in objectives
            ]
            if missing:
                raise ValueError(f"{chain}.{family} lacks contexts {missing}")
            color, marker = family_style(family)
            axis.plot(
                contexts,
                [100.0 * objectives[chain, family, context] for context in contexts],
                color=color,
                marker=marker,
                label=display_name(family),
            )
        axis.set_title(display_name(chain), loc="left")
        axis.set_xscale("log", base=2)
        axis.set_xticks(contexts)
        axis.set_xticklabels([str(context) for context in contexts])
        if row < len(chains) - 1:
            axis.tick_params(axis="x", labelbottom=False)
        else:
            axis.set_xlabel("Context blocks C")

    add_family_legend(figure, axes[0, 0])
    figure.supylabel("Cost over optimum (%)", fontsize=8)
    path = save_pdf(figure, output_directory / "context-study.pdf")
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
