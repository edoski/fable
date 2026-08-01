"""Author and close the frozen context-sensitivity experiment."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

from bundle import StorageRoot, close_bundle, load_roster, open_bundle, run, write_tune_cells

from fable.config import TuneRequest
from fable.experiments import ExperimentKind
from fable.study import Study, load_study

_KIND = ExperimentKind.C_STUDY
_CONTEXTS = (25, 50, 100, 200, 400)
_CHAINS = ("ethereum", "polygon", "avalanche")
_FAMILIES = ("lstm", "transformer", "transformer_lstm")


def _full_feature_studies(storage_root: Path, experiment_id: UUID) -> dict[tuple[str, str], Study]:
    roster = load_roster(storage_root, ExperimentKind.FEATURE_ABLATION, experiment_id, "study_id")
    return {
        (chain, family): load_study(storage_root, roster[f"{chain}.{family}.full"])
        for chain in _CHAINS
        for family in _FAMILIES
    }


def prepare(storage_root: StorageRoot, feature_experiment_id: UUID) -> None:
    experiment_id = uuid4()
    selected = _full_feature_studies(storage_root, feature_experiment_id)
    bundle = open_bundle(storage_root, _KIND, experiment_id)

    cells: list[tuple[str, TuneRequest]] = []
    for chain in _CHAINS:
        for family in _FAMILIES:
            source = selected[chain, family]
            method = source.request.methods[0]
            for context in _CONTEXTS:
                request = TuneRequest(
                    corpus_id=source.request.corpus_id,
                    experiment=source.request.experiment.model_copy(
                        update={"context_blocks": context}
                    ),
                    methods=(method,),
                )
                cells.append((f"{chain}.{family}.C{context}", request))

    write_tune_cells(bundle, cells)

    print(experiment_id)


def close(storage_root: StorageRoot, experiment_id: UUID) -> None:
    close_bundle(storage_root, _KIND, experiment_id, "study_id", load_study)


if __name__ == "__main__":
    run(prepare, close)
