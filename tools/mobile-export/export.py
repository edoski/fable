from __future__ import annotations

import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

import torch
import typer
import yaml
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime
from pydantic import UUID4, Field, TypeAdapter
from torch import nn

from fable.config import FeatureName
from fable.corpus import Corpus, load_corpus
from fable.modeling import ArtifactAssociation, load_artifact

_Chain = Literal["ethereum", "polygon", "avalanche"]
_Horizon = Annotated[int, Field(ge=2, le=5)]

_CHAINS: dict[_Chain, int] = {
    "ethereum": 1,
    "polygon": 137,
    "avalanche": 43114,
}
_HORIZONS = (2, 3, 4, 5)

_Roster = Annotated[
    dict[_Chain, Annotated[dict[_Horizon, UUID4], Field(min_length=4)]],
    Field(min_length=3),
]
_ROSTER_ADAPTER = TypeAdapter(_Roster)


@dataclass(frozen=True, slots=True)
class _FeatureContract:
    context_blocks: int
    names: tuple[FeatureName, ...]
    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _Cell:
    horizon: int
    artifact_id: UUID
    features: _FeatureContract
    target_mean: float
    target_standard_deviation: float
    model: nn.Module


class _NamedOutputWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(inputs)
        return output.action_logits, output.minimum_fee_z


def _load_roster(roster_path: Path) -> _Roster:
    raw = yaml.safe_load(roster_path.read_bytes())
    roster = _ROSTER_ADAPTER.validate_json(json.dumps(raw), strict=True)
    artifact_ids = tuple(roster[chain][horizon] for chain in _CHAINS for horizon in _HORIZONS)
    if len(set(artifact_ids)) != len(artifact_ids):
        raise ValueError("roster artifact IDs must be unique")
    return roster


def _feature_contract(
    association: ArtifactAssociation,
    chain: str,
) -> _FeatureContract:
    experiment = association.training_definition.experiment
    names = tuple(experiment.ordered_features)
    if chain != "ethereum" and "log_exact_forming_base_fee_per_gas" in names:
        raise ValueError("exact forming base fee is Ethereum-only")
    return _FeatureContract(
        context_blocks=experiment.context_blocks,
        names=names,
        means=tuple(association.feature_state.means),
        standard_deviations=tuple(association.feature_state.standard_deviations),
    )


def _load_cells(
    storage_root: Path,
    roster: _Roster,
) -> dict[str, dict[int, _Cell]]:
    corpora: dict[UUID, Corpus] = {}
    cells: dict[str, dict[int, _Cell]] = {}
    for chain, chain_id in _CHAINS.items():
        cells[chain] = {}
        shared_features: _FeatureContract | None = None
        for horizon in _HORIZONS:
            artifact_id = roster[chain][horizon]
            association, model = load_artifact(storage_root, artifact_id)

            experiment = association.training_definition.experiment
            if experiment.horizon_blocks != horizon:
                raise ValueError(f"{chain} K={horizon} artifact has the wrong horizon")

            corpus_id = association.request.source.corpus_id
            if corpus_id not in corpora:
                corpora[corpus_id] = load_corpus(storage_root, corpus_id)
            artifact_chain_id = corpora[corpus_id].request.definition.chain_id
            if artifact_chain_id != chain_id:
                raise ValueError(f"{chain} K={horizon} artifact has the wrong chain")

            features = _feature_contract(association, chain)
            if shared_features is None:
                shared_features = features
            elif features != shared_features:
                raise ValueError(f"{chain} artifacts must share one feature contract")

            cells[chain][horizon] = _Cell(
                horizon=horizon,
                artifact_id=artifact_id,
                features=features,
                target_mean=association.target_state.mean,
                target_standard_deviation=association.target_state.standard_deviation,
                model=model,
            )
    return cells


def _example_inputs(features: _FeatureContract) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (1, features.context_blocks, len(features.names))
    zeros = torch.zeros(shape, dtype=torch.float32)
    nonzero = torch.linspace(
        -1.0,
        1.0,
        steps=math.prod(shape),
        dtype=torch.float32,
    ).reshape(shape)
    return zeros, nonzero


def _validated_native_outputs(outputs: object) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 2:
        raise ValueError("ExecuTorch host must return action_logits and minimum_fee_z")
    action_logits, minimum_fee_z = outputs
    if not isinstance(action_logits, torch.Tensor) or not isinstance(minimum_fee_z, torch.Tensor):
        raise ValueError("ExecuTorch host outputs must be tensors")
    return action_logits, minimum_fee_z


def _assert_parity(
    eager: tuple[torch.Tensor, torch.Tensor],
    exported: tuple[torch.Tensor, torch.Tensor],
    *,
    target_mean: float,
    target_standard_deviation: float,
) -> None:
    try:
        torch.testing.assert_close(exported[0], eager[0], atol=1e-5, rtol=1e-3)
        torch.testing.assert_close(exported[1], eager[1], atol=1e-5, rtol=1e-3)
    except AssertionError as error:
        raise ValueError("eager and ExecuTorch outputs do not match") from error

    if not all(torch.isfinite(output).all() for output in exported):
        raise ValueError("ExecuTorch outputs must be finite")
    if exported[0].argmax(dim=-1).item() != eager[0].argmax(dim=-1).item():
        raise ValueError("eager and ExecuTorch selected actions do not match")

    eager_fee = math.exp(target_mean + target_standard_deviation * eager[1].item())
    exported_fee = math.exp(target_mean + target_standard_deviation * exported[1].item())
    if abs(exported_fee - eager_fee) / eager_fee >= 0.001:
        raise ValueError("eager and ExecuTorch decoded fees differ by at least 0.1%")


def _export_model(cell: _Cell, destination: Path) -> None:
    model = _NamedOutputWrapper(cell.model.cpu().float().eval())
    samples = _example_inputs(cell.features)
    with torch.inference_mode():
        eager_outputs = [model(sample) for sample in samples]

    exported = torch.export.export(model, (samples[0],), strict=True)
    program = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()
    destination.write_bytes(program.buffer)

    method = Runtime.get().load_program(destination).load_method("forward")
    for sample, eager in zip(samples, eager_outputs, strict=True):
        host = _validated_native_outputs(method.execute((sample,)))
        _assert_parity(
            eager,
            host,
            target_mean=cell.target_mean,
            target_standard_deviation=cell.target_standard_deviation,
        )


def _manifest(cells: dict[str, dict[int, _Cell]]) -> dict[str, object]:
    chains: dict[str, object] = {}
    for chain in _CHAINS:
        chain_cells = cells[chain]
        features = chain_cells[2].features
        chains[chain] = {
            "context_blocks": features.context_blocks,
            "features": [
                {
                    "name": name,
                    "mean": mean,
                    "standard_deviation": standard_deviation,
                }
                for name, mean, standard_deviation in zip(
                    features.names,
                    features.means,
                    features.standard_deviations,
                    strict=True,
                )
            ],
            "models": {
                str(horizon): {
                    "artifact_id": str(chain_cells[horizon].artifact_id),
                    "target": {
                        "mean": chain_cells[horizon].target_mean,
                        "standard_deviation": (chain_cells[horizon].target_standard_deviation),
                    },
                }
                for horizon in _HORIZONS
            },
        }
    return {"chains": chains}


def export_bundle(
    storage_root: Path,
    roster_path: Path,
    output_directory: Path,
) -> None:
    if output_directory.exists():
        raise FileExistsError(output_directory)

    roster = _load_roster(roster_path)
    cells = _load_cells(storage_root, roster)

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.",
            dir=output_directory.parent,
        )
    )
    try:
        for chain in _CHAINS:
            for horizon in _HORIZONS:
                _export_model(
                    cells[chain][horizon],
                    scratch / f"{chain}-k{horizon}.pte",
                )
        (scratch / "manifest.json").write_text(
            json.dumps(_manifest(cells), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if output_directory.exists():
            raise FileExistsError(output_directory)
        scratch.rename(output_directory)
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def main(
    roster_path: Path,
    output_directory: Path,
    storage_root: Annotated[Path, typer.Option(envvar="STORAGE_ROOT")],
) -> None:
    export_bundle(
        storage_root,
        roster_path,
        output_directory,
    )


if __name__ == "__main__":
    typer.run(main)
