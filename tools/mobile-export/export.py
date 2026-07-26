from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Self
from uuid import UUID

import torch
import yaml
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (  # pyright: ignore[reportMissingImports]
    XnnpackPartitioner,
)
from executorch.exir import to_edge_transform_and_lower  # pyright: ignore[reportMissingImports]
from executorch.runtime import Runtime  # pyright: ignore[reportMissingImports]
from pydantic import UUID4, BaseModel, ConfigDict, model_validator
from torch import nn

from fable.corpus import Corpus, load_corpus
from fable.modeling import ArtifactAssociation, load_artifact

_EXECUTORCH_VERSION = "1.2.0"
_TORCH_VERSION = "2.11.0"
_CHAINS = {
    "ethereum": 1,
    "polygon": 137,
    "avalanche": 43114,
}
_HORIZONS = (2, 3, 4, 5)
_SUPPORTED_FEATURES = frozenset(
    {
        "log_base_fee_per_gas",
        "gas_utilization",
        "log_exact_forming_base_fee_per_gas",
        "log_gas_limit",
        "log1p_tx_count",
        "log1p_effective_priority_fee_per_gas_p50",
        "block_interval_seconds",
        "hour_sin",
        "hour_cos",
    }
)


class _RosterChain(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    k2_artifact_id: UUID4
    k3_artifact_id: UUID4
    k4_artifact_id: UUID4
    k5_artifact_id: UUID4


class _Roster(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    ethereum: _RosterChain
    polygon: _RosterChain
    avalanche: _RosterChain

    @model_validator(mode="after")
    def validate_unique_artifacts(self) -> Self:
        artifact_ids = tuple(
            getattr(getattr(self, chain), f"k{horizon}_artifact_id")
            for chain in _CHAINS
            for horizon in _HORIZONS
        )
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("roster artifact IDs must be unique")
        return self


@dataclass(frozen=True, slots=True)
class _FeatureContract:
    context_blocks: int
    names: tuple[str, ...]
    means: tuple[float, ...]
    standard_deviations: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _Cell:
    chain_id: int
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


def _require_versions() -> None:
    if torch.__version__.split("+", maxsplit=1)[0] != _TORCH_VERSION:
        raise RuntimeError(f"mobile export requires torch=={_TORCH_VERSION}")
    if version("executorch") != _EXECUTORCH_VERSION:
        raise RuntimeError(f"mobile export requires executorch=={_EXECUTORCH_VERSION}")


def _load_roster(roster_path: Path) -> _Roster:
    return _Roster.model_validate_strings(
        yaml.safe_load(roster_path.read_bytes()),
        strict=True,
    )


def _feature_contract(
    association: ArtifactAssociation,
    chain: str,
) -> _FeatureContract:
    experiment = association.training_definition.experiment
    names = tuple(experiment.ordered_features)
    unsupported = set(names) - _SUPPORTED_FEATURES
    if unsupported:
        raise ValueError(f"{chain} artifact contains unsupported features: {sorted(unsupported)}")
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
            artifact_id = getattr(getattr(roster, chain), f"k{horizon}_artifact_id")
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
                chain_id=chain_id,
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


def _validated_native_outputs(
    outputs: object,
    *,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 2:
        raise ValueError("ExecuTorch host must return action_logits and minimum_fee_z")
    action_logits, minimum_fee_z = outputs
    if not isinstance(action_logits, torch.Tensor) or not isinstance(minimum_fee_z, torch.Tensor):
        raise ValueError("ExecuTorch host outputs must be tensors")
    if action_logits.shape != (1, horizon):
        raise ValueError(f"ExecuTorch host action_logits must have shape [1, {horizon}]")
    if minimum_fee_z.shape != (1,):
        raise ValueError("ExecuTorch host minimum_fee_z must have shape [1]")
    if action_logits.dtype != torch.float32 or minimum_fee_z.dtype != torch.float32:
        raise ValueError("ExecuTorch host outputs must be float32")
    if not torch.isfinite(action_logits).all() or not torch.isfinite(minimum_fee_z).all():
        raise ValueError("ExecuTorch host outputs must be finite")
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
        host = _validated_native_outputs(
            method.execute((sample,)),
            horizon=cell.horizon,
        )
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
            "chain_id": chain_cells[2].chain_id,
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
    return {
        "executorch_version": _EXECUTORCH_VERSION,
        "chains": chains,
    }


def export_bundle(
    storage_root: Path,
    roster_path: Path,
    output_directory: Path,
) -> None:
    _require_versions()
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


def _report_sizes(output_directory: Path) -> None:
    total = 0
    for chain in _CHAINS:
        for horizon in _HORIZONS:
            path = output_directory / f"{chain}-k{horizon}.pte"
            size = path.stat().st_size
            total += size
            print(f"{path.name}: {size} bytes")
    print(f"total: {total} bytes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roster_path", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()

    raw_storage_root = os.environ.get("STORAGE_ROOT")
    if not raw_storage_root:
        parser.error("STORAGE_ROOT is required")
    export_bundle(
        Path(raw_storage_root),
        args.roster_path,
        args.output_directory,
    )
    _report_sizes(args.output_directory)


if __name__ == "__main__":
    main()
