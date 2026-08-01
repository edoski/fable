"""Run the post-evaluation CPU inference-latency experiment."""

from __future__ import annotations

import hashlib
import math
import os
import platform
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, cast
from uuid import UUID, uuid4

import polars as pl
import torch
import typer
from pydantic import UUID4, Field
from torch import nn

from fable.addresses import (
    artifact_checkpoint_path,
    evaluation_json_path,
    evaluation_observations_path,
)
from fable.config import BlockWindow, EvaluateRequest, FeatureName, Method
from fable.corpus import load_corpus_blocks, load_corpus_request
from fable.evaluation import OBSERVATION_SCHEMA
from fable.experiments import ExperimentKind, load_experiment_manifest
from fable.min_block_fee import MinBlockFeeOutput, decode_action
from fable.modeling import ArtifactAssociation, load_artifact
from fable.records import StrictFrozenRecord
from fable.study import load_study
from fable.temporal import HistoricalDataset, prepare_historical_window

_CHAINS = ("ethereum", "polygon", "avalanche")
_CHAIN_IDS = {"ethereum": 1, "polygon": 137, "avalanche": 43_114}
_FAMILIES = ("lstm", "transformer", "transformer_lstm")
_HORIZONS = (2, 3, 4, 5)
_CASCADE_HORIZONS = (5, 4, 3, 2)
_WORKLOADS = ("standalone_k2", "standalone_k3", "standalone_k4", "standalone_k5", "cascade")
_ORIGIN_RULE = "full held-out windows; same unchanged K5 origin for K5->K4->K3->K2 cascade"
_WARMUP_RULE = "repeat the first testing origin once per resident horizon before full parity"
_ORDER_RULE = "left-rotate the declared cell and workload orders by zero-based sweep index"
_QUANTILE_METHOD = "inverted_cdf"
_ROOT = Path(__file__).parents[1]

_LatencyPurpose = Literal["pilot", "main"]
_Clock = Callable[[], int]


class ArtifactProvenance(StrictFrozenRecord):
    cell: str
    chain: str
    family: str
    horizon_blocks: int
    artifact_id: UUID4
    evaluation_id: UUID4
    corpus_id: UUID4
    study_id: UUID4
    study_result_index: int
    artifact_sha256: str
    testing_window: BlockWindow
    context_blocks: int
    ordered_features: tuple[FeatureName, ...]
    method: Method


class HostMetadata(StrictFrozenRecord):
    hardware_model: str
    hardware_identifier: str
    machine: str
    macos_version: str
    macos_build: str
    power_source: Literal["AC Power"]
    power_mode: Literal["automatic"]
    low_power_mode: Literal[False]
    intraop_threads: int
    interop_threads: int


class LatencyProtocol(StrictFrozenRecord):
    schema_version: Literal[1] = 1
    campaign_id: UUID4
    purpose: _LatencyPurpose
    created_at_utc: datetime
    repository_commit: str
    python_version: str
    pytorch_version: str
    host: HostMetadata
    k_study_experiment_id: UUID4
    held_out_experiment_id: UUID4
    artifacts: tuple[ArtifactProvenance, ...]
    origin_rule: Literal[
        "full held-out windows; same unchanged K5 origin for K5->K4->K3->K2 cascade"
    ]
    warmup_rule: Literal[
        "repeat the first testing origin once per resident horizon before full parity"
    ]
    order_rule: Literal[
        "left-rotate the declared cell and workload orders by zero-based sweep index"
    ]
    warmup_iterations: Annotated[int, Field(ge=1)]
    prediction_atol: Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
    quantile_method: Literal["inverted_cdf"]
    latency_sweeps: Annotated[int, Field(ge=1)]


_LATENCY_SCHEMA = pl.Schema(
    {
        "campaign_id": pl.String,
        "protocol_sha256": pl.String,
        "cell": pl.String,
        "sweep": pl.Int64,
        "cell_order": pl.Int64,
        "pass_order": pl.Int64,
        "execution_order": pl.Int64,
        "workload": pl.String,
        "origin_block": pl.Int64,
        "horizon_blocks": pl.List(pl.Int64),
        "artifact_ids": pl.List(pl.String),
        "artifact_sha256s": pl.List(pl.String),
        "predicted_action_ks": pl.List(pl.Int64),
        "predicted_minimum_log_base_fees": pl.List(pl.Float64),
        "elapsed_ns": pl.Int64,
    }
)

_PARITY_SCHEMA = pl.Schema(
    {
        "campaign_id": pl.String,
        "protocol_sha256": pl.String,
        "cell": pl.String,
        "sweep": pl.Int64,
        "horizon_blocks": pl.Int64,
        "origin_block": pl.Int64,
        "effective_origin_block": pl.Int64,
        "artifact_id": pl.String,
        "artifact_sha256": pl.String,
        "predicted_action_k": pl.Int64,
        "predicted_minimum_log_base_fee": pl.Float64,
        "canonical_action_k": pl.Int64,
        "canonical_minimum_log_base_fee": pl.Float64,
    }
)


@dataclass(frozen=True, slots=True)
class _Reference:
    origins: tuple[int, ...]
    actions: tuple[int, ...]
    minimum_logs: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _LoadedHorizon:
    provenance: ArtifactProvenance
    association: ArtifactAssociation
    model: nn.Module
    dataset: HistoricalDataset
    reference: _Reference


@dataclass(frozen=True, slots=True)
class _LoadedCell:
    name: str
    horizons: Mapping[int, _LoadedHorizon]


def _approved_cells() -> tuple[str, ...]:
    return tuple(
        f"{chain}.{family}.K{horizon}"
        for chain in _CHAINS
        for family in _FAMILIES
        for horizon in _HORIZONS
    )


def _base_cells() -> tuple[str, ...]:
    return tuple(f"{chain}.{family}" for chain in _CHAINS for family in _FAMILIES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_observations(path: Path, request: EvaluateRequest, horizon: int) -> _Reference:
    observations = pl.read_parquet(path)
    if observations.schema != OBSERVATION_SCHEMA:
        raise ValueError(f"{path} must have the canonical ordered observation schema")
    if observations.null_count().row(0) != (0,) * len(OBSERVATION_SCHEMA):
        raise ValueError(f"{path} must not contain null observations")
    origins = tuple(cast(list[int], observations["origin_block"].to_list()))
    expected_origins = tuple(
        range(
            request.testing_window.first_parent_block, request.testing_window.last_parent_block + 1
        )
    )
    if origins != expected_origins:
        raise ValueError(f"{path} must cover the exact chronological testing window")
    actions = tuple(cast(list[int], observations["predicted_action_k"].to_list()))
    if any(action < 0 or action >= horizon for action in actions):
        raise ValueError(f"{path} contains an invalid predicted action")
    logs = tuple(cast(list[float], observations["predicted_minimum_log_base_fee"].to_list()))
    if not all(math.isfinite(value) for value in logs):
        raise ValueError(f"{path} contains a non-finite predicted minimum-log fee")
    return _Reference(origins=origins, actions=actions, minimum_logs=logs)


def _validate_artifact(
    storage_root: Path, *, cell: str, artifact_id: UUID, evaluation_id: UUID
) -> ArtifactProvenance:
    chain, family, horizon_label = cell.split(".")
    horizon = int(horizon_label.removeprefix("K"))
    request = EvaluateRequest.model_validate_json(
        evaluation_json_path(storage_root, evaluation_id).read_bytes(), strict=True
    )
    if request.evaluation_id != evaluation_id:
        raise ValueError(f"{cell} evaluation ID does not match its manifest")
    if request.artifact_id != artifact_id:
        raise ValueError(f"{cell} evaluation artifact does not match the K-study manifest")

    association, model = load_artifact(storage_root, artifact_id)
    del model
    source = association.request.source
    experiment = source.experiment
    if source.corpus_id != request.corpus_id:
        raise ValueError(f"{cell} artifact and evaluation must use the same Corpus")
    if experiment.horizon_blocks != horizon:
        raise ValueError(f"{cell} artifact horizon does not match its cell")
    if association.method.model.family != family:
        raise ValueError(f"{cell} artifact family does not match its cell")

    corpus = load_corpus_request(storage_root, request.corpus_id)
    if corpus.definition.chain_id != _CHAIN_IDS[chain]:
        raise ValueError(f"{cell} Corpus chain does not match its cell")
    study = load_study(storage_root, source.study_id)
    if study.request.corpus_id != source.corpus_id:
        raise ValueError(f"{cell} selected Study and artifact must use the same Corpus")
    if source.study_result_index != study.best_result()[0]:
        raise ValueError(f"{cell} artifact must use the selected Study result")
    if association.method != study.request.method_at(source.study_result_index):
        raise ValueError(f"{cell} embedded Method does not match the selected Study result")
    expected_experiment = study.request.experiment.model_copy(update={"horizon_blocks": horizon})
    if experiment != expected_experiment:
        raise ValueError(f"{cell} artifact experiment does not match its selected Study")

    _strict_observations(
        evaluation_observations_path(storage_root, evaluation_id), request, horizon
    )
    checkpoint = artifact_checkpoint_path(storage_root, artifact_id)
    return ArtifactProvenance(
        cell=cell,
        chain=chain,
        family=family,
        horizon_blocks=horizon,
        artifact_id=artifact_id,
        evaluation_id=evaluation_id,
        corpus_id=request.corpus_id,
        study_id=source.study_id,
        study_result_index=source.study_result_index,
        artifact_sha256=_sha256(checkpoint),
        testing_window=request.testing_window,
        context_blocks=experiment.context_blocks,
        ordered_features=experiment.ordered_features,
        method=association.method,
    )


def _validate_cell_geometry(records: Sequence[ArtifactProvenance]) -> None:
    by_cell: dict[str, list[ArtifactProvenance]] = {}
    for record in records:
        by_cell.setdefault(record.cell.rsplit(".", maxsplit=1)[0], []).append(record)
    for cell, cell_records in by_cell.items():
        by_horizon = {record.horizon_blocks: record for record in cell_records}
        if set(by_horizon) != set(_HORIZONS):
            raise ValueError(f"{cell} must contain K2 through K5 exactly once")
        k5 = by_horizon[5]
        for horizon, record in by_horizon.items():
            if record.corpus_id != k5.corpus_id:
                raise ValueError(f"{cell} horizons must share one Corpus")
            if record.testing_window.first_parent_block != k5.testing_window.first_parent_block:
                raise ValueError(f"{cell} horizons must share the first testing origin")
            if record.testing_window.last_parent_block != (
                k5.testing_window.last_parent_block + 5 - horizon
            ):
                raise ValueError(f"{cell} testing windows do not have canonical K geometry")
            if record.context_blocks != k5.context_blocks:
                raise ValueError(f"{cell} horizons must share context length")
            if record.ordered_features != k5.ordered_features:
                raise ValueError(f"{cell} horizons must share ordered features")
            if (
                record.study_id != k5.study_id
                or record.study_result_index != k5.study_result_index
                or record.method != k5.method
            ):
                raise ValueError(f"{cell} horizons must share one selected Study Method")


def _resolve_campaign(
    storage_root: Path, k_study_experiment_id: UUID, held_out_experiment_id: UUID
) -> tuple[ArtifactProvenance, ...]:
    k_manifest = load_experiment_manifest(
        storage_root, ExperimentKind.K_STUDY, k_study_experiment_id
    )
    held_out = load_experiment_manifest(
        storage_root, ExperimentKind.HELD_OUT, held_out_experiment_id
    )
    approved = _approved_cells()
    missing_k = set(approved) - k_manifest.keys()
    missing_held_out = set(approved) - held_out.keys()
    if missing_k or missing_held_out:
        raise ValueError(
            "completed manifests must contain the exact approved K2-K5 roster; "
            f"K-study missing={sorted(missing_k)}, held-out missing={sorted(missing_held_out)}"
        )
    if len({k_manifest[cell] for cell in approved}) != len(approved):
        raise ValueError("approved K-study cells must identify 36 distinct artifacts")
    if len({held_out[cell] for cell in approved}) != len(approved):
        raise ValueError("approved held-out cells must identify 36 distinct evaluations")

    records = tuple(
        _validate_artifact(
            storage_root, cell=cell, artifact_id=k_manifest[cell], evaluation_id=held_out[cell]
        )
        for cell in approved
    )
    _validate_cell_geometry(records)
    return records


def _command(*arguments: str) -> str:
    result = subprocess.run(arguments, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _host_metadata() -> HostMetadata:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("CPU benchmark requires the declared Apple Silicon reference Mac")
    hardware_model = _command("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string")
    if hardware_model != "Apple M2 Max":
        raise RuntimeError(f"CPU benchmark requires Apple M2 Max, got {hardware_model!r}")
    power_source = _command("/usr/bin/pmset", "-g", "batt")
    if "drawing from 'AC Power'" not in power_source:
        raise RuntimeError("CPU benchmark requires AC power")
    power_configuration = _command("/usr/bin/pmset", "-g", "custom")
    ac_configuration = re.search(r"^AC Power:\s*$([\s\S]*)", power_configuration, re.M)
    low_power_values = (
        []
        if ac_configuration is None
        else re.findall(r"^\s*lowpowermode\s+(\d+)\s*$", ac_configuration.group(1), re.M)
    )
    if low_power_values != ["0"]:
        raise RuntimeError("CPU benchmark requires Low Power Mode disabled")
    return HostMetadata(
        hardware_model=hardware_model,
        hardware_identifier=_command("/usr/sbin/sysctl", "-n", "hw.model"),
        machine=platform.machine(),
        macos_version=platform.mac_ver()[0],
        macos_build=_command("/usr/bin/sw_vers", "-buildVersion"),
        power_source="AC Power",
        power_mode="automatic",
        low_power_mode=False,
        intraop_threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
    )


def _repository_commit() -> str:
    return _command("/usr/bin/git", "-C", str(_ROOT), "rev-parse", "HEAD")


def _protocol_digest(protocol: LatencyProtocol) -> str:
    return hashlib.sha256(protocol.model_dump_json().encode()).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_bytes(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4()}.tmp")
    try:
        with temporary.open("xb") as destination:
            destination.write(contents)
            destination.flush()
            os.fsync(destination.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_frame(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4()}.tmp")
    try:
        frame.write_parquet(temporary)
        with temporary.open("rb") as source:
            os.fsync(source.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_protocol(
    output: Path,
    *,
    purpose: _LatencyPurpose,
    k_study_experiment_id: UUID,
    held_out_experiment_id: UUID,
    artifacts: tuple[ArtifactProvenance, ...],
    warmup_iterations: int,
    prediction_atol: float,
    latency_sweeps: int,
    host: HostMetadata,
) -> LatencyProtocol:
    if purpose == "main" and latency_sweeps < 10:
        raise ValueError("main campaigns require at least ten latency sweeps")
    if output.exists() and not output.is_dir():
        raise ValueError("output must be a directory")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "protocol.json"
    existing: LatencyProtocol | None = None
    if path.exists():
        existing = LatencyProtocol.model_validate_json(path.read_bytes(), strict=True)
        campaign_id = existing.campaign_id
        created_at = existing.created_at_utc
    else:
        if any(output.iterdir()):
            raise ValueError("a new output directory must be empty")
        campaign_id = uuid4()
        created_at = datetime.now(UTC)
    proposed = LatencyProtocol(
        campaign_id=campaign_id,
        purpose=purpose,
        created_at_utc=created_at,
        repository_commit=_repository_commit(),
        python_version=platform.python_version(),
        pytorch_version=torch.__version__,
        host=host,
        k_study_experiment_id=k_study_experiment_id,
        held_out_experiment_id=held_out_experiment_id,
        artifacts=artifacts,
        origin_rule=_ORIGIN_RULE,
        warmup_rule=_WARMUP_RULE,
        order_rule=_ORDER_RULE,
        warmup_iterations=warmup_iterations,
        prediction_atol=prediction_atol,
        quantile_method=_QUANTILE_METHOD,
        latency_sweeps=latency_sweeps,
    )
    if existing is not None:
        if existing != proposed:
            raise ValueError("existing output protocol does not match this invocation or runtime")
        return existing
    _publish_bytes(path, proposed.model_dump_json(indent=2).encode())
    return proposed


def _assert_model_cpu_eval(model: nn.Module, cell: str) -> None:
    if model.training:
        raise ValueError(f"{cell} model must be in evaluation mode")
    devices = {value.device.type for value in (*model.parameters(), *model.buffers())}
    if devices - {"cpu"}:
        raise ValueError(f"{cell} model must be resident only on CPU")


def _load_cell(storage_root: Path, records: Sequence[ArtifactProvenance]) -> _LoadedCell:
    names = {record.cell.rsplit(".", maxsplit=1)[0] for record in records}
    if len(names) != 1:
        raise ValueError("one loaded cell must contain exactly one architecture-chain pair")
    cell = names.pop()
    if len(records) != 4:
        raise ValueError(f"{cell} must load exactly four horizon models")
    corpus_ids = {record.corpus_id for record in records}
    if len(corpus_ids) != 1:
        raise ValueError(f"{cell} must use one Corpus")
    blocks = load_corpus_blocks(storage_root, corpus_ids.pop())

    loaded: dict[int, _LoadedHorizon] = {}
    for record in sorted(records, key=lambda candidate: candidate.horizon_blocks):
        association, model = load_artifact(storage_root, record.artifact_id)
        _assert_model_cpu_eval(model, record.cell)
        if _sha256(artifact_checkpoint_path(storage_root, record.artifact_id)) != (
            record.artifact_sha256
        ):
            raise ValueError(f"{record.cell} artifact bytes changed after protocol resolution")
        if association.request.artifact_id != record.artifact_id:
            raise ValueError(f"{record.cell} loaded artifact identity changed after resolution")
        if association.request.source.corpus_id != record.corpus_id:
            raise ValueError(f"{record.cell} loaded Corpus association changed after resolution")
        if association.method != record.method:
            raise ValueError(f"{record.cell} loaded Method changed after resolution")
        experiment = association.training_definition.experiment
        if (
            experiment.horizon_blocks != record.horizon_blocks
            or experiment.context_blocks != record.context_blocks
            or experiment.ordered_features != record.ordered_features
        ):
            raise ValueError(f"{record.cell} loaded experiment changed after resolution")
        dataset = prepare_historical_window(
            blocks,
            experiment,
            record.testing_window,
            feature_state=association.feature_state,
            target_state=association.target_state,
        )
        evaluation = EvaluateRequest.model_validate_json(
            evaluation_json_path(storage_root, record.evaluation_id).read_bytes(), strict=True
        )
        if (
            evaluation.evaluation_id != record.evaluation_id
            or evaluation.artifact_id != record.artifact_id
            or evaluation.corpus_id != record.corpus_id
            or evaluation.testing_window != record.testing_window
        ):
            raise ValueError(f"{record.cell} evaluation changed after protocol resolution")
        reference = _strict_observations(
            evaluation_observations_path(storage_root, record.evaluation_id),
            evaluation,
            record.horizon_blocks,
        )
        if len(dataset) != len(reference.origins):
            raise ValueError(f"{record.cell} dataset and canonical observations must align")
        loaded[record.horizon_blocks] = _LoadedHorizon(
            provenance=record,
            association=association,
            model=model,
            dataset=dataset,
            reference=reference,
        )
    if set(loaded) != set(_HORIZONS):
        raise ValueError(f"{cell} must keep exactly K2 through K5 resident")
    return _LoadedCell(name=cell, horizons=loaded)


def _batch_one(dataset: HistoricalDataset, index: int, expected_origin: int) -> torch.Tensor:
    item = dataset[index]
    if int(item["origin_block"]) != expected_origin:
        raise ValueError("dataset origin does not match chronological traversal")
    inputs = item["inputs"].unsqueeze(0)
    if inputs.shape[0] != 1:
        raise ValueError("latency benchmark requires true batch size one")
    return inputs


def _decoded(
    output: MinBlockFeeOutput,
    association: ArtifactAssociation,
    action_tensor: torch.Tensor | None = None,
) -> tuple[int, float]:
    if output.action_logits.shape[0] != 1 or output.minimum_fee_z.shape != (1,):
        raise ValueError("model output must describe exactly one batch item")
    action = int((decode_action(output) if action_tensor is None else action_tensor).item())
    minimum_z = float(output.minimum_fee_z.item())
    minimum_log = (
        association.target_state.mean + association.target_state.standard_deviation * minimum_z
    )
    if not math.isfinite(minimum_log):
        raise ValueError("predicted minimum-log fee must be finite")
    return action, minimum_log


def _warm(cell: _LoadedCell, iterations: int) -> None:
    with torch.inference_mode():
        for horizon in _HORIZONS:
            loaded = cell.horizons[horizon]
            origin = loaded.reference.origins[0]
            inputs = _batch_one(loaded.dataset, 0, origin)
            for _ in range(iterations):
                _decoded(cast(MinBlockFeeOutput, loaded.model(inputs)), loaded.association)


def _parity_rows(
    cell: _LoadedCell, protocol: LatencyProtocol, protocol_sha256: str, sweep: int
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for horizon in _HORIZONS:
            loaded = cell.horizons[horizon]
            for index, (origin, canonical_action, canonical_log) in enumerate(
                zip(
                    loaded.reference.origins,
                    loaded.reference.actions,
                    loaded.reference.minimum_logs,
                    strict=True,
                )
            ):
                inputs = _batch_one(loaded.dataset, index, origin)
                action, minimum_log = _decoded(
                    cast(MinBlockFeeOutput, loaded.model(inputs)), loaded.association
                )
                if action != canonical_action:
                    raise ValueError(
                        f"{loaded.provenance.cell} origin {origin} action parity failed: "
                        f"CPU={action}, canonical={canonical_action}"
                    )
                if not math.isclose(
                    minimum_log, canonical_log, rel_tol=0.0, abs_tol=protocol.prediction_atol
                ):
                    raise ValueError(
                        f"{loaded.provenance.cell} origin {origin} minimum-log parity failed"
                    )
                rows.append(
                    {
                        "campaign_id": str(protocol.campaign_id),
                        "protocol_sha256": protocol_sha256,
                        "cell": cell.name,
                        "sweep": sweep,
                        "horizon_blocks": horizon,
                        "origin_block": origin,
                        "effective_origin_block": origin,
                        "artifact_id": str(loaded.provenance.artifact_id),
                        "artifact_sha256": loaded.provenance.artifact_sha256,
                        "predicted_action_k": action,
                        "predicted_minimum_log_base_fee": minimum_log,
                        "canonical_action_k": canonical_action,
                        "canonical_minimum_log_base_fee": canonical_log,
                    }
                )
    return rows


def _rotation(values: Sequence[str], offset: int) -> tuple[str, ...]:
    split = offset % len(values)
    return tuple((*values[split:], *values[:split]))


def _pass_order(sweep: int) -> tuple[str, ...]:
    return _rotation(_WORKLOADS, sweep - 1)


def _cell_order(sweep: int) -> tuple[str, ...]:
    return _rotation(_base_cells(), sweep - 1)


def _time_cell(
    cell: _LoadedCell, protocol: LatencyProtocol, protocol_sha256: str, sweep: int, *, clock: _Clock
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    execution_order = 0
    with torch.inference_mode():
        for pass_order, workload in enumerate(_pass_order(sweep)):
            if workload.startswith("standalone_k"):
                horizon = int(workload.removeprefix("standalone_k"))
                loaded = cell.horizons[horizon]
                for index, origin in enumerate(loaded.reference.origins):
                    inputs = _batch_one(loaded.dataset, index, origin)
                    start = clock()
                    output = cast(MinBlockFeeOutput, loaded.model(inputs))
                    action_tensor = decode_action(output)
                    end = clock()
                    action, minimum_log = _decoded(
                        MinBlockFeeOutput(output.action_logits, output.minimum_fee_z),
                        loaded.association,
                        action_tensor,
                    )
                    if int(action_tensor.item()) != action:
                        raise RuntimeError("decoded action changed inside one CPU call")
                    elapsed = end - start
                    if elapsed <= 0:
                        raise RuntimeError("latency clock must advance for every call")
                    rows.append(
                        _latency_row(
                            protocol,
                            protocol_sha256,
                            cell=cell.name,
                            sweep=sweep,
                            pass_order=pass_order,
                            execution_order=execution_order,
                            workload=workload,
                            origin=origin,
                            loaded=(loaded,),
                            actions=(action,),
                            minimum_logs=(minimum_log,),
                            elapsed=elapsed,
                        )
                    )
                    execution_order += 1
            else:
                roots = cell.horizons[5].reference.origins
                for index, origin in enumerate(roots):
                    loaded_horizons = tuple(cell.horizons[horizon] for horizon in _CASCADE_HORIZONS)
                    inputs = tuple(
                        _batch_one(loaded.dataset, index, origin) for loaded in loaded_horizons
                    )
                    outputs: list[tuple[torch.Tensor, MinBlockFeeOutput]] = []
                    start = clock()
                    for loaded, batch in zip(loaded_horizons, inputs, strict=True):
                        output = cast(MinBlockFeeOutput, loaded.model(batch))
                        outputs.append((decode_action(output), output))
                    end = clock()
                    decoded = tuple(
                        _decoded(output, loaded.association, action_tensor)
                        for loaded, (action_tensor, output) in zip(
                            loaded_horizons, outputs, strict=True
                        )
                    )
                    actions = tuple(value[0] for value in decoded)
                    if any(
                        int(action_tensor.item()) != action
                        for (action_tensor, _), action in zip(outputs, actions, strict=True)
                    ):
                        raise RuntimeError("decoded action changed inside one CPU cascade")
                    elapsed = end - start
                    if elapsed <= 0:
                        raise RuntimeError("latency clock must advance for every cascade")
                    rows.append(
                        _latency_row(
                            protocol,
                            protocol_sha256,
                            cell=cell.name,
                            sweep=sweep,
                            pass_order=pass_order,
                            execution_order=execution_order,
                            workload="cascade",
                            origin=origin,
                            loaded=loaded_horizons,
                            actions=actions,
                            minimum_logs=tuple(value[1] for value in decoded),
                            elapsed=elapsed,
                        )
                    )
                    execution_order += 1
            _assert_runtime(protocol)
    return rows


def _latency_row(
    protocol: LatencyProtocol,
    protocol_sha256: str,
    *,
    cell: str,
    sweep: int,
    pass_order: int,
    execution_order: int,
    workload: str,
    origin: int,
    loaded: Sequence[_LoadedHorizon],
    actions: Sequence[int],
    minimum_logs: Sequence[float],
    elapsed: int,
) -> dict[str, object]:
    return {
        "campaign_id": str(protocol.campaign_id),
        "protocol_sha256": protocol_sha256,
        "cell": cell,
        "sweep": sweep,
        "cell_order": _cell_order(sweep).index(cell),
        "pass_order": pass_order,
        "execution_order": execution_order,
        "workload": workload,
        "origin_block": origin,
        "horizon_blocks": [item.provenance.horizon_blocks for item in loaded],
        "artifact_ids": [str(item.provenance.artifact_id) for item in loaded],
        "artifact_sha256s": [item.provenance.artifact_sha256 for item in loaded],
        "predicted_action_ks": list(actions),
        "predicted_minimum_log_base_fees": list(minimum_logs),
        "elapsed_ns": elapsed,
    }


def _assert_runtime(protocol: LatencyProtocol) -> None:
    if torch.get_num_threads() != protocol.host.intraop_threads:
        raise RuntimeError("PyTorch intra-op thread count changed during the campaign")
    if torch.get_num_interop_threads() != protocol.host.interop_threads:
        raise RuntimeError("PyTorch inter-op thread count changed during the campaign")
    if _repository_commit() != protocol.repository_commit:
        raise RuntimeError("repository HEAD changed during the campaign")


def _records_for_cell(
    records: Sequence[ArtifactProvenance], cell: str
) -> tuple[ArtifactProvenance, ...]:
    selected = tuple(record for record in records if record.cell.rsplit(".", maxsplit=1)[0] == cell)
    if len(selected) != 4:
        raise ValueError(f"{cell} protocol roster must contain exactly four artifacts")
    return selected


def _unit_paths(output: Path, cell: str, sweep: int) -> tuple[Path, Path]:
    directory = output / "latency" / cell
    return directory / f"parity-{sweep:03d}.parquet", directory / f"sweep-{sweep:03d}.parquet"


def _validate_parity(
    path: Path,
    protocol: LatencyProtocol,
    protocol_sha256: str,
    records: Sequence[ArtifactProvenance],
    cell: str,
    sweep: int,
) -> None:
    frame = pl.read_parquet(path)
    if frame.schema != _PARITY_SCHEMA:
        raise ValueError(f"{path} has an invalid parity schema")
    if frame.null_count().row(0) != (0,) * len(_PARITY_SCHEMA):
        raise ValueError(f"{path} contains null parity values")
    expected_count = sum(
        record.testing_window.last_parent_block - record.testing_window.first_parent_block + 1
        for record in records
    )
    if frame.height != expected_count:
        raise ValueError(f"{path} does not contain complete parity coverage")
    if set(frame["campaign_id"].to_list()) != {str(protocol.campaign_id)}:
        raise ValueError(f"{path} campaign identity does not match protocol")
    if set(frame["protocol_sha256"].to_list()) != {protocol_sha256}:
        raise ValueError(f"{path} protocol digest does not match protocol")
    if set(frame["cell"].to_list()) != {cell} or set(frame["sweep"].to_list()) != {sweep}:
        raise ValueError(f"{path} cell or sweep does not match its address")
    for record in records:
        rows = frame.filter(pl.col("horizon_blocks") == record.horizon_blocks)
        expected_origins = list(
            range(
                record.testing_window.first_parent_block,
                record.testing_window.last_parent_block + 1,
            )
        )
        if rows["origin_block"].to_list() != expected_origins:
            raise ValueError(f"{path} parity origins are incomplete or unordered")
        if rows["effective_origin_block"].to_list() != expected_origins:
            raise ValueError(f"{path} parity effective origins must equal CPU origins")
        if set(rows["artifact_id"].to_list()) != {str(record.artifact_id)}:
            raise ValueError(f"{path} parity artifact identity does not match protocol")
        if set(rows["artifact_sha256"].to_list()) != {record.artifact_sha256}:
            raise ValueError(f"{path} parity artifact digest does not match protocol")
        actions = cast(list[int], rows["predicted_action_k"].to_list())
        if any(action < 0 or action >= record.horizon_blocks for action in actions):
            raise ValueError(f"{path} contains invalid parity actions")
        predictions = cast(list[float], rows["predicted_minimum_log_base_fee"].to_list())
        if not all(math.isfinite(value) for value in predictions):
            raise ValueError(f"{path} contains non-finite parity predictions")
        errors = (
            rows["predicted_minimum_log_base_fee"] - rows["canonical_minimum_log_base_fee"]
        ).abs()
        if (rows["predicted_action_k"] != rows["canonical_action_k"]).any() or (
            errors > protocol.prediction_atol
        ).any():
            raise ValueError(f"{path} contains failed parity rows")


def _validate_latency(
    path: Path,
    protocol: LatencyProtocol,
    protocol_sha256: str,
    records: Sequence[ArtifactProvenance],
    cell: str,
    sweep: int,
) -> None:
    frame = pl.read_parquet(path)
    if frame.schema != _LATENCY_SCHEMA:
        raise ValueError(f"{path} has an invalid latency schema")
    if frame.null_count().row(0) != (0,) * len(_LATENCY_SCHEMA):
        raise ValueError(f"{path} contains null latency values")
    by_horizon = {record.horizon_blocks: record for record in records}
    expected_count = sum(
        record.testing_window.last_parent_block - record.testing_window.first_parent_block + 1
        for record in records
    ) + (
        by_horizon[5].testing_window.last_parent_block
        - by_horizon[5].testing_window.first_parent_block
        + 1
    )
    if frame.height != expected_count:
        raise ValueError(f"{path} does not contain a complete latency sweep")
    if frame["execution_order"].to_list() != list(range(expected_count)):
        raise ValueError(f"{path} execution order must be complete and unique")
    if (frame["elapsed_ns"] <= 0).any():
        raise ValueError(f"{path} contains a non-positive latency")
    if set(frame["campaign_id"].to_list()) != {str(protocol.campaign_id)}:
        raise ValueError(f"{path} campaign identity does not match protocol")
    if set(frame["protocol_sha256"].to_list()) != {protocol_sha256}:
        raise ValueError(f"{path} protocol digest does not match protocol")
    if set(frame["cell"].to_list()) != {cell} or set(frame["sweep"].to_list()) != {sweep}:
        raise ValueError(f"{path} cell or sweep does not match its address")
    expected_cell_order = _cell_order(sweep).index(cell)
    if set(frame["cell_order"].to_list()) != {expected_cell_order}:
        raise ValueError(f"{path} cell order does not match the frozen rotation")

    pass_order = _pass_order(sweep)
    if set(frame["workload"].to_list()) != set(_WORKLOADS):
        raise ValueError(f"{path} workloads are incomplete")
    for order, workload in enumerate(pass_order):
        rows = frame.filter(pl.col("workload") == workload)
        if set(rows["pass_order"].to_list()) != {order}:
            raise ValueError(f"{path} pass order does not match the frozen rotation")
        if workload == "cascade":
            record = by_horizon[5]
            horizons = list(_CASCADE_HORIZONS)
            selected = [by_horizon[horizon] for horizon in _CASCADE_HORIZONS]
        else:
            horizon = int(workload.removeprefix("standalone_k"))
            record = by_horizon[horizon]
            horizons = [horizon]
            selected = [record]
        expected_origins = list(
            range(
                record.testing_window.first_parent_block,
                record.testing_window.last_parent_block + 1,
            )
        )
        if rows["origin_block"].to_list() != expected_origins:
            raise ValueError(f"{path} {workload} origins are incomplete or unordered")
        if rows["horizon_blocks"].to_list() != [horizons] * len(expected_origins):
            raise ValueError(f"{path} {workload} horizon identities do not match protocol")
        expected_ids = [str(item.artifact_id) for item in selected]
        expected_hashes = [item.artifact_sha256 for item in selected]
        if rows["artifact_ids"].to_list() != [expected_ids] * len(expected_origins):
            raise ValueError(f"{path} {workload} artifact identities do not match protocol")
        if rows["artifact_sha256s"].to_list() != [expected_hashes] * len(expected_origins):
            raise ValueError(f"{path} {workload} artifact digests do not match protocol")
        widths = rows.select(
            pl.col("predicted_action_ks").list.len().alias("actions"),
            pl.col("predicted_minimum_log_base_fees").list.len().alias("logs"),
        )
        if set(widths["actions"].to_list()) != {len(horizons)} or set(widths["logs"].to_list()) != {
            len(horizons)
        }:
            raise ValueError(f"{path} {workload} decoded values are incomplete")
        actions = cast(list[list[int]], rows["predicted_action_ks"].to_list())
        if any(
            action < 0 or action >= horizon
            for row in actions
            for action, horizon in zip(row, horizons, strict=True)
        ):
            raise ValueError(f"{path} {workload} contains invalid decoded actions")
        logs = cast(list[list[float]], rows["predicted_minimum_log_base_fees"].to_list())
        if not all(math.isfinite(value) for row in logs for value in row):
            raise ValueError(f"{path} {workload} contains non-finite predictions")


def _run_unit(
    storage_root: Path,
    output: Path,
    protocol: LatencyProtocol,
    records: Sequence[ArtifactProvenance],
    cell_name: str,
    sweep: int,
    *,
    clock: _Clock,
) -> None:
    protocol_sha256 = _protocol_digest(protocol)
    parity_path, latency_path = _unit_paths(output, cell_name, sweep)
    if latency_path.exists():
        if not parity_path.exists():
            raise ValueError(f"{latency_path} exists without its parity record")
        _validate_parity(parity_path, protocol, protocol_sha256, records, cell_name, sweep)
        _validate_latency(latency_path, protocol, protocol_sha256, records, cell_name, sweep)
        return

    loaded = _load_cell(storage_root, records)
    _assert_runtime(protocol)
    _warm(loaded, protocol.warmup_iterations)
    parity_rows = _parity_rows(loaded, protocol, protocol_sha256, sweep)
    parity_frame = pl.DataFrame(parity_rows, schema=_PARITY_SCHEMA)
    if parity_path.exists():
        _validate_parity(parity_path, protocol, protocol_sha256, records, cell_name, sweep)
    else:
        _publish_frame(parity_path, parity_frame)
    latency_rows = _time_cell(loaded, protocol, protocol_sha256, sweep, clock=clock)
    _assert_runtime(protocol)
    latency_frame = pl.DataFrame(latency_rows, schema=_LATENCY_SCHEMA)
    _publish_frame(latency_path, latency_frame)
    _validate_latency(latency_path, protocol, protocol_sha256, records, cell_name, sweep)


def run_cpu(
    storage_root: Path,
    k_study_experiment_id: UUID,
    held_out_experiment_id: UUID,
    output: Path,
    purpose: _LatencyPurpose,
    warmup_iterations: int,
    prediction_atol: float,
    latency_sweeps: int,
    *,
    clock: _Clock = time.perf_counter_ns,
    host: HostMetadata | None = None,
) -> None:
    """Validate, resume, and complete one CPU latency campaign."""

    artifacts = _resolve_campaign(storage_root, k_study_experiment_id, held_out_experiment_id)
    protocol = _ensure_protocol(
        output,
        purpose=purpose,
        k_study_experiment_id=k_study_experiment_id,
        held_out_experiment_id=held_out_experiment_id,
        artifacts=artifacts,
        warmup_iterations=warmup_iterations,
        prediction_atol=prediction_atol,
        latency_sweeps=latency_sweeps,
        host=_host_metadata() if host is None else host,
    )
    for sweep in range(1, latency_sweeps + 1):
        for cell in _cell_order(sweep):
            _run_unit(
                storage_root,
                output,
                protocol,
                _records_for_cell(artifacts, cell),
                cell,
                sweep,
                clock=clock,
            )


StorageRoot = Annotated[Path, typer.Argument(resolve_path=True, exists=True, file_okay=False)]
Output = Annotated[Path, typer.Argument(resolve_path=True, file_okay=False)]


def cpu(
    storage_root: StorageRoot,
    k_study_experiment_id: UUID,
    held_out_experiment_id: UUID,
    output: Output,
    purpose: Annotated[_LatencyPurpose, typer.Option()],
    warmup_iterations: Annotated[int, typer.Option(min=1)],
    prediction_atol: Annotated[float, typer.Option(min=0.0)],
    latency_sweeps: Annotated[int, typer.Option(min=1)] = 10,
) -> None:
    run_cpu(
        storage_root,
        k_study_experiment_id,
        held_out_experiment_id,
        output,
        purpose,
        warmup_iterations,
        prediction_atol,
        latency_sweeps,
    )


app = typer.Typer(add_completion=False)
app.command()(cpu)


if __name__ == "__main__":
    app()
