"""Study manifest creation, persistence, and validation."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from ..config import (
    ModelConfig,
    SplitConfig,
    TrainConfig,
    TrainingConfig,
    TuneConfig,
    TuningSpaceConfig,
    coerce_dataset_builder_config,
    coerce_feature_set_config,
    coerce_prediction_config,
    coerce_problem_spec,
)
from ..core.errors import (
    ConfigResolutionError,
    MissingStateError,
    StateConflictError,
    StateLayoutError,
)
from ..features import compile_feature_contract
from ..modeling.dataset_builders import compile_dataset_builder_contract
from ..modeling.families.registry import (
    coerce_model_config,
    coerce_tuning_space_config,
    resolve_model_representation_id,
)
from ..modeling.representations import compile_representation_contract
from ..modeling.result_codecs import study_semantics_from_payload, study_semantics_payload
from ..prediction import compile_prediction_contract
from ..semantics import StudySemantics
from ..temporal.contracts import compile_problem_contract
from ..temporal.input_normalization import compile_input_normalization_contract
from .engine import STUDY_ROOT_KIND, create_state_engine, ensure_state_db
from .layout import resolve_workflow_paths
from .payloads import PayloadCodec, SingletonPayloadStore, mapping_payload
from .schema import STUDY_TABLES, study_manifest
from .study_models import StudyManifest

STUDY_SAMPLER_NAME = "TPESampler"


_STUDY_MANIFEST_STORE = SingletonPayloadStore(
    table=study_manifest,
    codec=PayloadCodec(
        encode=lambda manifest: manifest_payload(manifest),
        decode=lambda payload: manifest_from_payload(payload),
    ),
)


def manifest_from_tune_config(config: TuneConfig) -> StudyManifest:
    """Build the canonical study manifest from one validated tuning request."""

    paths = resolve_workflow_paths(config)
    if paths.study_id is None:
        raise ConfigResolutionError("study_id is required for study manifests")
    feature_contract = compile_feature_contract(feature_set=config.feature_set)
    problem_contract = compile_problem_contract(
        problem=config.problem,
        feature_contract=feature_contract,
        chain_runtime=config.chain.runtime,
    )
    prediction_contract = compile_prediction_contract(
        prediction_id=config.prediction.id,
        family_config=config.prediction.family,
    )
    dataset_builder_contract = compile_dataset_builder_contract(config.dataset_builder)
    input_normalization_contract = compile_input_normalization_contract(
        config.training.input_normalization
    )
    representation_contract = compile_representation_contract(
        resolve_model_representation_id(config.model)
    )
    return StudyManifest(
        study_id=paths.study_id,
        dataset_builder=config.dataset_builder,
        prediction=config.prediction,
        study_name=config.study.name,
        chain_name=config.chain.name,
        dataset_id=paths.corpus_id,
        dataset_name=config.dataset.name,
        problem=config.problem,
        feature_set=config.feature_set,
        model=config.model,
        split=config.split,
        training=config.training,
        sampler_name=STUDY_SAMPLER_NAME,
        sampler_seed=config.tuning.sampler_seed,
        pruner_name=pruner_name(config.tuning.enable_pruning),
        enable_pruning=config.tuning.enable_pruning,
        tuning_space=config.tuning_space,
        semantics=StudySemantics(
            problem=problem_contract.semantics,
            feature=feature_contract.semantics,
            prediction=prediction_contract.semantics,
            input_normalization=input_normalization_contract.semantics,
            representation=representation_contract.semantics,
            dataset_builder=dataset_builder_contract.semantics,
        ),
    )


def insert_study_manifest(db_path: Path, *, manifest: StudyManifest) -> None:
    """Insert the study manifest exactly once for a study root."""

    ensure_state_db(db_path, root_kind=STUDY_ROOT_KIND, tables=STUDY_TABLES)
    engine = create_state_engine(db_path)
    try:
        with engine.begin() as conn:
            existing = _STUDY_MANIFEST_STORE.load(conn)
            if existing is not None:
                raise StateConflictError(f"Study manifest already exists: {db_path}")
            _STUDY_MANIFEST_STORE.upsert(conn, manifest)
    finally:
        engine.dispose()


def load_study_manifest(db_path: Path) -> StudyManifest:
    """Load the canonical study manifest that owns persisted study provenance."""

    ensure_state_db(db_path, root_kind=STUDY_ROOT_KIND, tables=STUDY_TABLES)
    engine = create_state_engine(db_path)
    try:
        with engine.connect() as conn:
            manifest = _STUDY_MANIFEST_STORE.load(conn)
        if manifest is None:
            raise MissingStateError(f"Missing study manifest: {db_path}")
        return manifest
    finally:
        engine.dispose()


def try_load_study_manifest(db_path: Path) -> StudyManifest | None:
    ensure_state_db(db_path, root_kind=STUDY_ROOT_KIND, tables=STUDY_TABLES)
    engine = create_state_engine(db_path)
    try:
        with engine.connect() as conn:
            row = conn.execute(select(study_manifest.c.singleton)).scalar_one_or_none()
        if row is None:
            return None
    finally:
        engine.dispose()
    return load_study_manifest(db_path)


def diff_study_manifests(stored: StudyManifest, requested: StudyManifest) -> list[str]:
    stored_payload = study_manifest_identity_payload(stored)
    requested_payload = study_manifest_identity_payload(requested)
    return [key for key in stored_payload if stored_payload[key] != requested_payload[key]]


def validate_tuned_train_request(config: TrainConfig, *, manifest: StudyManifest) -> None:
    """Reject tuned-artifact requests whose identity diverges from the stored study."""

    paths = resolve_workflow_paths(config)
    if paths.study_id is None:
        raise ConfigResolutionError("study_id is required for tuned artifacts")
    stored_payload = _study_manifest_request_payload(manifest)
    requested_payload = _train_request_identity_payload(
        config,
        study_id=paths.study_id,
        dataset_id=paths.corpus_id,
    )
    mismatches = [key for key in stored_payload if stored_payload[key] != requested_payload[key]]
    if mismatches:
        raise StateConflictError(
            "Tuned artifact request does not match study definition: " + ", ".join(mismatches)
        )


def study_manifest_identity_payload(manifest: StudyManifest) -> dict[str, object]:
    return {
        **_study_manifest_request_payload(manifest),
        "split": manifest.split.model_dump(mode="json"),
        "training": manifest.training.model_dump(mode="json"),
        "sampler_name": manifest.sampler_name,
        "sampler_seed": manifest.sampler_seed,
        "pruner_name": manifest.pruner_name,
        "enable_pruning": manifest.enable_pruning,
        "tuning_space": manifest.tuning_space.model_dump(mode="json", exclude_none=True),
    }


def _study_request_identity_payload(
    *,
    study_name: str,
    study_id: str | None,
    dataset_builder_payload: dict[str, object],
    prediction_payload: dict[str, object],
    chain_name: str,
    dataset_id: str,
    dataset_name: str,
    problem_payload: dict[str, object],
    feature_set_payload: dict[str, object],
    model_payload: dict[str, object],
) -> dict[str, object]:
    return {
        "study_name": study_name,
        "study_id": study_id,
        "dataset_builder": dataset_builder_payload,
        "prediction": prediction_payload,
        "chain_name": chain_name,
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "problem": problem_payload,
        "feature_set": feature_set_payload,
        "model": model_payload,
    }


def _study_manifest_request_payload(manifest: StudyManifest) -> dict[str, object]:
    return _study_request_identity_payload(
        study_name=manifest.study_name,
        study_id=manifest.study_id,
        dataset_builder_payload=manifest.dataset_builder.model_dump(mode="json", exclude_none=True),
        prediction_payload=manifest.prediction.model_dump(mode="json"),
        chain_name=manifest.chain_name,
        dataset_id=manifest.dataset_id,
        dataset_name=manifest.dataset_name,
        problem_payload=manifest.problem.model_dump(mode="json"),
        feature_set_payload=manifest.feature_set.model_dump(mode="json"),
        model_payload=manifest.model.model_dump(mode="json", exclude_none=True),
    )


def _train_request_identity_payload(
    config: TrainConfig,
    *,
    study_id: str,
    dataset_id: str,
) -> dict[str, object]:
    return _study_request_identity_payload(
        study_name=config.study.name,
        study_id=study_id,
        dataset_builder_payload=config.dataset_builder.model_dump(mode="json", exclude_none=True),
        prediction_payload=config.prediction.model_dump(mode="json"),
        chain_name=config.chain.name,
        dataset_id=dataset_id,
        dataset_name=config.dataset.name,
        problem_payload=config.problem.model_dump(mode="json"),
        feature_set_payload=config.feature_set.model_dump(mode="json"),
        model_payload=config.model.model_dump(mode="json", exclude_none=True),
    )


def manifest_payload(manifest: StudyManifest) -> dict[str, object]:
    return {
        **study_manifest_identity_payload(manifest),
        "semantics": study_semantics_payload(manifest.semantics),
    }


def manifest_from_payload(payload: dict[str, object]) -> StudyManifest:
    model = coerce_model_config(mapping_payload(payload["model"], label="study.model"))
    semantics_payload = mapping_payload(payload["semantics"], label="study.semantics")
    return StudyManifest(
        study_id=str(payload["study_id"]),
        dataset_builder=coerce_dataset_builder_config(
            mapping_payload(payload["dataset_builder"], label="study.dataset_builder")
        ),
        prediction=coerce_prediction_config(
            mapping_payload(payload["prediction"], label="study.prediction")
        ),
        study_name=str(payload["study_name"]),
        chain_name=str(payload["chain_name"]),
        dataset_id=str(payload["dataset_id"]),
        dataset_name=str(payload["dataset_name"]),
        problem=coerce_problem_spec(mapping_payload(payload["problem"], label="study.problem")),
        feature_set=coerce_feature_set_config(
            mapping_payload(payload["feature_set"], label="study.feature_set")
        ),
        model=model,
        split=SplitConfig.model_validate(mapping_payload(payload["split"], label="study.split")),
        training=TrainingConfig.model_validate(
            mapping_payload(payload["training"], label="study.training")
        ),
        sampler_name=str(payload["sampler_name"]),
        sampler_seed=coerce_int(payload["sampler_seed"], label="sampler_seed"),
        pruner_name=str(payload["pruner_name"]),
        enable_pruning=bool(payload["enable_pruning"]),
        tuning_space=coerce_study_tuning_space(payload["tuning_space"], model=model),
        semantics=study_semantics_from_payload(semantics_payload),
    )


def coerce_study_tuning_space(payload: object, *, model: ModelConfig) -> TuningSpaceConfig:
    tuning_space = coerce_tuning_space_config(
        mapping_payload(payload, label="study.tuning_space"),
        model_config=model,
    )
    if tuning_space is None:
        raise StateLayoutError("Study tuning_space payload is required")
    return tuning_space


def pruner_name(enable_pruning: bool) -> str:
    return "MedianPruner" if enable_pruning else "NopPruner"


def coerce_int(value: object, *, label: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value
