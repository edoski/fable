"""Study manifest creation, persistence, and validation."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from ..config.models import (
    EvaluateConfig,
    PredictionConfig,
    ProblemSpec,
    SplitConfig,
    TrainConfig,
    TrainingConfig,
    TuneConfig,
    TuningConfig,
    TuningSpaceConfig,
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
from ..modeling._training_context import compile_training_context
from ..modeling.dataset_builders import coerce_dataset_builder_config
from ..modeling.families.base import ModelConfig
from ..modeling.families.registry import coerce_model_config
from ..modeling.tuned_config import coerce_tuning_space_config
from ..modeling.result_codecs import study_semantics_from_payload, study_semantics_payload
from ..objectives import coerce_objective_config
from ..semantics import StudySemantics
from .engine import STUDY_ROOT_KIND, create_state_engine, ensure_state_db
from .identity import (
    IdentityModel,
    identity_payload,
    study_manifest_identity,
    study_request_identity_from_manifest,
    study_request_identity_from_tuned_config,
)
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
    context = compile_training_context(config)
    return StudyManifest(
        study_id=paths.study_id,
        dataset_builder=config.dataset_builder,
        prediction=config.prediction,
        objective=config.objective,
        study_name=config.study.name,
        chain_name=config.chain.name,
        dataset_id=paths.corpus_id,
        dataset_name=config.dataset.name,
        problem=config.problem,
        feature_set=config.feature_set,
        model=config.model,
        split=config.split,
        training=config.training,
        tuning=config.tuning,
        sampler_name=STUDY_SAMPLER_NAME,
        sampler_seed=config.tuning.sampler_seed,
        pruner_name=pruner_name(config.tuning.enable_pruning),
        enable_pruning=config.tuning.enable_pruning,
        tuning_space=config.tuning_space,
        semantics=StudySemantics(
            problem=context.problem_contract.semantics,
            realization_policy=context.problem_contract.realization_policy.semantics,
            objective=context.objective_contract.semantics,
            feature=context.feature_contract.semantics,
            prediction=context.prediction_contract.semantics,
            input_normalization=context.input_normalization_contract.semantics,
            representation=context.representation_contract.semantics,
            dataset_builder=context.dataset_builder_contract.semantics,
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

    if not db_path.is_file():
        raise MissingStateError(f"Missing study manifest: {db_path}")
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
    if not db_path.is_file():
        return None
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
    return _mismatched_identity_fields(
        study_manifest_identity(stored),
        study_manifest_identity(requested),
    )


def validate_tuned_train_request(
    config: TrainConfig | EvaluateConfig,
    *,
    manifest: StudyManifest,
) -> None:
    """Reject tuned-artifact requests whose identity diverges from the stored study."""

    paths = resolve_workflow_paths(config)
    if paths.study_id is None:
        raise ConfigResolutionError("study_id is required for tuned artifacts")
    mismatches = _mismatched_identity_fields(
        study_request_identity_from_manifest(manifest),
        study_request_identity_from_tuned_config(
            config,
            study_id=paths.study_id,
            dataset_id=paths.corpus_id,
        ),
    )
    if mismatches:
        raise StateConflictError(
            "Tuned artifact request does not match study definition: " + ", ".join(mismatches)
        )


def _mismatched_identity_fields(
    stored_identity: IdentityModel,
    requested_identity: IdentityModel,
) -> list[str]:
    if type(stored_identity) is not type(requested_identity):
        raise TypeError("identity comparisons require matching identity types")
    return _identity_field_mismatches(stored_identity, requested_identity)


def _identity_field_mismatches(
    stored_identity: IdentityModel,
    requested_identity: IdentityModel,
    *,
    prefix: str = "",
) -> list[str]:
    mismatches: list[str] = []
    for field_name in stored_identity.__class__.model_fields:
        stored_value = getattr(stored_identity, field_name)
        requested_value = getattr(requested_identity, field_name)
        label = field_name if not prefix else f"{prefix}.{field_name}"
        if isinstance(stored_value, IdentityModel) and isinstance(requested_value, IdentityModel):
            mismatches.extend(
                _identity_field_mismatches(
                    stored_value,
                    requested_value,
                    prefix=label,
                )
            )
            continue
        if stored_value != requested_value:
            mismatches.append(label)
    return mismatches


def manifest_payload(manifest: StudyManifest) -> dict[str, object]:
    return {
        **identity_payload(study_manifest_identity(manifest)),
        "semantics": study_semantics_payload(manifest.semantics),
    }


def manifest_from_payload(payload: dict[str, object]) -> StudyManifest:
    model = coerce_model_config(mapping_payload(payload["model"], label="study.model"))
    problem = coerce_problem_spec(mapping_payload(payload["problem"], label="study.problem"))
    prediction = coerce_prediction_config(
        mapping_payload(payload["prediction"], label="study.prediction")
    )
    semantics_payload = mapping_payload(payload["semantics"], label="study.semantics")
    return StudyManifest(
        study_id=str(payload["study_id"]),
        dataset_builder=coerce_dataset_builder_config(
            mapping_payload(payload["dataset_builder"], label="study.dataset_builder")
        ),
        prediction=prediction,
        objective=coerce_objective_config(
            mapping_payload(payload["objective"], label="study.objective")
        ),
        study_name=str(payload["study_name"]),
        chain_name=str(payload["chain_name"]),
        dataset_id=str(payload["dataset_id"]),
        dataset_name=str(payload["dataset_name"]),
        problem=problem,
        feature_set=coerce_feature_set_config(
            mapping_payload(payload["feature_set"], label="study.feature_set")
        ),
        model=model,
        split=SplitConfig.model_validate(mapping_payload(payload["split"], label="study.split")),
        training=TrainingConfig.model_validate(
            mapping_payload(payload["training"], label="study.training")
        ),
        tuning=TuningConfig.model_validate(
            mapping_payload(payload["tuning"], label="study.tuning")
        ),
        sampler_name=str(payload["sampler_name"]),
        sampler_seed=coerce_int(payload["sampler_seed"], label="sampler_seed"),
        pruner_name=str(payload["pruner_name"]),
        enable_pruning=bool(payload["enable_pruning"]),
        tuning_space=coerce_study_tuning_space(
            payload["tuning_space"],
            model=model,
            problem=problem,
        ),
        semantics=study_semantics_from_payload(semantics_payload),
    )


def coerce_study_tuning_space(
    payload: object,
    *,
    model: ModelConfig,
    problem: ProblemSpec,
) -> TuningSpaceConfig:
    tuning_space = coerce_tuning_space_config(
        mapping_payload(payload, label="study.tuning_space"),
        model_config=model,
        problem_config=problem,
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
