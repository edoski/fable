from __future__ import annotations

import shutil
from collections.abc import Mapping
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import cast

import pytest

from spice.config import (
    AcquireConfig,
    AcquireWorkflowRequest,
    EvaluateConfig,
    EvaluateWorkflowRequest,
    TrainConfig,
    TrainWorkflowRequest,
    TuneConfig,
    TuneWorkflowRequest,
    WorkflowTask,
    resolve_workflow_config,
)
from spice.config.registry import (
    dump_canonical_yaml,
    load_named_group,
    named_group_keys,
    normalize_group_name,
)
from spice.config.resolution import WorkflowConfigRequest, workflow_request_type
from spice.config.surfaces import SurfaceFrame

_CONF_ROOT = Path(__file__).resolve().parents[1] / "src" / "spice" / "conf"
_REQUEST_FIELD_NAMES = frozenset().union(
    AcquireWorkflowRequest.model_fields,
    TrainWorkflowRequest.model_fields,
    TuneWorkflowRequest.model_fields,
    EvaluateWorkflowRequest.model_fields,
)
_SELECTION_GROUP_KEYS = _REQUEST_FIELD_NAMES & frozenset(named_group_keys())
_SURFACE_FIELDS = frozenset(SurfaceFrame.model_fields)
TEST_EVALUATION_DATE = date(2025, 11, 9)
_IDENTITY_FIELDS = {
    "chain": "name",
    "dataset": "name",
    "dataset_builder": "id",
    "execution": "id",
    "evaluation": "id",
    "features": "id",
    "model": "id",
    "prediction": "id",
    "problem": "id",
    "provider": "name",
}


def _write_named_spec(
    conf_root: Path,
    *,
    group: str,
    name: str,
    payload: dict[str, object],
) -> None:
    normalized_group = normalize_group_name(group)
    identity_field = _IDENTITY_FIELDS.get(normalized_group)
    if identity_field is not None:
        payload = {**payload, identity_field: name}
    path = conf_root / normalized_group / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_canonical_yaml(payload), encoding="utf-8")


def _named_group_copy(name: str, group: str) -> dict[str, object]:
    return deepcopy(load_named_group(name, group))


def _spec_name_for_payload(group: str, default_name: str, payload: Mapping[str, object]) -> str:
    identity_field = _IDENTITY_FIELDS.get(normalize_group_name(group))
    if identity_field is None:
        return default_name
    identity_value = payload.get(identity_field)
    return identity_value if isinstance(identity_value, str) else default_name


def _set_surface_evaluation_delay(
    surface_payload: dict[str, object],
    delay_seconds: object,
) -> None:
    if delay_seconds is None:
        return
    evaluation = surface_payload.get("evaluation")
    if isinstance(evaluation, Mapping):
        surface_payload["evaluation"] = {**evaluation, "delay_seconds": delay_seconds}
    else:
        surface_payload["evaluation"] = {"id": evaluation, "delay_seconds": delay_seconds}


@pytest.fixture
def model_workflow_override():
    def _override(
        *,
        sample_count: int = 24,
        lookback_seconds: int = 120,
        max_delay_seconds: int = 36,
        delay_seconds: int | None = None,
    ) -> dict[str, object]:
        problem = _named_group_copy("current_row_nominal", "problem")
        problem["id"] = "test_problem"
        problem["lookback_seconds"] = lookback_seconds
        problem["sample_count"] = sample_count
        problem["max_delay_seconds"] = max_delay_seconds
        problem["compiler"] = cast(dict[str, object], problem["compiler"])
        dataset = _named_group_copy("icdcs_2026", "dataset")
        dataset["evaluation_date"] = TEST_EVALUATION_DATE.isoformat()
        return {
            "chain": "ethereum",
            "model": "lstm",
            "features": "core_fee_dynamics",
            "dataset": dataset,
            "problem": problem,
            "training": {
                "learning_rate": 0.0003,
                "weight_decay": 0.01,
                "batch_size": 8,
                "max_epochs": 1,
                "early_stopping": {
                    "patience": 1,
                    "min_delta": 0.0,
                },
                "gradient_clip_norm": 1.0,
                "seed": 2026,
                "deterministic": True,
                "log_every_n_steps": 1,
                "input_normalization": {"id": "row_standard"},
            },
            "evaluation": {
                "id": "poisson_replay_2h_mean",
                "engine": "replay",
                "sampler": "poisson_arrivals",
                "aggregation": {"id": "event_mean"},
                "window_seconds": 600,
                "arrival_rate_per_second": 0.02,
                "repetitions": 3,
                "seed": 2026,
                "delay_seconds": max_delay_seconds if delay_seconds is None else delay_seconds,
            },
            "tuning": {
                "trial_count": 2,
                "timeout_seconds": None,
                "sampler_seed": 2026,
                "enable_pruning": False,
            },
        }

    return _override


@pytest.fixture
def tune_override():
    def _override() -> dict[str, object]:
        tuning_space = _named_group_copy("lstm_default", "tuning_space")
        tuning_space.pop("problem", None)
        tuning_space.pop("prediction", None)
        tuning_space["training"] = {
            "learning_rate": [0.0001, 0.0003],
            "weight_decay": [0.0, 0.01],
        }
        tuning_space["model"] = {
            "id": "lstm",
            "hidden_size": [64, 128],
            "dropout": [0.0, 0.1],
        }
        return {"tuning_space": tuning_space}

    return _override


@pytest.fixture
def acquire_override():
    def _override(
        *,
        sample_count: int = 4,
        lookback_seconds: int = 24,
        max_delay_seconds: int = 12,
    ) -> dict[str, object]:
        problem = _named_group_copy("current_row_nominal", "problem")
        problem["id"] = "acquire_test_problem"
        problem["lookback_seconds"] = lookback_seconds
        problem["sample_count"] = sample_count
        problem["max_delay_seconds"] = max_delay_seconds
        dataset = _named_group_copy("icdcs_2026", "dataset")
        dataset["evaluation_date"] = TEST_EVALUATION_DATE.isoformat()
        return {
            "chain": "ethereum",
            "dataset": dataset,
            "problem": problem,
            "acquisition": {
                "dry_run": False,
                "chunk_size": 64,
                "rpc": {
                    "batch_size": 16,
                    "concurrency": 8,
                    "min_batch_size": 8,
                    "concurrency_rungs": [8],
                },
            },
        }

    return _override


@pytest.fixture
def isolate_conf_root(tmp_path: Path, monkeypatch):
    def _isolate() -> Path:
        from spice.config import registry

        target = tmp_path / "conf"
        if not target.exists():
            shutil.copytree(_CONF_ROOT, target)
        monkeypatch.setattr(registry, "_CONF_ROOT", target)
        return target

    return _isolate


@pytest.fixture
def load_workflow_config(tmp_path: Path, isolate_conf_root):
    def _load(
        workflow: WorkflowTask,
        *,
        workspace: Path | None = None,
        surface: str = "current_row_fee_dynamics",
        override: Mapping[str, object] | None = None,
        chain: str | None = None,
        study: str | None = None,
        variant: str | None = None,
        delay_seconds: int | None = None,
        trial_count: int | None = None,
        dry_run: bool | None = None,
    ) -> AcquireConfig | TrainConfig | TuneConfig | EvaluateConfig:
        conf_root = isolate_conf_root()
        workspace_root = tmp_path if workspace is None else workspace
        surface_payload = _named_group_copy(surface, "surface")
        surface_name = f"test_{workflow.value}"
        override_payload = dict(override or {})
        selection_values: dict[str, object] = {"storage_root": workspace_root / "outputs"}
        selection_values.update(
            {
                key: value
                for key, value in {
                    "chain": chain,
                    "study": study,
                    "variant": variant,
                    "delay_seconds": delay_seconds,
                    "trial_count": trial_count,
                    "dry_run": dry_run,
                }.items()
                if value is not None
            }
        )
        workflow_fields = frozenset(workflow_request_type(workflow).model_fields)
        for key, value in override_payload.items():
            if key in _SELECTION_GROUP_KEYS and key in workflow_fields:
                if isinstance(value, Mapping):
                    payload = dict(value)
                    if key == "evaluation":
                        _set_surface_evaluation_delay(
                            surface_payload,
                            payload.pop("delay_seconds", None),
                        )
                    spec_name = _spec_name_for_payload(
                        key,
                        f"test_{workflow.value}_{key}",
                        payload,
                    )
                    _write_named_spec(
                        conf_root,
                        group=key,
                        name=spec_name,
                        payload=payload,
                    )
                    selection_values[key] = spec_name
                    continue
                selection_values[key] = value
                continue
            if key in _SURFACE_FIELDS:
                if key == "acquisition" and isinstance(value, Mapping):
                    spec_name = f"test_{workflow.value}_provider"
                    provider = _named_group_copy("publicnode", "provider")
                    provider["acquisition"] = dict(value)
                    _write_named_spec(
                        conf_root,
                        group="provider",
                        name=spec_name,
                        payload=provider,
                    )
                    surface_payload[key] = {"provider": spec_name}
                    continue
                if isinstance(value, Mapping) and key in named_group_keys():
                    spec_name = _spec_name_for_payload(
                        key,
                        f"test_{workflow.value}_{key}",
                        value,
                    )
                    _write_named_spec(
                        conf_root,
                        group=key,
                        name=spec_name,
                        payload=dict(value),
                    )
                    surface_payload[key] = spec_name
                    continue
                if isinstance(value, Mapping):
                    surface_payload[key] = dict(value)
                    continue
                surface_payload[key] = value
                continue
            if key in workflow_fields and not isinstance(value, Mapping):
                selection_values[key] = value
                continue
            if key in _REQUEST_FIELD_NAMES:
                raise ValueError(
                    f"Override key {key!r} is not valid for {workflow.value} workflow"
                )
            raise ValueError(f"Unsupported test workflow override key: {key}")
        _write_named_spec(conf_root, group="surface", name=surface_name, payload=surface_payload)
        selection_values["surface"] = surface_name
        request_type = workflow_request_type(workflow)
        request = cast(WorkflowConfigRequest, request_type.model_validate(selection_values))
        return resolve_workflow_config(
            workflow,
            request,
        )

    return _load
