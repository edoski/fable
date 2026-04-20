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
    EvaluateConfig,
    PresetSpec,
    TrainConfig,
    TuneConfig,
    WorkflowSelections,
    WorkflowTask,
    resolve_workflow_config,
)
from spice.config.registry import (
    dump_canonical_yaml,
    load_named_group,
    named_group_keys,
    normalize_group_name,
)

_CONF_ROOT = Path(__file__).resolve().parents[1] / "src" / "spice" / "conf"
_SELECTION_GROUP_KEYS = frozenset(WorkflowSelections.model_fields) & frozenset(named_group_keys())
_PRESET_FIELDS = frozenset(PresetSpec.model_fields)
TEST_EVALUATION_DATE = date(2025, 11, 9)


def _write_named_spec(
    conf_root: Path,
    *,
    group: str,
    name: str,
    payload: dict[str, object],
) -> None:
    path = conf_root / normalize_group_name(group) / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_canonical_yaml(payload), encoding="utf-8")


def _named_group_copy(name: str, group: str) -> dict[str, object]:
    return deepcopy(load_named_group(name, group))


@pytest.fixture
def model_workflow_override():
    def _override(
        *,
        sample_count: int = 24,
        lookback_seconds: int = 120,
        max_delay_seconds: int = 36,
        delay_seconds: int | None = None,
        compiler_id: str = "estimated_block",
    ) -> dict[str, object]:
        problem = _named_group_copy("icdcs_2026", "problem")
        problem["id"] = "test_problem"
        problem["lookback_seconds"] = lookback_seconds
        problem["sample_count"] = sample_count
        problem["max_delay_seconds"] = max_delay_seconds
        problem["compiler"] = (
            {"id": "timestamp_native"}
            if compiler_id == "timestamp_native"
            else cast(dict[str, object], problem["compiler"])
        )
        dataset = _named_group_copy("icdcs_2026", "dataset")
        dataset["evaluation_date"] = TEST_EVALUATION_DATE.isoformat()
        return {
            "chain": "ethereum",
            "model": "lstm",
            "feature_set": (
                "time_native_baseline" if compiler_id == "timestamp_native" else "icdcs_2026"
            ),
            "dataset": dataset,
            "problem": problem,
            "delay_seconds": max_delay_seconds if delay_seconds is None else delay_seconds,
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
                "device": "cpu",
                "seed": 2026,
                "deterministic": True,
                "log_every_n_steps": 1,
                "input_normalization": {"id": "row_standard"},
                "precision": "fp32",
                "compile": "off",
            },
            "evaluation": {
                "evaluator": {
                    "id": "poisson_replay",
                    "window_seconds": 600,
                    "arrival_rate_per_second": 0.02,
                    "repetitions": 3,
                    "seed": 2026,
                }
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
        problem = _named_group_copy("icdcs_2026", "problem")
        problem["id"] = "acquire_test_problem"
        problem["lookback_seconds"] = lookback_seconds
        problem["sample_count"] = sample_count
        problem["max_delay_seconds"] = max_delay_seconds
        problem["compiler"] = {"id": "timestamp_native"}
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
        preset: str = "icdcs_2026",
        override: Mapping[str, object] | None = None,
        chain: str | None = None,
        dataset: str | None = None,
        problem: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        dataset_builder: str | None = None,
        feature_set: str | None = None,
        prediction: str | None = None,
        evaluation: str | None = None,
        study: str | None = None,
        variant: str | None = None,
        delay_seconds: int | None = None,
        trial_count: int | None = None,
        dry_run: bool | None = None,
    ) -> AcquireConfig | TrainConfig | TuneConfig | EvaluateConfig:
        conf_root = isolate_conf_root()
        workspace_root = tmp_path if workspace is None else workspace
        preset_payload = _named_group_copy(preset, "preset")
        preset_name = f"test_{workflow.value}"
        override_payload = dict(override or {})
        selection_values: dict[str, object] = {"storage_root": workspace_root / "outputs"}
        selection_values.update(
            {
                key: value
                for key, value in {
                    "chain": chain,
                    "dataset": dataset,
                    "problem": problem,
                    "provider": provider,
                    "model": model,
                    "dataset_builder": dataset_builder,
                    "feature_set": feature_set,
                    "prediction": prediction,
                    "evaluation": evaluation,
                    "study": study,
                    "variant": variant,
                    "delay_seconds": delay_seconds,
                    "trial_count": trial_count,
                    "dry_run": dry_run,
                }.items()
                if value is not None
            }
        )
        for key, value in override_payload.items():
            if key in _SELECTION_GROUP_KEYS:
                if isinstance(value, Mapping):
                    spec_name = f"test_{workflow.value}_{key}"
                    _write_named_spec(
                        conf_root,
                        group=key,
                        name=spec_name,
                        payload=dict(value),
                    )
                    selection_values[key] = spec_name
                    continue
                selection_values[key] = value
                continue
            if key in WorkflowSelections.model_fields and not isinstance(value, Mapping):
                selection_values[key] = value
                continue
            if key in _PRESET_FIELDS:
                if isinstance(value, Mapping) and key in named_group_keys():
                    spec_name = f"test_{workflow.value}_{key}"
                    _write_named_spec(
                        conf_root,
                        group=key,
                        name=spec_name,
                        payload=dict(value),
                    )
                    preset_payload[key] = spec_name
                    continue
                if isinstance(value, Mapping):
                    preset_payload[key] = dict(value)
                    continue
                preset_payload[key] = value
                continue
            raise ValueError(f"Unsupported test workflow override key: {key}")
        _write_named_spec(conf_root, group="preset", name=preset_name, payload=preset_payload)
        selection_values["preset"] = preset_name
        return resolve_workflow_config(
            workflow, WorkflowSelections.model_validate(selection_values)
        )

    return _load
