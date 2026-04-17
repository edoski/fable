from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path

import pytest

from spice.config import (
    AcquireConfig,
    PresetSpec,
    EvaluateConfig,
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


def _deep_merge(base: dict[str, object], override: Mapping[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            existing_id = existing.get("id")
            override_id = value.get("id")
            if (
                isinstance(existing_id, str)
                and isinstance(override_id, str)
                and existing_id != override_id
            ):
                merged[key] = dict(value)
                continue
            merged[key] = _deep_merge(existing, dict(value))
            continue
        merged[key] = dict(value) if isinstance(value, Mapping) else value
    return merged


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
        feature_set: str | None = None,
        prediction: str | None = None,
        study: str | None = None,
        variant: str | None = None,
        delay_seconds: int | None = None,
        trial_count: int | None = None,
        dry_run: bool | None = None,
    ) -> AcquireConfig | TrainConfig | TuneConfig | EvaluateConfig:
        conf_root = isolate_conf_root()
        workspace_root = tmp_path if workspace is None else workspace
        preset_payload = load_named_group(preset, "preset")
        preset_name = f"test_{workflow.value}"
        override_payload = dict(override or {})
        selection_values: dict[str, object] = {"storage_root": workspace_root / "outputs"}
        explicit_selection_values = {
            "chain": chain,
            "dataset": dataset,
            "problem": problem,
            "provider": provider,
            "model": model,
            "feature_set": feature_set,
            "prediction": prediction,
            "study": study,
            "variant": variant,
            "delay_seconds": delay_seconds,
            "trial_count": trial_count,
            "dry_run": dry_run,
        }
        for key, explicit_value in explicit_selection_values.items():
            if explicit_value is not None:
                selection_values[key] = explicit_value
        for key, value in override_payload.items():
            if key in _SELECTION_GROUP_KEYS:
                if isinstance(value, Mapping):
                    base_name = selection_values.get(key, preset_payload.get(key))
                    if not isinstance(base_name, str):
                        raise ValueError(f"Missing base spec name for {key}")
                    spec_name = f"test_{workflow.value}_{key}"
                    _write_named_spec(
                        conf_root,
                        group=key,
                        name=spec_name,
                        payload=_deep_merge(load_named_group(base_name, key), dict(value)),
                    )
                    selection_values[key] = spec_name
                    continue
                selection_values[key] = value
                continue
            if key in WorkflowSelections.model_fields and not isinstance(value, Mapping):
                selection_values[key] = value
                continue
            if key in _PRESET_FIELDS:
                if isinstance(value, Mapping):
                    base_name = preset_payload.get(key)
                    if isinstance(base_name, str) and key in named_group_keys():
                        spec_name = f"test_{workflow.value}_{key}"
                        _write_named_spec(
                            conf_root,
                            group=key,
                            name=spec_name,
                            payload=_deep_merge(load_named_group(base_name, key), dict(value)),
                        )
                        preset_payload[key] = spec_name
                    else:
                        preset_payload[key] = _deep_merge(
                            dict(base_name) if isinstance(base_name, Mapping) else {},
                            dict(value),
                        )
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
