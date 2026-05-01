"""Resolved Workflow Snapshot codec."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from ..core.errors import ConfigResolutionError
from .models import (
    StorageSpec,
    WorkflowTask,
)
from .workflow_configs import (
    ResolvedWorkflowConfig,
    coerce_resolved_snapshot_workflow_config,
)

_SNAPSHOT_WORKFLOWS = frozenset(
    {WorkflowTask.TRAIN, WorkflowTask.TUNE, WorkflowTask.EVALUATE}
)


def workflow_config_snapshot_payload(
    config: ResolvedWorkflowConfig,
    *,
    storage_root_override: Path | None = None,
) -> dict[str, object]:
    snapshot_config = config
    if storage_root_override is not None:
        snapshot_config = config.model_copy(
            update={"storage": StorageSpec(root=storage_root_override)}
        )
    return cast(
        dict[str, object],
        snapshot_config.model_dump(mode="json", exclude_none=True),
    )


def workflow_config_snapshot_json(
    config: ResolvedWorkflowConfig,
    *,
    storage_root_override: Path | None = None,
) -> str:
    return json.dumps(
        workflow_config_snapshot_payload(
            config,
            storage_root_override=storage_root_override,
        ),
        sort_keys=True,
    )


def hydrate_workflow_config_snapshot_json(
    workflow: WorkflowTask,
    payload: str,
) -> ResolvedWorkflowConfig:
    try:
        raw_payload = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ConfigResolutionError(str(exc)) from exc
    if not isinstance(raw_payload, Mapping):
        raise ConfigResolutionError("resolved workflow snapshot must be a mapping")
    return hydrate_workflow_config_snapshot(workflow, raw_payload)


def hydrate_workflow_config_snapshot(
    workflow: WorkflowTask,
    payload: Mapping[str, object],
) -> ResolvedWorkflowConfig:
    _validate_snapshot_workflow(workflow, payload)
    return coerce_resolved_snapshot_workflow_config(workflow, payload)


def _validate_snapshot_workflow(
    workflow: WorkflowTask,
    payload: Mapping[str, object],
) -> None:
    if workflow not in _SNAPSHOT_WORKFLOWS:
        raise ConfigResolutionError(f"Unsupported resolved workflow: {workflow.value}")
    raw_workflow = payload.get("workflow")
    if raw_workflow is None:
        return
    try:
        snapshot_workflow = WorkflowTask(str(raw_workflow))
    except ValueError as exc:
        raise ConfigResolutionError(
            f"resolved workflow snapshot has invalid workflow: {raw_workflow}"
        ) from exc
    if snapshot_workflow is not workflow:
        raise ConfigResolutionError(
            "resolved workflow snapshot workflow mismatch: "
            f"expected {workflow.value}, got {snapshot_workflow.value}"
        )
