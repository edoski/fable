"""Reporting state dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class StageMetricDescriptor:
    id: str
    label: str


@dataclass(frozen=True, slots=True)
class StageMetricValue:
    id: str
    value: str


@dataclass(slots=True)
class _TaskBinding:
    stage_key: str
    done_status: str


@dataclass(slots=True)
class _StageState:
    key: str
    label: str
    status: str = "pending"
    metric_descriptors: tuple[StageMetricDescriptor, ...] = ()
    metric_values: dict[str, str] = field(default_factory=dict)
    total: int | None = None
    unit: str | None = None
    completed: int = 0
    detail: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    last_emitted_status: str | None = None
    last_emitted_detail: str | None = None
    last_emitted_metric_values: dict[str, str] = field(default_factory=dict)
    last_emitted_completed: int | None = None
    last_emitted_bucket: int | None = None
