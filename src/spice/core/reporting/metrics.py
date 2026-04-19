"""Plain reporting helpers."""

from __future__ import annotations

from .state import _StageState

_FINAL_STAGE_STATUSES = frozenset(
    {"done", "failed", "reused", "extended", "rebuilt", "created"}
)
_ACTIVE_STAGE_STATUSES = frozenset({"planning", "running", "pulling", "writing"})


def _progress_bucket(stage: _StageState) -> int | None:
    if stage.total is None:
        return None
    if stage.total <= 0:
        return 10
    bucket_count = min(10, stage.total)
    return min(bucket_count, (stage.completed * bucket_count) // stage.total)


def format_compact_number(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 100_000:
        return f"{value / 1_000:.0f}k"
    if value >= 10_000:
        return f"{value / 1_000:.1f}k"
    if value >= 1_000:
        return f"{value / 1_000:.2f}k"
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def format_compact_count(value: int) -> str:
    absolute = abs(value)
    sign = "-" if value < 0 else ""
    if absolute >= 1_000_000:
        return f"{sign}{absolute / 1_000_000:.2f}M"
    if absolute >= 100_000:
        return f"{sign}{absolute / 1_000:.0f}k"
    if absolute >= 10_000:
        return f"{sign}{absolute / 1_000:.1f}k"
    if absolute >= 1_000:
        return f"{sign}{absolute / 1_000:.2f}k"
    return f"{value:d}"
