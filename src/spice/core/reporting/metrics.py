"""Shared reporting metrics and rendering helpers."""

from __future__ import annotations

import re
import time
from collections.abc import Iterable

from rich.console import Console, RenderableType
from rich.padding import Padding
from rich.text import Text

from .state import StageMetricDescriptor, _StageLayout, _StageState

_FINAL_STAGE_STATUSES = frozenset(
    {"done", "failed", "reused", "extended", "rebuilt", "created"}
)
_ACTIVE_STAGE_STATUSES = frozenset({"planning", "running", "pulling", "writing"})
_STAGE_STATUS_STYLES = {
    "pending": "dim",
    "planning": "cyan",
    "running": "cyan",
    "pulling": "cyan",
    "writing": "cyan",
    "done": "green",
    "reused": "green",
    "created": "green",
    "extended": "yellow",
    "rebuilt": "yellow",
    "failed": "bold red",
}
_PROGRESS_BAR_STYLES = {
    "pending": ("grey23", "grey35", "grey50", "grey50"),
    "planning": ("grey23", "cyan", "cyan", "cyan"),
    "running": ("grey23", "cyan", "cyan", "cyan"),
    "pulling": ("grey23", "cyan", "cyan", "cyan"),
    "writing": ("grey23", "cyan", "cyan", "cyan"),
    "done": ("grey23", "green", "green", "green"),
    "reused": ("grey23", "green", "green", "green"),
    "created": ("grey23", "green", "green", "green"),
    "extended": ("grey23", "yellow", "yellow", "yellow"),
    "rebuilt": ("grey23", "yellow", "yellow", "yellow"),
    "failed": ("grey23", "red", "red", "red"),
}
_RATE_COLUMN_WIDTH = 11
_TIME_COLUMN_WIDTH = 7


def _panel_body_width(console: Console) -> int:
    configured_width = getattr(console, "_width", None)
    if isinstance(configured_width, int) and configured_width > 0:
        return max(40, configured_width - 4)
    return max(40, console.size.width - 4)


def _with_top_terminal_spacer(renderable: RenderableType) -> Padding:
    return Padding(renderable, (1, 0, 0, 0))


def _active_stage_metric_descriptors(
    stages: Iterable[_StageState],
    *,
    available_width: int,
) -> tuple[StageMetricDescriptor, ...]:
    active_metrics: list[StageMetricDescriptor] = []
    seen: set[str] = set()
    for stage in stages:
        for descriptor in stage.metric_descriptors:
            if descriptor.id in seen:
                continue
            if descriptor.id not in stage.metric_values:
                continue
            seen.add(descriptor.id)
            active_metrics.append(descriptor)
    if not active_metrics:
        return ()
    if available_width >= 150:
        return tuple(active_metrics)
    if available_width >= 138:
        return tuple(active_metrics[:2])
    if available_width >= 126:
        return tuple(active_metrics[:1])
    return ()


def _stage_layout(
    available_width: int,
    *,
    has_detail: bool,
    metric_columns: tuple[StageMetricDescriptor, ...] = (),
) -> _StageLayout:
    if has_detail:
        if available_width >= 132:
            return _StageLayout(10, 8, 18, True, True, True, metric_columns)
        if available_width >= 112:
            return _StageLayout(10, 8, 16, True, True, False, metric_columns)
        if available_width >= 92:
            return _StageLayout(9, 8, 12, False, False, True, ())
        return _StageLayout(8, 7, 12, False, False, False, ())

    if available_width >= 96:
        return _StageLayout(10, 8, 20, True, True, False, metric_columns)
    if available_width >= 78:
        return _StageLayout(9, 8, 16, True, False, False, ())
    return _StageLayout(8, 7, 12, False, False, False, ())


def _progress_bucket(stage: _StageState) -> int | None:
    if stage.total is None:
        return None
    if stage.total <= 0:
        return 10
    bucket_count = min(10, stage.total)
    return min(bucket_count, (stage.completed * bucket_count) // stage.total)


def _format_progress_count(stage: _StageState, *, include_unit: bool = True) -> str:
    suffix = ""
    if include_unit and stage.unit is not None:
        suffix = f" {stage.unit}"
    if stage.total is None:
        return f"{stage.completed:,}{suffix}" if stage.completed else "--"
    return f"{stage.completed:,}/{stage.total:,}{suffix}"


def _format_progress_percent(stage: _StageState) -> str:
    if stage.total is None or stage.total <= 0:
        return "--"
    percent = int((min(stage.completed, stage.total) * 100) / stage.total)
    return f"{percent:>3d}%"


def _unit_rate_suffix(unit: str | None) -> str:
    if unit is None:
        return "u/s"
    aliases = {
        "blocks": "blk/s",
        "block": "blk/s",
        "batches": "bat/s",
        "batch": "bat/s",
        "repetitions": "rep/s",
        "repetition": "rep/s",
        "trials": "trl/s",
        "trial": "trl/s",
    }
    return aliases.get(unit, f"{unit}/s")


def _elapsed_seconds(stage: _StageState) -> float | None:
    if stage.started_at is None:
        return None
    end = stage.finished_at if stage.finished_at is not None else time.monotonic()
    return max(0.0, end - stage.started_at)


def _remaining_seconds(stage: _StageState) -> float | None:
    if (
        stage.total is None
        or stage.total <= 0
        or stage.completed <= 0
        or stage.completed >= stage.total
    ):
        return None
    elapsed = _elapsed_seconds(stage)
    if elapsed is None or elapsed <= 0:
        return None
    rate = stage.completed / elapsed
    if rate <= 0:
        return None
    return (stage.total - stage.completed) / rate


def _format_clock(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    whole_seconds = max(0, int(seconds))
    hours, remainder = divmod(whole_seconds, 60 * 60)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _render_elapsed(stage: _StageState) -> Text:
    elapsed = _elapsed_seconds(stage)
    if elapsed is None:
        return Text("--", style="dim")
    style = "green" if stage.status in _FINAL_STAGE_STATUSES else "cyan"
    return Text(_format_clock(elapsed), style=style)


def _render_eta(stage: _StageState) -> Text:
    if stage.status in _FINAL_STAGE_STATUSES:
        return Text("done", style="green")
    remaining = _remaining_seconds(stage)
    if remaining is None:
        return Text("--", style="dim")
    return Text(_format_clock(remaining), style="magenta")


def _render_rate(stage: _StageState) -> Text:
    if stage.status not in _ACTIVE_STAGE_STATUSES:
        return Text("--", style="dim")
    if stage.smoothed_rate is None:
        return Text("--", style="dim")
    suffix = _unit_rate_suffix(stage.unit)
    value = _format_rate_value(stage.smoothed_rate)
    return Text(f"{value} {suffix}", style="bright_cyan")


def _render_stage_metric(value: str | None) -> Text:
    if not value:
        return Text("--", style="dim")
    return Text(value, style="bright_cyan")


def _detail_with_hidden_metrics(
    stage: _StageState,
    *,
    visible_metric_ids: set[str],
) -> str | None:
    hidden_metric_tokens = [
        f"{descriptor.label}={stage.metric_values[descriptor.id]}"
        for descriptor in stage.metric_descriptors
        if descriptor.id in stage.metric_values and descriptor.id not in visible_metric_ids
    ]
    pieces = [*hidden_metric_tokens]
    if stage.detail:
        pieces.append(stage.detail)
    if not pieces:
        return None
    return " ".join(pieces)


def _render_stage_detail(stage: _StageState, *, visible_metric_ids: set[str]) -> Text:
    raw_detail = _detail_with_hidden_metrics(stage, visible_metric_ids=visible_metric_ids)
    if not raw_detail:
        return Text("")

    detail = Text()
    prefix, separator, remainder = raw_detail.partition(": ")
    if separator:
        detail.append(prefix, style="white")
        detail.append(separator, style="dim")
        _append_detail_parts(detail, remainder)
        return detail

    _append_detail_parts(detail, raw_detail)
    return detail


def _append_detail_parts(detail: Text, raw_detail: str) -> None:
    for index, part in enumerate(raw_detail.split(" | ")):
        if index > 0:
            detail.append(" | ", style="dim")
        _append_detail_fragment(detail, part.strip())


def _append_detail_fragment(detail: Text, fragment: str) -> None:
    if fragment.lower().startswith("waiting"):
        detail.append(fragment, style="yellow")
        return
    if fragment.lower().startswith("resolving"):
        detail.append(fragment, style="cyan")
        return

    number_match = re.match(r"^(?P<number>\d[\d,]*) (?P<label>[A-Za-z].+)$", fragment)
    if number_match is not None:
        detail.append(number_match.group("number"), style="bright_white")
        detail.append(" ", style="dim")
        detail.append(number_match.group("label"), style="dim")
        return

    detail.append(fragment, style="dim")


def _smooth_value(previous: float | None, current: float, *, alpha: float) -> float:
    if previous is None:
        return current
    return previous + alpha * (current - previous)


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


def _format_rate_value(rate: float) -> str:
    return format_compact_number(rate)
