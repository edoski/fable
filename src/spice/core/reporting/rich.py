"""Rich interactive reporter implementation."""

from __future__ import annotations

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .metrics import (
    _PROGRESS_BAR_STYLES,
    _STAGE_STATUS_STYLES,
    _active_stage_metric_descriptors,
    _panel_body_width,
    _render_elapsed,
    _render_eta,
    _render_rate,
    _render_stage_detail,
    _render_stage_metric,
    _stage_layout,
    _with_top_terminal_spacer,
)
from .plain import _BaseWorkflowReporter
from .state import _StageState


class RichReporter(_BaseWorkflowReporter):
    """Interactive reporter with shared workflow staging."""

    def __init__(self, console: Console | None = None) -> None:
        super().__init__(console=console)
        self._live: Live | None = None

    def close(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _on_workflow_configured(self) -> None:
        self._refresh_live()

    def _on_stage_change(self, stage: _StageState) -> None:
        del stage
        self._refresh_live()

    def _refresh_live(self) -> None:
        if self._live is None:
            self._live = Live(
                self._render_workflow(),
                console=self.console,
                auto_refresh=False,
                transient=False,
            )
            self._live.start()
            return
        self._live.update(self._render_workflow(), refresh=True)

    def _render_workflow(self) -> RenderableType:
        elements: list[RenderableType] = []
        if self._workflow_facts:
            elements.append(self._render_fact_grid())
        if self._stages:
            if elements:
                elements.append(Rule(style="grey35"))
            elements.append(self._render_stage_table())
        body = Group(*elements) if elements else Text("")
        return _with_top_terminal_spacer(
            Panel(
                body,
                title=Text(self._workflow_title or "", style="bold cyan"),
                border_style="cyan",
                padding=(0, 1),
                expand=True,
            )
        )

    def _render_fact_grid(self) -> Table:
        facts = Table.grid(padding=(0, 1), expand=False)
        facts.add_column(style="bold cyan", no_wrap=True)
        facts.add_column()
        for label, value in self._workflow_facts:
            facts.add_row(label, Text(value))
        return facts

    def _render_stage_table(self) -> Table:
        available_width = _panel_body_width(self.console)
        metric_columns = _active_stage_metric_descriptors(
            self._stages.values(),
            available_width=available_width,
        )
        visible_metric_ids = {descriptor.id for descriptor in metric_columns}
        has_detail = any(
            _render_stage_detail(stage, visible_metric_ids=visible_metric_ids).plain
            for stage in self._stages.values()
        )
        layout = _stage_layout(
            available_width,
            has_detail=has_detail,
            metric_columns=metric_columns,
        )
        table = Table(
            show_header=True,
            header_style="bold dim",
            expand=False,
            box=None,
            pad_edge=False,
            padding=(0, 1),
            collapse_padding=False,
        )
        table.add_column(
            "stage",
            width=layout.stage_width,
            no_wrap=True,
            overflow="ellipsis",
            style="bold",
        )
        table.add_column(
            "status",
            width=layout.status_width,
            no_wrap=True,
            overflow="ellipsis",
        )
        table.add_column("progress", width=layout.progress_width, no_wrap=True)
        for descriptor in layout.metric_columns:
            table.add_column(
                descriptor.label,
                width=descriptor.width,
                no_wrap=True,
                justify="right",
            )
        if layout.show_rate:
            table.add_column("rate", width=11, no_wrap=True, justify="right")
        table.add_column("elapsed", width=7, no_wrap=True, justify="right")
        if layout.show_eta:
            table.add_column("eta", width=7, no_wrap=True, justify="right")
        if layout.show_detail:
            table.add_column("detail", ratio=1, no_wrap=True, overflow="ellipsis")
        for stage in self._stages.values():
            row = [
                Text(stage.label, style="bold"),
                Text(stage.status, style=_STAGE_STATUS_STYLES.get(stage.status, "")),
                self._render_progress(stage, bar_width=layout.progress_bar_width),
            ]
            for descriptor in layout.metric_columns:
                row.append(_render_stage_metric(stage.metric_values.get(descriptor.id)))
            if layout.show_rate:
                row.append(_render_rate(stage))
            row.append(_render_elapsed(stage))
            if layout.show_eta:
                row.append(_render_eta(stage))
            if layout.show_detail:
                row.append(_render_stage_detail(stage, visible_metric_ids=visible_metric_ids))
            table.add_row(*row)
        return table

    def _render_progress(self, stage: _StageState, *, bar_width: int):
        if stage.total is None:
            return Text("--", style="dim")
        progress = Table.grid(padding=(0, 1))
        progress.add_column(width=bar_width)
        progress.add_column(width=4, no_wrap=True, justify="right")
        style, complete_style, finished_style, pulse_style = _PROGRESS_BAR_STYLES.get(
            stage.status,
            ("grey23", "cyan", "cyan", "cyan"),
        )
        progress.add_row(
            ProgressBar(
                total=max(float(stage.total), 1.0),
                completed=float(min(stage.completed, stage.total)),
                width=bar_width,
                style=style,
                complete_style=complete_style,
                finished_style=finished_style,
                pulse_style=pulse_style,
            ),
            Text(_format_progress_percent(stage), style="dim"),
        )
        return progress


def _format_progress_percent(stage: _StageState) -> str:
    if stage.total is None or stage.total <= 0:
        return "--"
    percent = int((min(stage.completed, stage.total) * 100) / stage.total)
    return f"{percent:>3d}%"
