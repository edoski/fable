from __future__ import annotations

from pathlib import Path

import pytest

from spice.config import TrainConfig, WorkflowTask
from spice.core.errors import SpiceOperatorError
from spice.execution.session import ExecutionJobSubmission
from spice.execution.submission import (
    WorkflowSubmissionEvent,
    submit_resolved_workflow,
)


class _FakeSession:
    def __init__(
        self,
        *,
        follow_by_default: bool = True,
        state: str | None = "COMPLETED",
        interrupt: bool = False,
    ) -> None:
        self.follow_by_default = follow_by_default
        self.state = state
        self.interrupt = interrupt
        self.followed = False
        self.submitted_dependency: str | None = None

    def submit_workflow(
        self,
        task: WorkflowTask,
        *,
        config,
        dependency: str | None = None,
    ) -> ExecutionJobSubmission:
        del config
        self.submitted_dependency = dependency
        return ExecutionJobSubmission(
            task=task,
            job_id="12345",
            log_path=Path("/logs/spice-train-12345.out"),
        )

    def follow_job(self, _submission: ExecutionJobSubmission) -> str | None:
        self.followed = True
        if self.interrupt:
            raise KeyboardInterrupt
        return self.state


def _train_config() -> TrainConfig:
    return TrainConfig.model_construct(workflow=WorkflowTask.TRAIN)


def test_submit_resolved_workflow_detach_skips_follow() -> None:
    session = _FakeSession()
    events: list[WorkflowSubmissionEvent] = []

    result = submit_resolved_workflow(
        _train_config(),
        target="target",
        dependency="afterok:999",
        detach=True,
        on_event=events.append,
        session_factory=lambda _target: session,
    )

    assert result.detached is True
    assert result.state is None
    assert session.followed is False
    assert session.submitted_dependency == "afterok:999"
    assert [event.kind for event in events] == ["submitted"]


def test_submit_resolved_workflow_follows_and_reports_success() -> None:
    session = _FakeSession(state="COMPLETED")
    events: list[WorkflowSubmissionEvent] = []

    result = submit_resolved_workflow(
        _train_config(),
        target="target",
        on_event=events.append,
        session_factory=lambda _target: session,
    )

    assert result.detached is False
    assert result.state == "COMPLETED"
    assert session.followed is True
    assert [(event.kind, event.state) for event in events] == [
        ("submitted", None),
        ("finished", "COMPLETED"),
    ]


def test_submit_resolved_workflow_failed_final_state_raises_after_reporting() -> None:
    session = _FakeSession(state="FAILED")
    events: list[WorkflowSubmissionEvent] = []

    with pytest.raises(SpiceOperatorError, match="Job 12345 ended with state FAILED"):
        submit_resolved_workflow(
            _train_config(),
            target="target",
            on_event=events.append,
            session_factory=lambda _target: session,
        )

    assert [(event.kind, event.state) for event in events] == [
        ("submitted", None),
        ("finished", "FAILED"),
    ]


def test_submit_resolved_workflow_keyboard_interrupt_detaches() -> None:
    session = _FakeSession(interrupt=True)
    events: list[WorkflowSubmissionEvent] = []

    result = submit_resolved_workflow(
        _train_config(),
        target="target",
        on_event=events.append,
        session_factory=lambda _target: session,
    )

    assert result.detached is True
    assert result.state == "running"
    assert [(event.kind, event.state) for event in events] == [
        ("submitted", None),
        ("detached", "running"),
    ]
