from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from spice.workflows import _shared


class _Reporter:
    def __init__(self) -> None:
        self.closed = False
        self.messages: list[tuple[str, str]] = []

    def close(self) -> None:
        self.closed = True

    def log(self, message: str, *, level: str = "info") -> None:
        self.messages.append((message, level))


def test_abort_cleanup_runs_for_sigterm_like_interrupt(monkeypatch) -> None:
    @contextmanager
    def fake_capture_interrupts() -> Iterator[_shared._InterruptState]:
        state = _shared._InterruptState(interrupted=True)
        yield state

    monkeypatch.setattr(_shared, "_capture_interrupts", fake_capture_interrupts)
    reporter = _Reporter()
    cleaned: list[str] = []

    with pytest.raises(KeyboardInterrupt):
        with _shared.abort_cleanup(
            reporter,
            label="train",
            cleanup=lambda: cleaned.append("cleaned"),
        ):
            pass

    assert cleaned == ["cleaned"]
    assert reporter.closed is True
    assert reporter.messages == [
        ("train cancelled; partial outputs removed", "warning"),
    ]
