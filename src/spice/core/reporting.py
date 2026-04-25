"""Human-readable CLI reporting."""

from __future__ import annotations

import shlex
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO

RenderableValue = str | int | float | Path


class Reporter:
    """Human-facing CLI reporter with separate result and diagnostic streams."""

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        error_stream: TextIO | None = None,
    ) -> None:
        self._stream = stream or sys.stdout
        self._error_stream = error_stream or sys.stderr

    def header(
        self,
        name: str,
        facts: Iterable[tuple[str, RenderableValue]] = (),
    ) -> None:
        self._emit(" ".join([name, *(_field(key, value) for key, value in facts)]).strip())

    def milestone(self, message: str, *, level: str = "info") -> None:
        prefix = ""
        stream = self._stream
        if level == "warning":
            prefix = "warning: "
            stream = self._error_stream
        elif level == "error":
            prefix = "error: "
            stream = self._error_stream
        self._emit(f"{prefix}{message}", stream=stream)

    def result(
        self,
        name: str,
        fields: Iterable[tuple[str, RenderableValue]] = (),
        *,
        status: str = "complete",
    ) -> None:
        parts = [name, status]
        parts.extend(_field(key, value) for key, value in fields)
        self._emit(" ".join(parts))

    def sections(
        self,
        title: str,
        sections: Iterable[tuple[str, Iterable[tuple[str, RenderableValue]]]],
    ) -> None:
        self._emit_sections(title, sections, stream=self._stream)

    def diagnostic_sections(
        self,
        title: str,
        sections: Iterable[tuple[str, Iterable[tuple[str, RenderableValue]]]],
    ) -> None:
        self._emit_sections(title, sections, stream=self._error_stream)

    def _emit_sections(
        self,
        title: str,
        sections: Iterable[tuple[str, Iterable[tuple[str, RenderableValue]]]],
        *,
        stream: TextIO,
    ) -> None:
        self._emit(title, stream=stream)
        for section_title, rows in sections:
            self._emit(f"{section_title}:", stream=stream)
            for label, value in rows:
                self._emit(f"  {label}: {_stringify(value)}", stream=stream)

    def _emit(self, line: str, *, stream: TextIO | None = None) -> None:
        target = stream or self._stream
        print(line, file=target)
        target.flush()


def _stringify(value: RenderableValue) -> str:
    text = str(value).replace("\n", " ").strip()
    return text


def _field(key: str, value: RenderableValue) -> str:
    rendered = _stringify(value)
    if not rendered:
        return f"{key}=''"
    if any(char.isspace() for char in rendered):
        rendered = shlex.quote(rendered)
    return f"{key}={rendered}"
