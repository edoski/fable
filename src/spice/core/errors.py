"""Operator-facing SPICE error types for CLI and automated workflow entrypoints."""

from __future__ import annotations

from collections.abc import Sequence


class SpiceOperatorError(Exception):
    """Base error for operator-facing failures that should render without a traceback."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class ConfigResolutionError(SpiceOperatorError):
    """Raised when selectors or config payloads cannot resolve into a runnable request."""


class MissingStateError(SpiceOperatorError):
    """Raised when a requested stored state root or required state row is missing."""


class StateLayoutError(SpiceOperatorError):
    """Raised when an on-disk state database does not match the current schema."""


class StateConflictError(SpiceOperatorError):
    """Raised when an existing state root conflicts with the requested operation."""


class SelectorResolutionError(SpiceOperatorError):
    """Raised when selector-driven lookup yields zero or multiple matches."""

    def __init__(self, *, kind: str, records: Sequence[object]) -> None:
        self.kind = kind
        self.records = tuple(records)
        message = f"Expected exactly one {kind} match" if records else f"No {kind} matches found"
        super().__init__(message)
