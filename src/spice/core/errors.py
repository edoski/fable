"""Operator-facing SPICE error base."""


class SpiceOperatorError(Exception):
    """Base error for operator-facing failures that should render without a traceback."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)
