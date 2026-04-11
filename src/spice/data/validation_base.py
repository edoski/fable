"""Shared validation report primitives."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

ValidationStatus = Literal["clean", "error"]


class ValidationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MutableValidationReport(Protocol):
    status: ValidationStatus
    errors: list[str]


def finalize_validation_status(report: MutableValidationReport) -> None:
    report.status = "error" if report.errors else "clean"
