"""Closed dispatch for supported problem compilers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from ...core.errors import ConfigResolutionError
from .base import ProblemCompilerConfig, ProblemCompilerSpec

_KNOWN_PROBLEM_COMPILERS = ("estimated_block", "timestamp_future_window", "timestamp_native")


def problem_compiler_spec(compiler_id: str) -> ProblemCompilerSpec[ProblemCompilerConfig]:
    if compiler_id == "timestamp_native":
        from .timestamp_native import (
            TimestampNativeCompilerConfig,
        )
        from .timestamp_native import (
            compile_problem as compile_timestamp_native,
        )

        return ProblemCompilerSpec(
            id="timestamp_native",
            config_type=TimestampNativeCompilerConfig,
            compile_problem=compile_timestamp_native,
        )
    if compiler_id == "timestamp_future_window":
        from .timestamp_future_window import TimestampFutureWindowCompilerConfig
        from .timestamp_future_window import compile_problem as compile_timestamp_future_window

        return ProblemCompilerSpec(
            id="timestamp_future_window",
            config_type=TimestampFutureWindowCompilerConfig,
            compile_problem=compile_timestamp_future_window,
        )
    if compiler_id == "estimated_block":
        from .estimated_block import (
            EstimatedBlockCompilerConfig,
        )
        from .estimated_block import (
            compile_problem as compile_estimated_block,
        )

        return ProblemCompilerSpec(
            id="estimated_block",
            config_type=EstimatedBlockCompilerConfig,
            compile_problem=compile_estimated_block,
        )
    known = ", ".join(_KNOWN_PROBLEM_COMPILERS)
    raise ConfigResolutionError(
        f"Unknown problem.compiler.id: {compiler_id}. Known problem.compiler.id values: {known}"
    )


def coerce_problem_compiler_config(
    payload: Mapping[str, object] | ProblemCompilerConfig,
) -> ProblemCompilerConfig:
    if isinstance(payload, ProblemCompilerConfig):
        raw_payload = payload.model_dump(mode="json")
        compiler_id = payload.id
    else:
        raw_payload = dict(payload)
        compiler_id = _mapping_problem_compiler_id(raw_payload)
    spec = problem_compiler_spec(compiler_id)
    return spec.config_type.model_validate(raw_payload)


def _mapping_problem_compiler_id(payload: Mapping[str, object]) -> str:
    value = payload.get("id")
    if not isinstance(value, str):
        raise ConfigResolutionError("problem.compiler.id is required")
    return cast(str, value)
