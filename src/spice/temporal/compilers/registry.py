"""Direct problem-compiler dispatch for the fixed in-repo compilers."""

from __future__ import annotations

from collections.abc import Mapping

from ...core.errors import ConfigResolutionError
from .base import ProblemCompilerConfig


def coerce_problem_compiler_config(
    payload: Mapping[str, object] | ProblemCompilerConfig,
) -> ProblemCompilerConfig:
    if isinstance(payload, ProblemCompilerConfig):
        raw_payload = payload.model_dump(mode="json")
        compiler_id = payload.id
    else:
        raw_payload = dict(payload)
        compiler_id = _mapping_problem_compiler_id(raw_payload)
    if compiler_id == "timestamp_future_window":
        from .timestamp_future_window import TimestampFutureWindowCompilerConfig

        return TimestampFutureWindowCompilerConfig.model_validate(raw_payload)
    if compiler_id == "estimated_block":
        from .estimated_block import EstimatedBlockCompilerConfig

        return EstimatedBlockCompilerConfig.model_validate(raw_payload)
    known = ", ".join(sorted(("estimated_block", "timestamp_future_window")))
    raise ConfigResolutionError(
        f"Unknown problem.compiler.id: {compiler_id}. Known problem.compiler.id values: {known}"
    )


def compile_problem(
    problem,
    feature_contract,
    realization_policy,
    chain_runtime,
):
    compiler_id = problem.compiler.id
    if compiler_id == "timestamp_future_window":
        from .timestamp_future_window import compile_problem as compile_timestamp_future_window

        return compile_timestamp_future_window(
            problem,
            feature_contract,
            realization_policy,
            chain_runtime,
        )
    if compiler_id == "estimated_block":
        from .estimated_block import compile_problem as compile_estimated_block

        return compile_estimated_block(
            problem,
            feature_contract,
            realization_policy,
            chain_runtime,
        )
    raise ConfigResolutionError(f"Unsupported problem.compiler.id: {compiler_id}")


def problem_runtime_metadata_payload(
    compiler_id: str,
    metadata: object,
) -> dict[str, object]:
    if compiler_id == "timestamp_future_window":
        from .timestamp_future_window import runtime_metadata_payload

        return runtime_metadata_payload(metadata)
    if compiler_id == "estimated_block":
        from .estimated_block import runtime_metadata_payload

        return runtime_metadata_payload(metadata)
    raise ConfigResolutionError(f"Unsupported problem.compiler.id: {compiler_id}")


def problem_runtime_metadata_from_compiler_payload(
    compiler_id: str,
    payload: Mapping[str, object],
) -> object:
    if compiler_id == "timestamp_future_window":
        from .timestamp_future_window import runtime_metadata_from_payload

        return runtime_metadata_from_payload(payload)
    if compiler_id == "estimated_block":
        from .estimated_block import runtime_metadata_from_payload

        return runtime_metadata_from_payload(payload)
    raise ConfigResolutionError(f"Unsupported problem.compiler.id: {compiler_id}")


def _mapping_problem_compiler_id(payload: Mapping[str, object]) -> str:
    value = payload.get("id")
    if not isinstance(value, str):
        raise ConfigResolutionError("problem.compiler.id is required")
    return value
