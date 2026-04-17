"""Shared problem-compiler types."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import field_validator

from ...modeling.families.base import ConfigModel

if TYPE_CHECKING:
    from ...config.models import ChainRuntimeSpec, ProblemSpec
    from ...features import CompiledFeatureContract
    from ..contracts import CompiledProblemContract


def _validate_path_segment(value: str, *, label: str) -> str:
    if not value or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a non-empty path segment")
    return value


class ProblemCompilerConfig(ConfigModel):
    id: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return _validate_path_segment(value, label="problem.compiler.id")


ProblemCompilerConfigT = TypeVar("ProblemCompilerConfigT", bound=ProblemCompilerConfig)


@dataclass(frozen=True, slots=True)
class ProblemCompilerSpec(Generic[ProblemCompilerConfigT]):
    id: str
    config_type: type[ProblemCompilerConfigT]
    compile_problem: Callable[
        [ProblemSpec, CompiledFeatureContract, ChainRuntimeSpec | None],
        CompiledProblemContract,
    ]
