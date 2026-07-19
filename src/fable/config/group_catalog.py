"""Shared catalog for named config groups."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar

ConfigT = TypeVar("ConfigT")
_ValidateGroupPayload = Callable[[dict[str, object]], ConfigT]


class ConfigGroup(StrEnum):
    pass


@dataclass(frozen=True, slots=True)
class GroupSpec(Generic[ConfigT]):
    group: ConfigGroup
    seed_name: str | None
    validate: _ValidateGroupPayload[ConfigT]
    identity_field: str | None = None
    seed_from_requested_name: bool = False
    public: bool = False

    @property
    def token(self) -> str:
        return self.group.value

    @property
    def directory(self) -> str:
        return self.group.value


GROUP_SPECS: tuple[GroupSpec[object], ...] = ()
