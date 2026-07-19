"""Concrete model fitting and native Lightning artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .artifacts import ArtifactAssociation, FitDeployment, load_artifact, train

__all__ = ["ArtifactAssociation", "FitDeployment", "load_artifact", "train"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import artifacts

        return getattr(artifacts, name)
    raise AttributeError(name)
