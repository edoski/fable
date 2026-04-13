"""Internal model-input representation registry."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from numpy.typing import NDArray

from ..data.datasets import TemporalDatasetStore

IntVector = NDArray[Any]


class RepresentationLoader(Protocol):
    def __iter__(self) -> Iterator[object]: ...

    def __len__(self) -> int: ...


@dataclass(frozen=True, slots=True)
class InputRepresentationSpec:
    id: str
    build_loader: Callable[..., RepresentationLoader]


_REPRESENTATIONS: dict[str, InputRepresentationSpec] = {}


def register_input_representation(spec: InputRepresentationSpec) -> None:
    existing = _REPRESENTATIONS.get(spec.id)
    if existing is not None:
        raise ValueError(f"Duplicate input representation id: {spec.id}")
    _REPRESENTATIONS[spec.id] = spec


def input_representation_spec(representation_id: str) -> InputRepresentationSpec:
    try:
        return _REPRESENTATIONS[representation_id]
    except KeyError as exc:
        known = ", ".join(sorted(_REPRESENTATIONS))
        raise ValueError(
            f"Unknown input representation: {representation_id}. Known representations: {known}"
        ) from exc


def build_representation_loader(
    representation_id: str,
    store: TemporalDatasetStore,
    sample_indices: IntVector,
    *,
    batch_size: int,
    shuffle: bool = False,
) -> RepresentationLoader:
    spec = input_representation_spec(representation_id)
    return spec.build_loader(
        store,
        sample_indices,
        batch_size=batch_size,
        shuffle=shuffle,
    )
