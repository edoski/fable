from __future__ import annotations

import numpy as np
import pytest
import torch

from spice.modeling.batch_sources import (
    _build_batch_source,
    _resolve_host_loader_worker_settings,
)
from spice.modeling.representations import RepresentationRuntimeContext


def test_host_loader_workers_default_to_zero_without_slurm_budget(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("SPICE_DATALOADER_WORKERS", raising=False)

    settings = _resolve_host_loader_worker_settings()

    assert settings.num_workers == 0
    assert settings.persistent_workers is False
    assert settings.prefetch_factor is None


def test_host_loader_workers_scale_from_slurm_cpu_budget(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.delenv("SPICE_DATALOADER_WORKERS", raising=False)

    settings = _resolve_host_loader_worker_settings()

    assert settings.num_workers == 6
    assert settings.persistent_workers is True
    assert settings.prefetch_factor == 4


def test_host_loader_worker_override_takes_precedence(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.setenv("SPICE_DATALOADER_WORKERS", "3")

    settings = _resolve_host_loader_worker_settings()

    assert settings.num_workers == 3
    assert settings.persistent_workers is True
    assert settings.prefetch_factor == 2


def test_host_loader_worker_override_rejects_invalid_values(monkeypatch) -> None:
    monkeypatch.setenv("SPICE_DATALOADER_WORKERS", "-1")

    with pytest.raises(ValueError, match="must be non-negative"):
        _resolve_host_loader_worker_settings()


def test_device_resident_oom_falls_back_to_host_loader(monkeypatch) -> None:
    class _Prepared:
        sample_count = 4
        batch_signatures = np.array([1, 1, 1, 1], dtype=np.int64)
        estimated_storage_bytes = 1024

        def build_batch(self, sample_positions: torch.Tensor) -> torch.Tensor:
            return sample_positions

        def to_device_storage(self, device: torch.device):
            raise torch.cuda.OutOfMemoryError("oom")

    empty_cache_calls: list[bool] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append(True))

    source = _build_batch_source(
        _Prepared(),
        required_bytes=1024,
        runtime_context=RepresentationRuntimeContext(
            batch_size=2,
            available_host_memory_bytes=1024,
            available_device_memory_bytes=2048,
        ),
        resolved_device=torch.device("cuda"),
        seed=2026,
        shuffle=False,
    )

    assert source.__class__.__name__ == "_HostDataLoaderBatchSource"
    assert empty_cache_calls == [True]
