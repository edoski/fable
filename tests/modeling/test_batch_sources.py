from __future__ import annotations

import pytest

from spice.modeling.batch_sources import _resolve_host_loader_worker_settings


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
