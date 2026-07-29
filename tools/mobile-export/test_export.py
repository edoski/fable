from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from uuid import UUID

import export as mobile_export
import pytest
import torch
import yaml
from torch import nn

torch.manual_seed(2026)

_CORPUS_IDS = {
    "ethereum": UUID("10000000-0000-4000-8000-000000000001"),
    "polygon": UUID("10000000-0000-4000-8000-000000000002"),
    "avalanche": UUID("10000000-0000-4000-8000-000000000003"),
}


def _artifact_id(index: int) -> UUID:
    return UUID(f"20000000-0000-4000-8000-{index:012d}")


def _write_roster(path: Path) -> dict[tuple[str, int], UUID]:
    artifact_ids: dict[tuple[str, int], UUID] = {}
    raw: dict[str, dict[int, str]] = {}
    index = 1
    for chain in mobile_export._CHAINS:
        raw[chain] = {}
        for horizon in mobile_export._HORIZONS:
            artifact_id = _artifact_id(index)
            artifact_ids[(chain, horizon)] = artifact_id
            raw[chain][horizon] = str(artifact_id)
            index += 1
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return artifact_ids


def _association(
    chain: str,
    horizon: int,
    artifact_id: UUID,
    *,
    feature_mean: float = 1.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        request=SimpleNamespace(
            artifact_id=artifact_id,
            source=SimpleNamespace(corpus_id=_CORPUS_IDS[chain]),
        ),
        training_definition=SimpleNamespace(
            experiment=SimpleNamespace(
                horizon_blocks=horizon,
                context_blocks=3,
                ordered_features=("log_base_fee_per_gas", "gas_utilization"),
            )
        ),
        feature_state=SimpleNamespace(
            means=(feature_mean, 2.0),
            standard_deviations=(0.5, 0.25),
        ),
        target_state=SimpleNamespace(mean=3.0, standard_deviation=0.75),
    )


def _install_artifact_fakes(
    monkeypatch: pytest.MonkeyPatch,
    artifact_ids: dict[tuple[str, int], UUID],
    *,
    mismatched_feature_cell: tuple[str, int] | None = None,
) -> None:
    cells_by_id = {artifact_id: cell for cell, artifact_id in artifact_ids.items()}

    def load_artifact(storage_root: Path, artifact_id: UUID) -> tuple[object, nn.Module]:
        del storage_root
        chain, horizon = cells_by_id[artifact_id]
        feature_mean = 9.0 if (chain, horizon) == mismatched_feature_cell else 1.0
        return (
            _association(
                chain,
                horizon,
                artifact_id,
                feature_mean=feature_mean,
            ),
            nn.Identity(),
        )

    chain_ids = {
        corpus_id: mobile_export._CHAINS[chain] for chain, corpus_id in _CORPUS_IDS.items()
    }

    def load_corpus_request(storage_root: Path, corpus_id: UUID) -> object:
        del storage_root
        return SimpleNamespace(definition=SimpleNamespace(chain_id=chain_ids[corpus_id]))

    monkeypatch.setattr(mobile_export, "load_artifact", load_artifact)
    monkeypatch.setattr(mobile_export, "load_corpus_request", load_corpus_request)


def test_export_bundle_publishes_complete_stable_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roster_path = tmp_path / "MOBILE.yaml"
    artifact_ids = _write_roster(roster_path)
    _install_artifact_fakes(monkeypatch, artifact_ids)

    def export_model(cell: mobile_export._Cell, destination: Path) -> None:
        destination.write_bytes(cell.artifact_id.bytes)

    monkeypatch.setattr(mobile_export, "_export_model", export_model)
    output = tmp_path / "models"

    mobile_export.export_bundle(tmp_path / "storage", roster_path, output)

    expected_files = {"manifest.json"} | {
        f"{chain}-k{horizon}.pte"
        for chain in mobile_export._CHAINS
        for horizon in mobile_export._HORIZONS
    }
    assert {path.name for path in output.iterdir()} == expected_files
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "chains": {
            chain: {
                "context_blocks": 3,
                "features": [
                    {
                        "name": "log_base_fee_per_gas",
                        "mean": 1.0,
                        "standard_deviation": 0.5,
                    },
                    {
                        "name": "gas_utilization",
                        "mean": 2.0,
                        "standard_deviation": 0.25,
                    },
                ],
                "models": {
                    str(horizon): {
                        "artifact_id": str(artifact_ids[(chain, horizon)]),
                        "target": {
                            "mean": 3.0,
                            "standard_deviation": 0.75,
                        },
                    }
                    for horizon in mobile_export._HORIZONS
                },
            }
            for chain in mobile_export._CHAINS
        },
    }


def test_export_bundle_rejects_incomplete_roster(tmp_path: Path) -> None:
    roster_path = tmp_path / "MOBILE.yaml"
    _write_roster(roster_path)
    raw = yaml.safe_load(roster_path.read_text(encoding="utf-8"))
    del raw["polygon"][5]
    roster_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="at least 4 items"):
        mobile_export.export_bundle(
            tmp_path / "storage",
            roster_path,
            tmp_path / "models",
        )


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("horizon", "wrong horizon"),
        ("chain", "wrong chain"),
    ],
)
def test_export_bundle_rejects_artifact_association_mismatch(
    mismatch: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roster_path = tmp_path / "MOBILE.yaml"
    artifact_ids = _write_roster(roster_path)
    _install_artifact_fakes(monkeypatch, artifact_ids)

    if mismatch == "horizon":
        fake_load_artifact = mobile_export.load_artifact

        def load_artifact(storage_root: Path, artifact_id: UUID) -> tuple[object, nn.Module]:
            if artifact_id == artifact_ids[("ethereum", 2)]:
                return _association("ethereum", 3, artifact_id), nn.Identity()
            return fake_load_artifact(storage_root, artifact_id)

        monkeypatch.setattr(mobile_export, "load_artifact", load_artifact)
    else:
        fake_load_corpus_request = mobile_export.load_corpus_request

        def load_corpus_request(storage_root: Path, corpus_id: UUID) -> object:
            if corpus_id == _CORPUS_IDS["ethereum"]:
                return SimpleNamespace(definition=SimpleNamespace(chain_id=137))
            return fake_load_corpus_request(storage_root, corpus_id)

        monkeypatch.setattr(mobile_export, "load_corpus_request", load_corpus_request)

    with pytest.raises(ValueError, match=message):
        mobile_export.export_bundle(
            tmp_path / "storage",
            roster_path,
            tmp_path / "models",
        )
    assert not (tmp_path / "models").exists()


def test_export_bundle_rejects_feature_mismatch_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roster_path = tmp_path / "MOBILE.yaml"
    artifact_ids = _write_roster(roster_path)
    _install_artifact_fakes(
        monkeypatch,
        artifact_ids,
        mismatched_feature_cell=("ethereum", 3),
    )
    monkeypatch.setattr(
        mobile_export,
        "_export_model",
        lambda *args: pytest.fail(f"unexpected export: {args}"),
    )
    output = tmp_path / "models"

    with pytest.raises(ValueError, match="share one feature contract"):
        mobile_export.export_bundle(tmp_path / "storage", roster_path, output)

    assert not output.exists()


def test_export_bundle_cleans_scratch_after_export_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roster_path = tmp_path / "MOBILE.yaml"
    artifact_ids = _write_roster(roster_path)
    _install_artifact_fakes(monkeypatch, artifact_ids)
    failing_artifact_id = artifact_ids[("ethereum", 3)]

    def export_model(cell: mobile_export._Cell, destination: Path) -> None:
        if cell.artifact_id == failing_artifact_id:
            raise RuntimeError("lowering failed")
        destination.write_bytes(b"pte")

    monkeypatch.setattr(mobile_export, "_export_model", export_model)
    output = tmp_path / "models"

    with pytest.raises(RuntimeError, match="lowering failed"):
        mobile_export.export_bundle(tmp_path / "storage", roster_path, output)

    assert not output.exists()
    assert list(tmp_path.glob(".models.*")) == []


def test_export_bundle_preserves_existing_output(tmp_path: Path) -> None:
    roster_path = tmp_path / "MOBILE.yaml"
    output = tmp_path / "models"
    output.mkdir()
    marker = output / "occupied"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        mobile_export.export_bundle(tmp_path / "storage", roster_path, output)

    assert marker.read_text(encoding="utf-8") == "preserve"


def test_export_bundle_refuses_output_directory_created_during_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roster_path = tmp_path / "MOBILE.yaml"
    artifact_ids = _write_roster(roster_path)
    _install_artifact_fakes(monkeypatch, artifact_ids)
    output = tmp_path / "models"

    def export_model(cell: mobile_export._Cell, destination: Path) -> None:
        del cell
        destination.write_bytes(b"pte")
        output.mkdir(exist_ok=True)

    monkeypatch.setattr(mobile_export, "_export_model", export_model)

    with pytest.raises(FileExistsError):
        mobile_export.export_bundle(tmp_path / "storage", roster_path, output)

    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert list(tmp_path.glob(".models.*")) == []


class _Output(NamedTuple):
    action_logits: torch.Tensor
    minimum_fee_z: torch.Tensor


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action = nn.Linear(2, 2)
        self.regression = nn.Linear(2, 1)

    def forward(self, inputs: torch.Tensor) -> _Output:
        final = inputs[:, -1]
        return _Output(
            action_logits=self.action(final),
            minimum_fee_z=self.regression(final).squeeze(-1),
        )


def test_parity_rejects_matching_nonfinite_outputs() -> None:
    matching = (
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        torch.tensor([float("inf")], dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="ExecuTorch outputs must be finite"):
        mobile_export._assert_parity(
            matching,
            matching,
            target_mean=1.0,
            target_standard_deviation=0.5,
        )


def test_portable_program_fails_xnnpack_delegation_gate() -> None:
    model = mobile_export._NamedOutputWrapper(_TinyModel().eval())
    sample = torch.zeros((1, 3, 2), dtype=torch.float32)
    exported = torch.export.export(model, (sample,), strict=True)
    portable_program = mobile_export.to_edge_transform_and_lower(exported).to_executorch()

    with pytest.raises(ValueError, match="XnnpackBackend"):
        mobile_export._assert_xnnpack_delegation(portable_program)


def test_real_xnnpack_export_and_host_execution(tmp_path: Path) -> None:
    runtime = mobile_export.Runtime.get()
    assert runtime.backend_registry.is_available("XnnpackBackend")

    destination = tmp_path / "tiny.pte"
    mobile_export._export_model(
        mobile_export._Cell(
            artifact_id=_artifact_id(1),
            features=mobile_export._FeatureContract(
                context_blocks=3,
                names=("log_base_fee_per_gas", "gas_utilization"),
                means=(0.0, 0.0),
                standard_deviations=(1.0, 1.0),
            ),
            target_mean=1.0,
            target_standard_deviation=0.5,
            model=_TinyModel(),
        ),
        destination,
    )

    assert b"XnnpackBackend" in destination.read_bytes()
