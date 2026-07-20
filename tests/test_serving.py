from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import numpy as np
import pytest
import torch

from fable import serving
from fable.config import (
    AdamWMethod,
    BaselineSource,
    BlockWindow,
    ExperimentSemantics,
    FitMethod,
    LossDefinition,
    LstmDefinition,
    SelectedStudySource,
    TrainingDefinition,
    TrainRequest,
)

_CHAIN_IDS = {
    "https://ethereum.example": 1,
    "https://polygon.example": 137,
    "https://avalanche.example": 43114,
}


def _run_async(test: Callable[..., Any]) -> Callable[..., None]:
    @wraps(test)
    def run(*args: object, **kwargs: object) -> None:
        asyncio.run(test(*args, **kwargs))

    return run


def _uuid(index: int) -> UUID:
    return UUID(int=index, version=4)


def _write_config(root: Path) -> dict[tuple[str, int], UUID]:
    artifact_ids = {
        (chain, horizon): _uuid(index)
        for index, (chain, horizon) in enumerate(
            (
                (chain, horizon)
                for chain in ("ethereum", "polygon", "avalanche")
                for horizon in (2, 3, 4, 5)
            ),
            start=1,
        )
    }
    lines: list[str] = []
    for chain in ("ethereum", "polygon", "avalanche"):
        lines.extend(
            [f"{chain}:", f"  rpc_url: https://{chain}.example"]
            + [
                f"  k{horizon}_artifact_id: {artifact_ids[(chain, horizon)]}"
                for horizon in (2, 3, 4, 5)
            ]
        )
    (root / "SERVING.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return artifact_ids


def _experiment(*, context_blocks: int = 3, horizon_blocks: int = 2) -> ExperimentSemantics:
    return ExperimentSemantics(
        training_window=BlockWindow(
            first_parent_block=0,
            last_parent_block=4,
        ),
        validation_window=BlockWindow(
            first_parent_block=10,
            last_parent_block=12,
        ),
        context_blocks=context_blocks,
        horizon_blocks=horizon_blocks,
        ordered_features=("log_base_fee_per_gas", "gas_utilization"),
        loss=LossDefinition(
            classification_algorithm="cross_entropy",
            classification_weighting="unweighted",
            regression_algorithm="smooth_l1",
            regression_threshold=1.0,
            classification_scale=1.0,
            regression_scale=1.0,
        ),
    )


def _association(
    artifact_id: UUID,
    *,
    context_blocks: int = 3,
    horizon_blocks: int = 2,
    selected: bool = True,
) -> SimpleNamespace:
    experiment = _experiment(
        context_blocks=context_blocks,
        horizon_blocks=horizon_blocks,
    )
    if selected:
        source = SelectedStudySource(
            kind="selected_study",
            corpus_id=_uuid(100),
            study_id=_uuid(101),
            study_result_index=0,
            experiment=experiment,
        )
        request: object = TrainRequest(
            workflow="train",
            artifact_id=artifact_id,
            source=source,
        )
    else:
        request = TrainRequest(
            workflow="train",
            artifact_id=artifact_id,
            source=BaselineSource(
                kind="baseline",
                corpus_id=_uuid(100),
                training_definition=TrainingDefinition(
                    experiment=experiment,
                    model=LstmDefinition(
                        family="lstm",
                        hidden=2,
                        layers=1,
                        head_hidden=2,
                        dropout=0.0,
                    ),
                    optimizer=AdamWMethod(learning_rate=0.001, weight_decay=0.0),
                    training_batch=2,
                    fit=FitMethod(
                        accumulation=1,
                        gradient_clip_norm=1.0,
                        scheduler="none",
                        seed=1,
                        max_epochs=1,
                        validate_every_completed_epoch=1,
                        patience=0,
                        min_delta=0.0,
                        improvement="strict_lower",
                        restore="earliest_best",
                    ),
                ),
            ),
        )
    return SimpleNamespace(request=request, feature_state=object())


def _block(number: int) -> dict[str, object]:
    return {
        "number": number,
        "timestamp": 1_000 + number,
        "baseFeePerGas": 20 + number,
        "gasUsed": 50,
        "gasLimit": 100,
        "transactions": [b"transaction"],
    }


class _FakeProvider:
    instances: list[_FakeProvider] = []

    def __init__(self, endpoint_uri: str) -> None:
        self.endpoint_uri = endpoint_uri
        self.disconnect_calls = 0
        self.instances.append(self)

    async def disconnect(self) -> None:
        self.disconnect_calls += 1


class _FakeMiddlewareOnion:
    def __init__(self) -> None:
        self.injections: list[tuple[object, int]] = []

    def inject(self, middleware: object, *, layer: int) -> None:
        self.injections.append((middleware, layer))


class _FakeEth:
    def __init__(self, endpoint_uri: str) -> None:
        self.endpoint_uri = endpoint_uri
        self.chain_id_calls = 0
        self.block_calls: list[int | str] = []

    @property
    def chain_id(self) -> Any:
        self.chain_id_calls += 1

        async def _value() -> int:
            return _CHAIN_IDS[self.endpoint_uri]

        return _value()

    async def get_block(self, number: int | str, full_transactions: bool) -> dict[str, object]:
        assert full_transactions is False
        self.block_calls.append(number)
        if number == "latest":
            return _block(12)
        assert isinstance(number, int)
        return _block(number)


class _FakeBatch:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    async def __aenter__(self) -> _FakeBatch:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def add(self, request: Any) -> None:
        self.requests.append(request)

    async def async_execute(self) -> list[dict[str, object]]:
        return [await request for request in self.requests]


class _FakeWeb3:
    instances: list[_FakeWeb3] = []

    def __init__(self, provider: _FakeProvider) -> None:
        self.provider = provider
        self.eth = _FakeEth(provider.endpoint_uri)
        self.middleware_onion = _FakeMiddlewareOnion()
        self.instances.append(self)

    def batch_requests(self) -> _FakeBatch:
        return _FakeBatch()


class _FakeModel:
    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []
        self.output = object()

    def __call__(self, values: torch.Tensor) -> object:
        self.inputs.append(values)
        return self.output


async def _post(app: Any, payload: dict[str, object]) -> tuple[int, dict[str, Any]]:
    sent: list[dict[str, Any]] = []
    request_sent = False

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {
                "type": "http.request",
                "body": json.dumps(payload).encode(),
                "more_body": False,
            }
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/inference",
            "raw_path": b"/inference",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json")],
            "client": ("test", 1),
            "server": ("test", 80),
        },
        receive,
        send,
    )
    start = next(message for message in sent if message["type"] == "http.response.start")
    body = b"".join(
        message.get("body", b"") for message in sent if message["type"] == "http.response.body"
    )
    return start["status"], json.loads(body)


def _install_web3(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeProvider.instances.clear()
    _FakeWeb3.instances.clear()
    monkeypatch.setattr(serving, "_AsyncHTTPProvider", _FakeProvider)
    monkeypatch.setattr(serving, "_AsyncWeb3", _FakeWeb3)


@_run_async
async def test_serves_one_selected_artifact_context_and_closes_clients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_ids = _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    _install_web3(monkeypatch)
    artifact_id = artifact_ids[("avalanche", 5)]
    model = _FakeModel()
    association = _association(artifact_id, horizon_blocks=5)
    transformed = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    load_calls: list[tuple[Path, UUID]] = []
    transform_calls: list[tuple[Any, tuple[str, ...], object]] = []
    decode_calls: list[object] = []
    config_loads = 0
    real_config_loader: Callable[[], object] = serving._load_serving_config

    def load_config() -> object:
        nonlocal config_loads
        config_loads += 1
        return real_config_loader()

    def load_artifact(storage_root: Path, selected_id: UUID) -> tuple[object, _FakeModel]:
        load_calls.append((storage_root, selected_id))
        return association, model

    def transform(
        blocks: Any,
        *,
        ordered_features: tuple[str, ...],
        state: object,
    ) -> np.ndarray[Any, Any]:
        transform_calls.append((blocks, ordered_features, state))
        return transformed

    def decode(output: object) -> torch.Tensor:
        decode_calls.append(output)
        return torch.tensor([1])

    monkeypatch.setattr(serving, "_load_serving_config", load_config)
    monkeypatch.setattr(serving, "_load_artifact", load_artifact)
    monkeypatch.setattr(serving, "_transform_feature_rows", transform)
    monkeypatch.setattr(serving, "_decode_action", decode)

    app = serving.create_app()
    async with app.router.lifespan_context(app):
        assert config_loads == 1
        assert [provider.endpoint_uri for provider in _FakeProvider.instances] == [
            "https://ethereum.example",
            "https://polygon.example",
            "https://avalanche.example",
        ]
        assert [web3.eth.block_calls for web3 in _FakeWeb3.instances] == [[], [], []]
        assert [web3.eth.chain_id_calls for web3 in _FakeWeb3.instances] == [0, 0, 0]
        assert [web3.middleware_onion.injections for web3 in _FakeWeb3.instances] == [
            [],
            [(serving._ExtraDataToPOAMiddleware, 0)],
            [(serving._ExtraDataToPOAMiddleware, 0)],
        ]

        status, body = await _post(app, {"chain": "avalanche", "K": 5})

        assert status == 200
        assert body == {
            "head_block": 12,
            "selected_action_k": 1,
            "target_block": 14,
        }
        assert config_loads == 1
        assert load_calls == [(tmp_path / "storage", artifact_id)]
        avalanche = _FakeWeb3.instances[2]
        assert avalanche.eth.chain_id_calls == 1
        assert avalanche.eth.block_calls == ["latest", 10, 11]
        assert len(transform_calls) == 1
        blocks, ordered_features, feature_state = transform_calls[0]
        assert blocks.to_polars().to_dict(as_series=False) == {
            "block_number": [10, 11, 12],
            "timestamp": [1010, 1011, 1012],
            "chain_id": [43114, 43114, 43114],
            "base_fee_per_gas": [30, 31, 32],
            "gas_used": [50, 50, 50],
            "gas_limit": [100, 100, 100],
            "tx_count": [1, 1, 1],
        }
        assert ordered_features == ("log_base_fee_per_gas", "gas_utilization")
        assert feature_state is association.feature_state
        assert len(model.inputs) == 1
        torch.testing.assert_close(model.inputs[0], torch.from_numpy(transformed).unsqueeze(0))
        assert decode_calls == [model.output]

    assert [provider.disconnect_calls for provider in _FakeProvider.instances] == [1, 1, 1]


@_run_async
async def test_request_is_strict_and_forbids_extra_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    _install_web3(monkeypatch)
    monkeypatch.setattr(
        serving,
        "_load_artifact",
        lambda *_: pytest.fail("invalid requests must not load artifacts"),
    )
    app = serving.create_app()

    async with app.router.lifespan_context(app):
        status, _ = await _post(
            app,
            {"chain": "ethereum", "K": 2, "unexpected": True},
        )

    assert status == 422


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("source", "SelectedStudySource"),
        ("horizon", "request K must match"),
        ("chain", "chain ID mismatch"),
        ("live", "base_fee_per_gas"),
    ],
)
@_run_async
async def test_rejects_artifact_and_chain_mismatches(
    failure: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_ids = _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    _install_web3(monkeypatch)
    artifact_id = artifact_ids[("ethereum", 2)]
    association = _association(
        artifact_id,
        horizon_blocks=3 if failure == "horizon" else 2,
        selected=failure != "source",
    )
    monkeypatch.setattr(serving, "_load_artifact", lambda *_: (association, _FakeModel()))
    if failure == "chain":
        monkeypatch.setitem(_CHAIN_IDS, "https://ethereum.example", 2)
    if failure == "live":
        get_block = _FakeEth.get_block

        async def malformed_live_block(
            eth: _FakeEth,
            number: int | str,
            full_transactions: bool,
        ) -> dict[str, object]:
            block = await get_block(eth, number, full_transactions)
            if number == "latest":
                block["baseFeePerGas"] = 0
            return block

        monkeypatch.setattr(_FakeEth, "get_block", malformed_live_block)
    app = serving.create_app()

    with pytest.raises(ValueError, match=message):
        async with app.router.lifespan_context(app):
            await _post(app, {"chain": "ethereum", "K": 2})


@_run_async
async def test_artifact_load_failure_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    _install_web3(monkeypatch)

    failure = RuntimeError("checkpoint owner failed")

    def fail(*_: object) -> None:
        raise failure

    monkeypatch.setattr(serving, "_load_artifact", fail)
    app = serving.create_app()

    with pytest.raises(RuntimeError) as caught:
        async with app.router.lifespan_context(app):
            await _post(app, {"chain": "ethereum", "K": 2})

    assert caught.value is failure
