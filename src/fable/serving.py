"""Stateless artifact inference over live chain heads."""

from __future__ import annotations

import json as _json
from collections.abc import AsyncIterator as _AsyncIterator
from collections.abc import Mapping as _Mapping
from collections.abc import Sized as _Sized
from contextlib import asynccontextmanager as _asynccontextmanager
from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path
from typing import Annotated as _Annotated
from typing import Literal as _Literal
from typing import SupportsInt as _SupportsInt
from typing import cast as _cast
from uuid import UUID as _UUID

import polars as _pl
import torch as _torch
import yaml as _yaml
from fastapi import FastAPI as _FastAPI
from fastapi import Request as _Request
from pydantic import (
    UUID4 as _UUID4,
)
from pydantic import (
    BaseModel as _BaseModel,
)
from pydantic import (
    ConfigDict as _ConfigDict,
)
from pydantic import (
    Field as _Field,
)
from web3 import AsyncHTTPProvider as _AsyncHTTPProvider
from web3 import AsyncWeb3 as _AsyncWeb3
from web3.middleware import ExtraDataToPOAMiddleware as _ExtraDataToPOAMiddleware

from .config import CorpusDefinition as _CorpusDefinition
from .config import SelectedStudySource as _SelectedStudySource
from .corpus import BlockFrame as _BlockFrame
from .environment import resolve_storage_root as _resolve_storage_root
from .min_block_fee import decode_action as _decode_action
from .modeling import load_artifact as _load_artifact
from .temporal.features import transform_feature_rows as _transform_feature_rows

_Chain = _Literal["ethereum", "polygon", "avalanche"]
_Horizon = _Literal[2, 3, 4, 5]
_NonEmptyString = _Annotated[str, _Field(min_length=1)]
_NonNegativeInt = _Annotated[int, _Field(strict=True, ge=0)]


class _ChainConfig(_BaseModel):
    model_config = _ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        strict=True,
    )

    rpc_url: _NonEmptyString
    k2_artifact_id: _UUID4
    k3_artifact_id: _UUID4
    k4_artifact_id: _UUID4
    k5_artifact_id: _UUID4

    def artifact_id(self, horizon: _Horizon) -> _UUID:
        return (
            self.k2_artifact_id,
            self.k3_artifact_id,
            self.k4_artifact_id,
            self.k5_artifact_id,
        )[horizon - 2]


class _ServingConfig(_BaseModel):
    model_config = _ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        strict=True,
    )

    ethereum: _ChainConfig
    polygon: _ChainConfig
    avalanche: _ChainConfig


class _InferenceRequest(_BaseModel):
    model_config = _ConfigDict(extra="forbid", strict=True)

    chain: _Chain
    K: _Horizon


class _InferenceResponse(_BaseModel):
    model_config = _ConfigDict(extra="forbid", strict=True)

    head_block: _NonNegativeInt
    selected_action_k: _NonNegativeInt
    target_block: _NonNegativeInt


@_dataclass(frozen=True, slots=True)
class _ServingState:
    storage_root: _Path
    config: _ServingConfig
    ethereum: _AsyncWeb3
    polygon: _AsyncWeb3
    avalanche: _AsyncWeb3


def _load_serving_config() -> _ServingConfig:
    raw = _yaml.safe_load(_Path("SERVING.yaml").read_text(encoding="utf-8"))
    return _ServingConfig.model_validate_json(_json.dumps(raw), strict=True)


@_asynccontextmanager
async def _lifespan(app: _FastAPI) -> _AsyncIterator[None]:
    storage_root = _resolve_storage_root()
    config = _load_serving_config()
    ethereum = _AsyncWeb3(_AsyncHTTPProvider(config.ethereum.rpc_url))
    polygon = _AsyncWeb3(_AsyncHTTPProvider(config.polygon.rpc_url))
    avalanche = _AsyncWeb3(_AsyncHTTPProvider(config.avalanche.rpc_url))
    polygon.middleware_onion.inject(_ExtraDataToPOAMiddleware, layer=0)
    avalanche.middleware_onion.inject(_ExtraDataToPOAMiddleware, layer=0)
    app.state._serving = _ServingState(
        storage_root=storage_root,
        config=config,
        ethereum=ethereum,
        polygon=polygon,
        avalanche=avalanche,
    )
    try:
        yield
    finally:
        await ethereum.provider.disconnect()
        await polygon.provider.disconnect()
        await avalanche.provider.disconnect()


def _serving_cell(
    state: _ServingState,
    chain: _Chain,
    horizon: _Horizon,
) -> tuple[_AsyncWeb3, int, _UUID]:
    match chain:
        case "ethereum":
            return state.ethereum, 1, state.config.ethereum.artifact_id(horizon)
        case "polygon":
            return state.polygon, 137, state.config.polygon.artifact_id(horizon)
        case "avalanche":
            return state.avalanche, 43114, state.config.avalanche.artifact_id(horizon)


def _quantity(value: object) -> int:
    if isinstance(value, str) and value.startswith("0x"):
        return int(value, 16)
    return int(_cast(_SupportsInt | str | bytes | bytearray, value))


def _live_row(block: object, chain_id: int) -> dict[str, int]:
    values = _cast(_Mapping[str, object], block)
    return {
        "block_number": _quantity(values["number"]),
        "timestamp": _quantity(values["timestamp"]),
        "chain_id": chain_id,
        "base_fee_per_gas": _quantity(values["baseFeePerGas"]),
        "gas_used": _quantity(values["gasUsed"]),
        "gas_limit": _quantity(values["gasLimit"]),
        "tx_count": len(_cast(_Sized, values["transactions"])),
    }


async def _infer(request: _InferenceRequest, state: _ServingState) -> _InferenceResponse:
    client, expected_chain_id, artifact_id = _serving_cell(state, request.chain, request.K)
    association, model = _load_artifact(state.storage_root, artifact_id)
    source = association.request.source
    if not isinstance(source, _SelectedStudySource):
        raise ValueError("serving artifacts must contain a SelectedStudySource")
    experiment = source.experiment
    if request.K != experiment.horizon_blocks:
        raise ValueError("request K must match the artifact horizon")

    chain_id = await client.eth.chain_id
    if chain_id != expected_chain_id:
        raise ValueError("provider chain ID mismatch")
    latest = _live_row(await client.eth.get_block("latest", False), chain_id)
    head_block = latest["block_number"]
    context_blocks = experiment.context_blocks
    if head_block < context_blocks - 1:
        raise ValueError("chain head is too early for the artifact context")
    first_block = head_block - context_blocks + 1
    async with client.batch_requests() as batch:
        for block_number in range(first_block, head_block):
            batch.add(client.eth.get_block(block_number, False))
        predecessors = await batch.async_execute()
    if len(predecessors) != context_blocks - 1:
        raise ValueError("provider returned the wrong predecessor count")
    rows = [_live_row(block, chain_id) for block in predecessors]
    rows.append(latest)
    blocks = _BlockFrame(
        _pl.DataFrame(rows),
        _CorpusDefinition(
            chain_id=chain_id,
            first_block=first_block,
            last_block=head_block,
        ),
    )
    model_input = _torch.from_numpy(
        _transform_feature_rows(
            blocks,
            ordered_features=experiment.ordered_features,
            state=association.feature_state,
        )
    ).unsqueeze(0)
    with _torch.inference_mode():
        output = model(model_input)
    selected_action_k = int(_decode_action(output).item())
    return _InferenceResponse(
        head_block=head_block,
        selected_action_k=selected_action_k,
        target_block=head_block + 1 + selected_action_k,
    )


def create_app() -> _FastAPI:
    app = _FastAPI(
        title="FABLE Inference API",
        openapi_url=None,
        docs_url=None,
        redoc_url=None,
        redirect_slashes=False,
        lifespan=_lifespan,
    )

    @app.post("/inference", response_model=_InferenceResponse)
    async def inference(payload: _InferenceRequest, request: _Request) -> _InferenceResponse:
        return await _infer(payload, _cast(_ServingState, request.app.state._serving))

    return app


__all__ = ["create_app"]
