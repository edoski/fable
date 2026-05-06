from __future__ import annotations

import asyncio

import pytest
from web3.exceptions import Web3RPCError

from spice.acquisition.errors import UnsupportedAcquisitionSourceError
from spice.acquisition.rpc.client import BlockRpcClient
from spice.config.models import ChainRuntimeSpec, ChainSpec, ResolvedRpcEndpointConfig
from spice.corpus.metadata import CorpusAcquisitionSourceRequirements


def _endpoint() -> ResolvedRpcEndpointConfig:
    return ResolvedRpcEndpointConfig(
        provider_name="test",
        url="https://rpc.example",
        reference="test:ethereum",
        timeout_seconds=5.0,
        retry_count=0,
        backoff_factor=0.0,
    )


def _chain() -> ChainSpec:
    return ChainSpec(
        name="ethereum",
        runtime=ChainRuntimeSpec(
            chain_id=1,
            uses_poa_extra_data=False,
            nominal_block_time_seconds=12.0,
        ),
    )


def _requirements(*, enrichments: frozenset[str]) -> CorpusAcquisitionSourceRequirements:
    return CorpusAcquisitionSourceRequirements(
        required_columns=frozenset({"block_number", "timestamp", "chain_id"}),
        optional_enrichments=enrichments,
        temporal_unit="block",
        ordering_key="block_number",
        partition_key="chain_id",
    )


class _FakeProvider:
    async def disconnect(self) -> None:
        return None


class _FakeWeb3:
    def __init__(self, fee_history_result: object) -> None:
        self.provider = _FakeProvider()
        self.eth = _FakeEth(fee_history_result)


class _FakeEth:
    def __init__(self, fee_history_result: object) -> None:
        self.fee_history_result = fee_history_result

    async def fee_history(self, *args, **kwargs):
        del args, kwargs
        if isinstance(self.fee_history_result, Exception):
            raise self.fee_history_result
        return self.fee_history_result


def _patch_web3(monkeypatch, fee_history_result: object) -> None:
    monkeypatch.setattr(
        "spice.acquisition.rpc.client.build_async_web3",
        lambda rpc_endpoint, chain: _FakeWeb3(fee_history_result),
    )


def test_rpc_client_maps_source_requirements_to_priority_fee_fetching(monkeypatch) -> None:
    _patch_web3(monkeypatch, {})

    baseline = BlockRpcClient(
        _endpoint(),
        _chain(),
        _requirements(enrichments=frozenset()),
    )
    priority = BlockRpcClient(
        _endpoint(),
        _chain(),
        _requirements(enrichments=frozenset({"priority_fee_percentiles"})),
    )

    assert baseline._include_priority_fee_percentiles is False
    assert priority._include_priority_fee_percentiles is True


def test_rpc_client_fails_when_required_fee_history_is_unsupported(monkeypatch) -> None:
    _patch_web3(
        monkeypatch,
        Web3RPCError(
            "method not found",
            rpc_response={"error": {"code": -32601, "message": "method not found"}},
        ),
    )
    client = BlockRpcClient(
        _endpoint(),
        _chain(),
        _requirements(enrichments=frozenset({"priority_fee_percentiles"})),
    )

    with pytest.raises(UnsupportedAcquisitionSourceError, match="eth_feeHistory"):
        asyncio.run(client._fee_history_rows(100, 102))


def test_rpc_client_rejects_unknown_source_enrichment(monkeypatch) -> None:
    _patch_web3(monkeypatch, {})

    with pytest.raises(UnsupportedAcquisitionSourceError, match="unknown_enrichment"):
        BlockRpcClient(
            _endpoint(),
            _chain(),
            _requirements(enrichments=frozenset({"unknown_enrichment"})),
        )


def test_rpc_client_rejects_malformed_fee_history_reward(monkeypatch) -> None:
    _patch_web3(monkeypatch, {"oldestBlock": 100, "reward": [[1, 2]]})
    client = BlockRpcClient(
        _endpoint(),
        _chain(),
        _requirements(enrichments=frozenset({"priority_fee_percentiles"})),
    )

    with pytest.raises(UnsupportedAcquisitionSourceError, match="reward row 0 length"):
        asyncio.run(client._fee_history_rows(100, 101))


def test_rpc_client_parses_fee_history_rows(monkeypatch) -> None:
    _patch_web3(
        monkeypatch,
        {
            "oldestBlock": 100,
            "reward": [
                ["0x1", "0x2", "0x4"],
                ["0x2", "0x3", "0x5"],
            ],
        },
    )
    client = BlockRpcClient(
        _endpoint(),
        _chain(),
        _requirements(enrichments=frozenset({"priority_fee_percentiles"})),
    )

    rows = asyncio.run(client._fee_history_rows(100, 102))

    assert [row.priority_fee_p50 for row in rows] == [2, 3]
