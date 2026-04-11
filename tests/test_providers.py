from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from web3.middleware import ExtraDataToPOAMiddleware

from spice.acquisition.provider import build_web3, redact_sensitive_text
from spice.acquisition.rpc import Web3BlockClient
from spice.core.config import ChainConfig, ChainName, ProviderConfig, RpcProviderName


def _provider(endpoint: str = "https://rpc.example.test") -> ProviderConfig:
    return ProviderConfig(
        name=RpcProviderName.DIRECT,
        endpoints={"ethereum": endpoint},
        references={"ethereum": "$ETHEREUM_RPC_URL"},
    )


def test_build_web3_uses_configured_endpoint() -> None:
    web3 = build_web3(
        _provider(),
        ChainConfig(name=ChainName.ETHEREUM, chain_id=1),
    )

    assert web3.provider is not None
    assert cast(Any, web3.provider).endpoint_uri == "https://rpc.example.test"


def test_build_web3_injects_poa_middleware_for_poa_extra_data_chains() -> None:
    web3 = build_web3(
        _provider(),
        ChainConfig(name=ChainName.ETHEREUM, chain_id=1, uses_poa_extra_data=True),
    )

    assert ExtraDataToPOAMiddleware in web3.middleware_onion


def test_redact_sensitive_text_masks_endpoint() -> None:
    text = "rpc=https://rpc.example.test"

    assert redact_sensitive_text(text, _provider()) == "rpc=***"


def test_web3_block_client_reads_gas_limits(monkeypatch) -> None:
    class FakeEth:
        def get_block(self, block_number: int) -> dict[str, int]:
            return {"gasLimit": 30_000_000 + block_number}

    monkeypatch.setattr(
        "spice.acquisition.rpc.build_web3",
        lambda _provider, _chain: SimpleNamespace(eth=FakeEth()),
    )

    client = Web3BlockClient(_provider(), ChainConfig(name=ChainName.ETHEREUM, chain_id=1))

    assert client.get_block_gas_limits([1, 2]) == {1: 30_000_001, 2: 30_000_002}
