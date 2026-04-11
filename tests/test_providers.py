from __future__ import annotations

from typing import Any, cast

from web3.middleware import ExtraDataToPOAMiddleware

from spice.acquisition.provider import build_web3, redact_sensitive_text
from spice.acquisition.rpc import Web3BlockClient
from tests.support import make_chain_config, make_provider_config


def test_build_web3_uses_configured_endpoint() -> None:
    web3 = build_web3(
        make_provider_config(),
        make_chain_config(),
    )

    assert web3.provider is not None
    assert cast(Any, web3.provider).endpoint_uri == "https://rpc.example.test"


def test_build_web3_injects_poa_middleware_for_poa_extra_data_chains() -> None:
    web3 = build_web3(
        make_provider_config(),
        make_chain_config(uses_poa_extra_data=True),
    )

    assert ExtraDataToPOAMiddleware in web3.middleware_onion


def test_redact_sensitive_text_masks_endpoint() -> None:
    text = "rpc=https://rpc.example.test"

    assert redact_sensitive_text(text, make_provider_config()) == "rpc=***"


def test_web3_block_client_reads_gas_limits(monkeypatch) -> None:
    class FakeBatch:
        def __init__(self) -> None:
            self.blocks: list[dict[str, int]] = []

        def __enter__(self) -> FakeBatch:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def add(self, block: dict[str, int]) -> None:
            self.blocks.append(block)

        def execute(self) -> list[dict[str, int]]:
            return self.blocks

    class FakeEth:
        def get_block(self, block_number: int) -> dict[str, int]:
            return {"gasLimit": 30_000_000 + block_number}

    class FakeWeb3:
        eth = FakeEth()

        def batch_requests(self) -> FakeBatch:
            return FakeBatch()

    monkeypatch.setattr(
        "spice.acquisition.rpc.build_web3",
        lambda _provider, _chain: FakeWeb3(),
    )

    client = Web3BlockClient(make_provider_config(), make_chain_config())

    assert client.get_block_gas_limits([1, 2]) == {1: 30_000_001, 2: 30_000_002}
