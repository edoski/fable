from __future__ import annotations

from typing import Any, cast

import pytest
from web3.exceptions import BlockNotFound
from web3.middleware import ExtraDataToPOAMiddleware

from spice.acquisition.provider import build_web3
from spice.acquisition.rpc import BlockPullPlan, BlockRange, TimestampRange, Web3BlockClient
from tests.support import make_chain_config, make_provider_config


def test_build_web3_uses_configured_http_endpoint() -> None:
    web3 = build_web3(
        make_provider_config(),
        make_chain_config(),
    )

    assert web3.provider is not None
    assert cast(Any, web3.provider).endpoint_uri == "https://rpc.example.test"


@pytest.mark.parametrize(
    ("endpoint", "expected_suffix"),
    [
        ("file:///tmp/geth.ipc", "/tmp/geth.ipc"),
        ("/tmp/geth.ipc", "/tmp/geth.ipc"),
    ],
)
def test_build_web3_supports_ipc_endpoints(endpoint: str, expected_suffix: str) -> None:
    web3 = build_web3(
        make_provider_config(endpoint=endpoint),
        make_chain_config(),
    )

    assert web3.provider is not None
    assert cast(Any, web3.provider).ipc_path.endswith(expected_suffix)


def test_build_web3_injects_poa_middleware_for_poa_extra_data_chains() -> None:
    web3 = build_web3(
        make_provider_config(),
        make_chain_config(uses_poa_extra_data=True),
    )

    assert ExtraDataToPOAMiddleware in web3.middleware_onion

def test_web3_block_client_reads_canonical_block_rows(monkeypatch) -> None:
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
        def get_block(
            self,
            block_number: int,
            _full_transactions: bool = False,
        ) -> dict[str, int]:
            return {
                "number": block_number,
                "timestamp": 1_700_000_000 + block_number,
                "baseFeePerGas": 1_000_000_000 + block_number,
                "gasUsed": 20_000_000 + block_number,
                "gasLimit": 30_000_000 + block_number,
            }

    class FakeWeb3:
        eth = FakeEth()

        def batch_requests(self) -> FakeBatch:
            return FakeBatch()

    monkeypatch.setattr(
        "spice.acquisition.rpc.build_web3",
        lambda _provider, _chain: FakeWeb3(),
    )

    client = Web3BlockClient(make_provider_config(), make_chain_config())
    rows = client.get_block_rows([1, 2])

    assert rows == [
        {
            "block_number": 1,
            "timestamp": 1_700_000_001,
            "base_fee_per_gas": 1_000_000_001,
            "gas_used": 20_000_001,
            "chain_id": 1,
            "gas_limit": 30_000_001,
        },
        {
            "block_number": 2,
            "timestamp": 1_700_000_002,
            "base_fee_per_gas": 1_000_000_002,
            "gas_used": 20_000_002,
            "chain_id": 1,
            "gas_limit": 30_000_002,
        },
    ]


def test_web3_block_client_uses_latest_block_for_head_resolution(monkeypatch) -> None:
    timestamps = {
        0: 100,
        1: 112,
        2: 124,
        3: 136,
    }

    class FakeEth:
        block_number = 4

        def get_block(
            self,
            block_number: int | str,
            _full_transactions: bool = False,
        ) -> dict[str, int]:
            if block_number == "latest":
                block_number = 3
            elif int(block_number) == 4:
                raise BlockNotFound("stale numeric head")
            number = int(block_number)
            return {
                "number": number,
                "timestamp": timestamps[number],
                "baseFeePerGas": 1,
                "gasUsed": 1,
                "gasLimit": 1,
            }

    class FakeWeb3:
        eth = FakeEth()

        def batch_requests(self):
            raise AssertionError("batch path not used in binary search")

    monkeypatch.setattr(
        "spice.acquisition.rpc.build_web3",
        lambda _provider, _chain: FakeWeb3(),
    )

    client = Web3BlockClient(make_provider_config(), make_chain_config())

    assert client.find_first_block_at_or_after(100) == 0
    assert client.find_first_block_at_or_after(113) == 2
    assert client.find_first_block_at_or_after(149) == 4
    assert client.resolve_block_range(TimestampRange(start=112, end=136)) == BlockRange(
        start=1,
        end=3,
    )


def test_web3_block_client_plans_history_by_exact_block_count(monkeypatch) -> None:
    class FakeEth:
        def get_block(
            self,
            block_number: int | str,
            _full_transactions: bool = False,
        ) -> dict[str, int]:
            if block_number == "latest":
                block_number = 6
            number = int(block_number)
            return {
                "number": number,
                "timestamp": 100 + number * 12,
                "baseFeePerGas": 1,
                "gasUsed": 1,
                "gasLimit": 1,
            }

    class FakeWeb3:
        eth = FakeEth()

        def batch_requests(self):
            raise AssertionError("batch path not used in planning")

    monkeypatch.setattr(
        "spice.acquisition.rpc.build_web3",
        lambda _provider, _chain: FakeWeb3(),
    )

    client = Web3BlockClient(make_provider_config(), make_chain_config())
    plan = client.plan_history_window(
        end_timestamp=136,
        required_history_blocks=2,
        chunk_size=5,
    )

    assert plan == BlockPullPlan(
        window=TimestampRange(start=112, end=136),
        block_range=BlockRange(start=1, end=3),
        expected_rows=2,
        expected_files=1,
    )


def test_web3_block_client_expands_history_plan_by_missing_blocks(monkeypatch) -> None:
    class FakeEth:
        def get_block(
            self,
            block_number: int | str,
            _full_transactions: bool = False,
        ) -> dict[str, int]:
            if block_number == "latest":
                block_number = 40
            number = int(block_number)
            return {
                "number": number,
                "timestamp": 100 + number * 12,
                "baseFeePerGas": 1,
                "gasUsed": 1,
                "gasLimit": 1,
            }

    class FakeWeb3:
        eth = FakeEth()

        def batch_requests(self):
            raise AssertionError("batch path not used in planning")

    monkeypatch.setattr(
        "spice.acquisition.rpc.build_web3",
        lambda _provider, _chain: FakeWeb3(),
    )

    client = Web3BlockClient(make_provider_config(), make_chain_config())
    current = BlockPullPlan(
        window=TimestampRange(start=340, end=460),
        block_range=BlockRange(start=20, end=30),
        expected_rows=10,
        expected_files=2,
    )

    expanded = client.expand_history_plan(
        current,
        observed_row_count=7,
        required_history_blocks=10,
        chunk_size=5,
    )

    assert expanded == BlockPullPlan(
        window=TimestampRange(start=244, end=460),
        block_range=BlockRange(start=12, end=30),
        expected_rows=18,
        expected_files=4,
    )
