"""web3.py-backed helpers for enrichment hydration."""

from __future__ import annotations

from dataclasses import dataclass, field

from web3 import Web3

from ..core.config import ChainConfig, ProviderConfig
from .provider import build_web3


@dataclass(slots=True)
class Web3BlockClient:
    provider: ProviderConfig
    chain: ChainConfig
    _web3: Web3 = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._web3 = build_web3(self.provider, self.chain)

    def get_block_gas_limits(self, block_numbers: list[int]) -> dict[int, int]:
        if not block_numbers:
            return {}

        with self._web3.batch_requests() as batch:
            for block_number in block_numbers:
                batch.add(self._web3.eth.get_block(block_number))
            blocks = batch.execute()

        if len(blocks) != len(block_numbers):
            raise RuntimeError(
                f"Expected {len(block_numbers)} block responses, got {len(blocks)}"
            )

        gas_limits: dict[int, int] = {}
        for block_number, block in zip(block_numbers, blocks, strict=True):
            gas_limit = block.get("gasLimit")
            if gas_limit is None:
                raise RuntimeError(f"Missing gasLimit for block {block_number}")
            gas_limits[block_number] = int(gas_limit)
        return gas_limits
