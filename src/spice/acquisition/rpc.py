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
        gas_limits: dict[int, int] = {}
        for block_number in block_numbers:
            block = self._web3.eth.get_block(block_number)
            gas_limit = block.get("gasLimit")
            if gas_limit is None:
                raise RuntimeError(f"Missing gasLimit for block {block_number}")
            gas_limits[block_number] = int(gas_limit)
        return gas_limits
