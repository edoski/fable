"""Minimal Alchemy JSON-RPC client used only during block enrichment."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass


def _hex_to_int(value: str) -> int:
    return int(value, 16)


@dataclass(slots=True)
class AlchemyRpcClient:
    url: str
    timeout_seconds: float = 30.0

    def get_block_gas_limits(self, block_numbers: list[int]) -> dict[int, int]:
        payload = [
            {
                "jsonrpc": "2.0",
                "method": "eth_getBlockByNumber",
                "params": [hex(block_number), False],
                "id": index,
            }
            for index, block_number in enumerate(block_numbers, start=1)
        ]
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))

        by_id = {item["id"]: item["result"] for item in body}
        return {
            _hex_to_int(by_id[index]["number"]): _hex_to_int(by_id[index]["gasLimit"])
            for index in range(1, len(payload) + 1)
        }
