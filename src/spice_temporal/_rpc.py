"""Minimal generic JSON-RPC client used during block enrichment."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


def _hex_to_int(value: str) -> int:
    return int(value, 16)


def _require_hex_field(payload: dict[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise TypeError(f"JSON-RPC block result field {field} must be a hex string")
    return value


def _coerce_error_code(error: dict[str, object]) -> int | None:
    value = error.get("code")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _require_response_id(item: dict[str, object]) -> int:
    value = item.get("id")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("JSON-RPC batch response id must be an integer")
    return value


@dataclass(slots=True)
class JsonRpcClient:
    url: str
    timeout_seconds: float = 30.0
    max_retries: int = 5
    retry_backoff_seconds: float = 1.0

    def _post(self, payload: list[dict[str, object]]) -> list[dict[str, object]]:
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "User-Agent": "Mozilla/5.0"
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        if not isinstance(body, list):
            raise TypeError("JSON-RPC batch response must be a list")
        if not all(isinstance(item, dict) for item in body):
            raise TypeError("JSON-RPC batch response items must be objects")
        return body

    def get_block_gas_limits(self, block_numbers: list[int]) -> dict[int, int]:
        pending = list(block_numbers)
        gas_limits: dict[int, int] = {}
        delay_seconds = self.retry_backoff_seconds

        for attempt in range(self.max_retries + 1):
            if not pending:
                return gas_limits

            payload = [
                {
                    "jsonrpc": "2.0",
                    "method": "eth_getBlockByNumber",
                    "params": [hex(block_number), False],
                    "id": index,
                }
                for index, block_number in enumerate(pending, start=1)
            ]
            requested_by_id = {
                index: block_number for index, block_number in enumerate(pending, start=1)
            }
            retry_blocks: list[int] = []

            try:
                response_items = self._post(payload)
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    pending = list(pending)
                    if attempt < self.max_retries:
                        time.sleep(delay_seconds)
                        delay_seconds *= 2
                        continue
                    break
                raise

            for item in response_items:
                if "id" not in item:
                    raise ValueError("JSON-RPC batch response item is missing id")
                response_id = _require_response_id(item)
                if response_id not in requested_by_id:
                    raise ValueError(f"Unexpected JSON-RPC response id: {response_id}")

                if "result" in item:
                    result = item["result"]
                    if not isinstance(result, dict):
                        raise TypeError("JSON-RPC result must be an object")
                    gas_limits[_hex_to_int(_require_hex_field(result, "number"))] = _hex_to_int(
                        _require_hex_field(result, "gasLimit")
                    )
                    continue

                error = item.get("error")
                if isinstance(error, dict) and _coerce_error_code(error) == 429:
                    retry_blocks.append(requested_by_id[response_id])
                    continue

                raise RuntimeError(f"JSON-RPC error response: {error!r}")

            pending = retry_blocks
            if pending and attempt < self.max_retries:
                time.sleep(delay_seconds)
                delay_seconds *= 2

        raise RuntimeError(
            f"JSON-RPC throttling persisted after {self.max_retries + 1} attempts"
        )
