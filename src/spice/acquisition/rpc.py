"""Minimal JSON-RPC client for enrichment hydration."""

from __future__ import annotations

import time

import httpx


def _hex_to_int(value: str) -> int:
    return int(value, 16)


class JsonRpcClient:
    def __init__(
        self,
        url: str,
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 5,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._client = httpx.Client(timeout=timeout_seconds)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> JsonRpcClient:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _post(self, payload: list[dict[str, object]]) -> list[dict[str, object]]:
        response = self._client.post(self.url, json=payload)
        response.raise_for_status()
        body = response.json()
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
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429 and attempt < self.max_retries:
                    time.sleep(delay_seconds)
                    delay_seconds *= 2
                    continue
                raise

            for item in response_items:
                if "id" not in item or not isinstance(item["id"], int):
                    raise ValueError("JSON-RPC batch response item is missing a valid integer id")
                response_id = item["id"]
                if response_id not in requested_by_id:
                    raise ValueError(f"Unexpected JSON-RPC response id: {response_id}")
                if "result" in item:
                    result = item["result"]
                    if not isinstance(result, dict):
                        raise TypeError("JSON-RPC result must be an object")
                    block_number = _hex_to_int(str(result["number"]))
                    gas_limit = _hex_to_int(str(result["gasLimit"]))
                    gas_limits[block_number] = gas_limit
                    continue
                error = item.get("error")
                if isinstance(error, dict) and error.get("code") == 429:
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
