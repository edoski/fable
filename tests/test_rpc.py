import unittest
import urllib.error
from unittest.mock import patch

from spice_temporal._rpc import JsonRpcClient


class JsonRpcClientTestCase(unittest.TestCase):
    def test_get_block_gas_limits_retries_only_throttled_items(self) -> None:
        client = JsonRpcClient(
            "https://rpc.example.test",
            max_retries=2,
            retry_backoff_seconds=0.0,
        )
        responses = [
            [
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"number": hex(10), "gasLimit": hex(100)},
                },
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "error": {"code": 429, "message": "rate limited"},
                },
            ],
            [
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"number": hex(11), "gasLimit": hex(101)},
                }
            ],
        ]

        with patch.object(JsonRpcClient, "_post", side_effect=responses) as post:
            result = client.get_block_gas_limits([10, 11])

        self.assertEqual(result, {10: 100, 11: 101})
        first_payload = post.call_args_list[0].args[0]
        second_payload = post.call_args_list[1].args[0]
        self.assertEqual([item["params"][0] for item in first_payload], [hex(10), hex(11)])
        self.assertEqual([item["params"][0] for item in second_payload], [hex(11)])

    def test_get_block_gas_limits_raises_after_exhausting_retries(self) -> None:
        client = JsonRpcClient(
            "https://rpc.example.test",
            max_retries=1,
            retry_backoff_seconds=0.0,
        )
        responses = [
            [{"jsonrpc": "2.0", "id": 1, "error": {"code": 429, "message": "slow down"}}],
            [{"jsonrpc": "2.0", "id": 1, "error": {"code": 429, "message": "slow down"}}],
        ]

        with patch.object(JsonRpcClient, "_post", side_effect=responses):
            with self.assertRaisesRegex(RuntimeError, "throttling persisted"):
                client.get_block_gas_limits([10])

    def test_get_block_gas_limits_raises_for_non_retryable_errors(self) -> None:
        client = JsonRpcClient("https://rpc.example.test", retry_backoff_seconds=0.0)

        with patch.object(
            JsonRpcClient,
            "_post",
            return_value=[
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "error": {"code": -32000, "message": "bad request"},
                }
            ],
        ):
            with self.assertRaisesRegex(RuntimeError, "JSON-RPC error response"):
                client.get_block_gas_limits([10])

    def test_get_block_gas_limits_retries_after_http_429(self) -> None:
        client = JsonRpcClient(
            "https://rpc.example.test",
            max_retries=2,
            retry_backoff_seconds=0.0,
        )
        responses = [
            urllib.error.HTTPError(
                "https://rpc.example.test",
                429,
                "Too Many Requests",
                hdrs=None,
                fp=None,
            ),
            [
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"number": hex(10), "gasLimit": hex(100)},
                }
            ],
        ]

        with patch.object(JsonRpcClient, "_post", side_effect=responses):
            result = client.get_block_gas_limits([10])

        self.assertEqual(result, {10: 100})


if __name__ == "__main__":
    unittest.main()
