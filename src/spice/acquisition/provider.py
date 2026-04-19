"""Thin async provider helpers built on top of runtime configuration."""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
from web3 import AsyncWeb3
from web3._utils.batching import sort_batch_response_by_response_ids
from web3.middleware import ExtraDataToPOAMiddleware
from web3.providers.rpc import AsyncHTTPProvider
from web3.providers.rpc.utils import ExceptionRetryConfiguration, check_if_retry_on_failure
from web3.types import RPCEndpoint, RPCResponse

from ..config import ChainSpec, ProviderSpec
from ..core.errors import ConfigResolutionError


def _retry_configuration(provider: ProviderSpec) -> ExceptionRetryConfiguration:
    return ExceptionRetryConfiguration(
        errors=[aiohttp.ClientError, OSError, TimeoutError],
        retries=provider.rpc.retry_count,
        backoff_factor=provider.rpc.backoff_factor,
    )


class _ManagedAsyncSessionManager:
    def __init__(self) -> None:
        self._session_lock = asyncio.Lock()
        self._session: aiohttp.ClientSession | None = None

    async def async_cache_and_return_session(
        self,
        endpoint_uri: str,
        session: aiohttp.ClientSession | None = None,
        request_timeout: aiohttp.ClientTimeout | None = None,
    ) -> aiohttp.ClientSession:
        del endpoint_uri, request_timeout
        async with self._session_lock:
            if self._session is None or self._session.closed:
                self._session = session or aiohttp.ClientSession(
                    raise_for_status=True,
                    connector=aiohttp.TCPConnector(
                        force_close=True,
                        enable_cleanup_closed=True,
                    ),
                )
            return self._session

    async def async_make_post_request(
        self,
        endpoint_uri: str,
        data: bytes,
        **kwargs: Any,
    ) -> bytes:
        session = await self.async_cache_and_return_session(
            endpoint_uri,
            request_timeout=kwargs.get("timeout"),
        )
        async with session.post(endpoint_uri, data=data, **kwargs) as response:
            response.raise_for_status()
            return await response.read()

    async def close(self) -> None:
        async with self._session_lock:
            session = self._session
            self._session = None
        if session is not None and not session.closed:
            await session.close()


class ManagedAsyncHTTPProvider(AsyncHTTPProvider):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._request_session_manager = _ManagedAsyncSessionManager()

    def _request_kwargs_mapping(self) -> dict[str, Any]:
        return dict(self.get_request_kwargs())

    def _endpoint_uri(self) -> str:
        endpoint_uri = self.endpoint_uri
        if endpoint_uri is None:
            raise ValueError("ManagedAsyncHTTPProvider requires an endpoint URI")
        return str(endpoint_uri)

    async def _request_with_retries(
        self,
        request_data: bytes,
        *,
        method: RPCEndpoint | None = None,
    ) -> bytes:
        if (
            self.exception_retry_configuration is not None
            and (
                method is None
                or check_if_retry_on_failure(
                    method,
                    self.exception_retry_configuration.method_allowlist,
                )
            )
        ):
            for attempt in range(self.exception_retry_configuration.retries):
                try:
                    return await self._request_session_manager.async_make_post_request(
                        self._endpoint_uri(),
                        request_data,
                        **self._request_kwargs_mapping(),
                    )
                except tuple(self.exception_retry_configuration.errors):
                    if attempt < self.exception_retry_configuration.retries - 1:
                        await asyncio.sleep(
                            self.exception_retry_configuration.backoff_factor * 2**attempt
                        )
                        continue
                    raise
            return b""
        return await self._request_session_manager.async_make_post_request(
            self._endpoint_uri(),
            request_data,
            **self._request_kwargs_mapping(),
        )

    async def _make_request(self, method: RPCEndpoint, request_data: bytes) -> bytes:
        return await self._request_with_retries(request_data, method=method)

    async def make_batch_request(
        self,
        batch_requests: list[tuple[RPCEndpoint, Any]],
    ) -> list[RPCResponse] | RPCResponse:
        request_data = self.encode_batch_rpc_request(batch_requests)
        raw_response = await self._request_with_retries(request_data)
        response = self.decode_rpc_response(raw_response)
        if not isinstance(response, list):
            return response
        return sort_batch_response_by_response_ids(response)

    async def disconnect(self) -> None:
        await self._request_session_manager.close()


def build_async_web3(provider: ProviderSpec, chain: ChainSpec) -> AsyncWeb3:
    endpoint = provider.endpoint_for(chain.name)
    if not endpoint.startswith(("http://", "https://")):
        raise ConfigResolutionError(
            f"Unsupported RPC endpoint format for provider {provider.name}: {endpoint}"
        )
    web3 = AsyncWeb3(
        ManagedAsyncHTTPProvider(
            endpoint,
            request_kwargs={"timeout": provider.rpc.timeout_seconds},
            exception_retry_configuration=_retry_configuration(provider),
        )
    )
    if chain.runtime.uses_poa_extra_data:
        web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    return web3
