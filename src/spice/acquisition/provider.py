"""Thin async provider helpers built on top of runtime configuration."""

from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import unquote, urlparse

import aiohttp
from web3 import AsyncWeb3
from web3.middleware import ExtraDataToPOAMiddleware
from web3.providers import AsyncIPCProvider
from web3.providers.rpc import AsyncHTTPProvider
from web3.providers.rpc.async_rpc import (
    check_if_retry_on_failure,
    sort_batch_response_by_response_ids,
)
from web3.providers.rpc.utils import ExceptionRetryConfiguration

from ..config import ChainSpec, ProviderSpec


def _ipc_path_from_endpoint(endpoint: str) -> str | None:
    parsed = urlparse(endpoint)
    if parsed.scheme == "file":
        return unquote(parsed.path)

    candidate = Path(endpoint)
    if candidate.is_absolute():
        return str(candidate)
    return None


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
        **kwargs,
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
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        legacy_manager = self._request_session_manager
        legacy_manager.session_pool.shutdown(wait=False, cancel_futures=True)
        self._request_session_manager = _ManagedAsyncSessionManager()

    async def _make_request(self, method, request_data: bytes) -> bytes:
        if (
            self.exception_retry_configuration is not None
            and check_if_retry_on_failure(
                method,
                self.exception_retry_configuration.method_allowlist,
            )
        ):
            for attempt in range(self.exception_retry_configuration.retries):
                try:
                    return await self._request_session_manager.async_make_post_request(
                        self.endpoint_uri,
                        request_data,
                        **self.get_request_kwargs(),
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
            self.endpoint_uri,
            request_data,
            **self.get_request_kwargs(),
        )

    async def make_batch_request(self, batch_requests):
        request_data = self.encode_batch_rpc_request(batch_requests)
        raw_response = await self._request_session_manager.async_make_post_request(
            self.endpoint_uri,
            request_data,
            **self.get_request_kwargs(),
        )
        response = self.decode_rpc_response(raw_response)
        if not isinstance(response, list):
            return response
        return sort_batch_response_by_response_ids(response)

    async def disconnect(self) -> None:
        await self._request_session_manager.close()


def build_async_web3(provider: ProviderSpec, chain: ChainSpec) -> AsyncWeb3:
    endpoint = provider.endpoint_for(chain.name)

    if endpoint.startswith(("http://", "https://")):
        web3 = AsyncWeb3(
            ManagedAsyncHTTPProvider(
                endpoint,
                request_kwargs={"timeout": provider.rpc.timeout_seconds},
                exception_retry_configuration=_retry_configuration(provider),
            )
        )
    else:
        ipc_path = _ipc_path_from_endpoint(endpoint)
        if ipc_path is None:
            raise ValueError(f"Unsupported RPC endpoint format: {endpoint}")
        web3 = AsyncWeb3(
            AsyncIPCProvider(
                ipc_path,
                request_timeout=provider.rpc.timeout_seconds,
                max_connection_retries=provider.rpc.retry_count,
            )
        )

    if chain.runtime.uses_poa_extra_data:
        web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    return web3
