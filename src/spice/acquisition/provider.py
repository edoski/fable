"""Thin async provider helpers built on top of runtime configuration."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

import aiohttp
from web3 import AsyncWeb3
from web3.middleware import ExtraDataToPOAMiddleware
from web3.providers import AsyncIPCProvider
from web3.providers.rpc import AsyncHTTPProvider
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


def build_async_web3(provider: ProviderSpec, chain: ChainSpec) -> AsyncWeb3:
    endpoint = provider.endpoint_for(chain.name)

    if endpoint.startswith(("http://", "https://")):
        web3 = AsyncWeb3(
            AsyncHTTPProvider(
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
