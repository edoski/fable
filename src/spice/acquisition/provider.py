"""Thin provider helpers built on top of runtime configuration."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from web3.providers.rpc.utils import ExceptionRetryConfiguration

from ..core.config import ChainConfig, ProviderConfig


def _ipc_path_from_endpoint(endpoint: str) -> str | None:
    parsed = urlparse(endpoint)
    if parsed.scheme == "file":
        return unquote(parsed.path)

    candidate = Path(endpoint)
    if candidate.is_absolute():
        return str(candidate)
    return None


def build_web3(provider: ProviderConfig, chain: ChainConfig) -> Web3:
    endpoint = provider.endpoint_for(chain.name)
    retry_configuration = ExceptionRetryConfiguration(
        errors=[requests.RequestException, OSError, TimeoutError],
        retries=provider.retry_count,
        backoff_factor=provider.backoff_factor,
    )

    if endpoint.startswith(("http://", "https://")):
        web3 = Web3(
            Web3.HTTPProvider(
                endpoint,
                request_kwargs={"timeout": provider.timeout_seconds},
                exception_retry_configuration=retry_configuration,
            )
        )
    else:
        ipc_path = _ipc_path_from_endpoint(endpoint)
        if ipc_path is None:
            raise ValueError(f"Unsupported RPC endpoint format: {endpoint}")
        web3 = Web3(Web3.IPCProvider(ipc_path, timeout=int(provider.timeout_seconds)))

    if chain.uses_poa_extra_data:
        web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    return web3
