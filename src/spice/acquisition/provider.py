"""Thin provider helpers built on top of runtime configuration."""

from __future__ import annotations

import requests
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from web3.providers.rpc.utils import ExceptionRetryConfiguration

from ..core.config import ChainConfig, ProviderConfig


def build_web3(provider: ProviderConfig, chain: ChainConfig) -> Web3:
    retry_configuration = ExceptionRetryConfiguration(
        errors=[requests.RequestException, OSError, TimeoutError],
        retries=provider.retry_count,
        backoff_factor=provider.backoff_factor,
    )
    http_provider = Web3.HTTPProvider(
        provider.endpoint_for(chain.name),
        request_kwargs={"timeout": provider.timeout_seconds},
        exception_retry_configuration=retry_configuration,
    )
    web3 = Web3(http_provider)
    if chain.uses_poa_extra_data:
        web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    return web3


def redact_sensitive_text(text: str, provider: ProviderConfig) -> str:
    redacted = text
    for sensitive_value in provider.sensitive_values():
        redacted = redacted.replace(sensitive_value, "***")
    return redacted
