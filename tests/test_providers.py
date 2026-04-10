from __future__ import annotations

from spice.acquisition.rpc_providers import (
    RpcProviderName,
    resolve_acquisition_providers,
    resolve_rpc_provider,
)
from spice.core.config import ChainName
from spice.core.settings import RuntimeSettings


def test_resolve_direct_provider_uses_runtime_settings() -> None:
    settings = RuntimeSettings(
        rpc_provider=RpcProviderName.DIRECT,
        ethereum_rpc_url="https://eth.example.test",
        polygon_rpc_url="https://polygon.example.test",
        avalanche_rpc_url="https://avax.example.test",
    )

    provider = resolve_rpc_provider(chains=(ChainName.ETHEREUM,), settings=settings)

    assert provider.name is RpcProviderName.DIRECT
    assert provider.url_for(ChainName.ETHEREUM) == "https://eth.example.test"
    assert provider.reference_for(ChainName.ETHEREUM) == "$ETHEREUM_RPC_URL"


def test_resolve_acquisition_providers_supports_distinct_pull_and_enrich() -> None:
    settings = RuntimeSettings(
        rpc_provider=RpcProviderName.PUBLICNODE,
        alchemy_api_key="test-key",
    )

    providers = resolve_acquisition_providers(
        pull_provider_name=RpcProviderName.ALCHEMY,
        enrich_provider_name=RpcProviderName.PUBLICNODE,
        chains=(ChainName.ETHEREUM,),
        settings=settings,
    )

    assert providers.pull.name is RpcProviderName.ALCHEMY
    assert providers.enrich.name is RpcProviderName.PUBLICNODE
    assert "test-key" in providers.pull.url_for(ChainName.ETHEREUM)
