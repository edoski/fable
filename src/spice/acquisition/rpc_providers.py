"""RPC provider resolution and redaction."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import Field

from ..core.config import ChainName, StrictModel

if TYPE_CHECKING:
    from ..core.settings import RuntimeSettings


class RpcProviderName(StrEnum):
    DIRECT = "direct"
    ALCHEMY = "alchemy"
    PUBLICNODE = "publicnode"


class RpcProvider(StrictModel):
    name: RpcProviderName
    urls: dict[ChainName, str]
    references: dict[ChainName, str]
    sensitive_values: tuple[str, ...] = Field(default_factory=tuple)

    def url_for(self, chain_name: ChainName) -> str:
        return self.urls[chain_name]

    def reference_for(self, chain_name: ChainName) -> str:
        return self.references[chain_name]


class AcquisitionProviders(StrictModel):
    pull: RpcProvider
    enrich: RpcProvider


def _coerce_chains(chains: Iterable[ChainName] | None) -> tuple[ChainName, ...]:
    return tuple(chains) if chains is not None else tuple(ChainName)


def _build_direct_provider(
    chains: tuple[ChainName, ...],
    *,
    ethereum_rpc_url: str | None,
    polygon_rpc_url: str | None,
    avalanche_rpc_url: str | None,
) -> RpcProvider:
    url_map = {
        ChainName.ETHEREUM: ethereum_rpc_url,
        ChainName.POLYGON: polygon_rpc_url,
        ChainName.AVALANCHE: avalanche_rpc_url,
    }
    env_names = {
        ChainName.ETHEREUM: "$ETHEREUM_RPC_URL",
        ChainName.POLYGON: "$POLYGON_RPC_URL",
        ChainName.AVALANCHE: "$AVALANCHE_RPC_URL",
    }
    missing = [chain.value for chain in chains if not url_map[chain]]
    if missing:
        raise RuntimeError("Missing direct RPC URLs for: " + ", ".join(missing))
    urls = {chain: str(url_map[chain]) for chain in chains}
    references = {chain: env_names[chain] for chain in chains}
    return RpcProvider(
        name=RpcProviderName.DIRECT,
        urls=urls,
        references=references,
        sensitive_values=tuple(urls.values()),
    )


def _build_alchemy_provider(
    chains: tuple[ChainName, ...],
    *,
    alchemy_api_key: str | None,
) -> RpcProvider:
    if not alchemy_api_key:
        raise RuntimeError("Missing ALCHEMY_API_KEY for RPC provider alchemy")
    all_urls = {
        ChainName.ETHEREUM: f"https://eth-mainnet.g.alchemy.com/v2/{alchemy_api_key}",
        ChainName.POLYGON: f"https://polygon-mainnet.g.alchemy.com/v2/{alchemy_api_key}",
        ChainName.AVALANCHE: f"https://avax-mainnet.g.alchemy.com/v2/{alchemy_api_key}",
    }
    all_refs = {
        ChainName.ETHEREUM: "https://eth-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY",
        ChainName.POLYGON: "https://polygon-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY",
        ChainName.AVALANCHE: "https://avax-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY",
    }
    return RpcProvider(
        name=RpcProviderName.ALCHEMY,
        urls={chain: all_urls[chain] for chain in chains},
        references={chain: all_refs[chain] for chain in chains},
        sensitive_values=(alchemy_api_key,),
    )


def _build_publicnode_provider(chains: tuple[ChainName, ...]) -> RpcProvider:
    all_urls = {
        ChainName.ETHEREUM: "https://ethereum-rpc.publicnode.com",
        ChainName.POLYGON: "https://polygon-bor-rpc.publicnode.com",
        ChainName.AVALANCHE: "https://avalanche-c-chain-rpc.publicnode.com",
    }
    urls = {chain: all_urls[chain] for chain in chains}
    return RpcProvider(
        name=RpcProviderName.PUBLICNODE,
        urls=urls,
        references=urls,
    )


def resolve_rpc_provider(
    provider_name: RpcProviderName | str | None = None,
    *,
    chains: Iterable[ChainName] | None = None,
    settings: RuntimeSettings | None = None,
) -> RpcProvider:
    from ..core.settings import load_settings

    resolved_settings = settings or load_settings()
    selected = RpcProviderName(provider_name or resolved_settings.rpc_provider)
    chain_tuple = _coerce_chains(chains)
    if selected is RpcProviderName.DIRECT:
        return _build_direct_provider(
            chain_tuple,
            ethereum_rpc_url=resolved_settings.ethereum_rpc_url,
            polygon_rpc_url=resolved_settings.polygon_rpc_url,
            avalanche_rpc_url=resolved_settings.avalanche_rpc_url,
        )
    if selected is RpcProviderName.ALCHEMY:
        return _build_alchemy_provider(
            chain_tuple,
            alchemy_api_key=resolved_settings.alchemy_api_key,
        )
    return _build_publicnode_provider(chain_tuple)


def resolve_acquisition_providers(
    provider_name: RpcProviderName | str | None = None,
    *,
    pull_provider_name: RpcProviderName | str | None = None,
    enrich_provider_name: RpcProviderName | str | None = None,
    chains: Iterable[ChainName] | None = None,
    settings: RuntimeSettings | None = None,
) -> AcquisitionProviders:
    return AcquisitionProviders(
        pull=resolve_rpc_provider(
            pull_provider_name or provider_name,
            chains=chains,
            settings=settings,
        ),
        enrich=resolve_rpc_provider(
            enrich_provider_name or provider_name,
            chains=chains,
            settings=settings,
        ),
    )


def redact_sensitive_text(text: str, provider: RpcProvider) -> str:
    redacted = text
    for sensitive_value in provider.sensitive_values:
        redacted = redacted.replace(sensitive_value, "***")
    return redacted
