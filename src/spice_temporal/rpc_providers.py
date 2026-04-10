"""RPC provider instantiation and redaction."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from spice_temporal.config import ChainName

RPC_PROVIDER_ENV_VAR = "RPC_PROVIDER"


class RpcProviderName(StrEnum):
    DIRECT = "direct"
    ALCHEMY = "alchemy"
    PUBLICNODE = "publicnode"


DEFAULT_RPC_PROVIDER = RpcProviderName.DIRECT


@dataclass(frozen=True, slots=True)
class RpcProvider:
    name: str
    urls: Mapping[ChainName, str]
    references: Mapping[ChainName, str]
    sensitive_values: tuple[str, ...] = ()

    def url_for(self, chain_name: ChainName) -> str:
        return self.urls[chain_name]

    def reference_for(self, chain_name: ChainName) -> str:
        return self.references[chain_name]


@dataclass(frozen=True, slots=True)
class AcquisitionProviders:
    pull: RpcProvider
    enrich: RpcProvider


def _require_env_var(env: Mapping[str, str], env_var: str, *, provider_name: str) -> str:
    value = env.get(env_var)
    if value:
        return value
    raise RuntimeError(f"Missing {env_var} in .env for RPC provider {provider_name}.")


def _coerce_chains(chains: Iterable[ChainName] | None) -> tuple[ChainName, ...]:
    if chains is None:
        return tuple(ChainName)
    return tuple(chains)


def _build_direct_provider(env: Mapping[str, str], chains: tuple[ChainName, ...]) -> RpcProvider:
    env_vars = {
        ChainName.ETHEREUM: "ETHEREUM_RPC_URL",
        ChainName.POLYGON: "POLYGON_RPC_URL",
        ChainName.AVALANCHE: "AVALANCHE_RPC_URL",
    }
    urls = {
        chain: _require_env_var(env, env_var, provider_name="direct")
        for chain, env_var in env_vars.items()
        if chain in chains
    }
    references = {
        chain: f"${env_var}"
        for chain, env_var in env_vars.items()
        if chain in chains
    }
    sensitive_values = tuple(urls.values())
    return RpcProvider(
        name="direct",
        urls=urls,
        references=references,
        sensitive_values=sensitive_values,
    )


def _build_alchemy_provider(env: Mapping[str, str], chains: tuple[ChainName, ...]) -> RpcProvider:
    api_key = _require_env_var(env, "ALCHEMY_API_KEY", provider_name="alchemy")
    all_urls = {
        ChainName.ETHEREUM: f"https://eth-mainnet.g.alchemy.com/v2/{api_key}",
        ChainName.POLYGON: f"https://polygon-mainnet.g.alchemy.com/v2/{api_key}",
        ChainName.AVALANCHE: f"https://avax-mainnet.g.alchemy.com/v2/{api_key}",
    }
    all_references = {
        ChainName.ETHEREUM: "https://eth-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY",
        ChainName.POLYGON: "https://polygon-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY",
        ChainName.AVALANCHE: "https://avax-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY",
    }
    urls = {chain: all_urls[chain] for chain in chains}
    references = {chain: all_references[chain] for chain in chains}
    return RpcProvider(
        name="alchemy",
        urls=urls,
        references=references,
        sensitive_values=(api_key,),
    )


def _build_publicnode_provider(_: Mapping[str, str], chains: tuple[ChainName, ...]) -> RpcProvider:
    all_urls = {
        ChainName.ETHEREUM: "https://ethereum-rpc.publicnode.com",
        ChainName.POLYGON: "https://polygon-bor-rpc.publicnode.com",
        ChainName.AVALANCHE: "https://avalanche-c-chain-rpc.publicnode.com",
    }
    urls = {chain: all_urls[chain] for chain in chains}
    return RpcProvider(
        name="publicnode",
        urls=urls,
        references=urls,
    )


ProviderBuilder = Callable[[Mapping[str, str], tuple[ChainName, ...]], RpcProvider]


PROVIDER_BUILDERS: dict[RpcProviderName, ProviderBuilder] = {
    RpcProviderName.DIRECT: _build_direct_provider,
    RpcProviderName.ALCHEMY: _build_alchemy_provider,
    RpcProviderName.PUBLICNODE: _build_publicnode_provider,
}


def resolve_rpc_provider(
    provider_name: RpcProviderName | str | None = None,
    *,
    chains: Iterable[ChainName] | None = None,
    env: Mapping[str, str] | None = None,
) -> RpcProvider:
    raw_selected = provider_name or (env or os.environ).get(
        RPC_PROVIDER_ENV_VAR,
        DEFAULT_RPC_PROVIDER,
    )
    try:
        selected = RpcProviderName(raw_selected)
    except ValueError as exc:
        supported = ", ".join(sorted(provider.value for provider in PROVIDER_BUILDERS))
        raise RuntimeError(
            f"Unsupported RPC provider {raw_selected!r}. Expected one of: {supported}."
        ) from exc
    builder = PROVIDER_BUILDERS[selected]
    return builder(os.environ if env is None else env, _coerce_chains(chains))


def resolve_acquisition_providers(
    provider_name: RpcProviderName | str | None = None,
    *,
    pull_provider_name: RpcProviderName | str | None = None,
    enrich_provider_name: RpcProviderName | str | None = None,
    chains: Iterable[ChainName] | None = None,
    env: Mapping[str, str] | None = None,
) -> AcquisitionProviders:
    pull = resolve_rpc_provider(
        pull_provider_name or provider_name,
        chains=chains,
        env=env,
    )
    enrich = resolve_rpc_provider(
        enrich_provider_name or provider_name,
        chains=chains,
        env=env,
    )
    return AcquisitionProviders(pull=pull, enrich=enrich)


def redact_sensitive_text(text: str, provider: RpcProvider) -> str:
    redacted = text
    for value in provider.sensitive_values:
        redacted = redacted.replace(value, "***")
    return redacted
