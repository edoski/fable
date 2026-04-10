import os
import unittest
from unittest.mock import patch

from spice_temporal.config import ChainName
from spice_temporal.rpc_providers import (
    RpcProviderName,
    redact_sensitive_text,
    resolve_acquisition_providers,
    resolve_rpc_provider,
)


class RpcProvidersTestCase(unittest.TestCase):
    def test_resolve_direct_provider_uses_chain_url_env_vars(self) -> None:
        with patch.dict(
            os.environ,
            {"ETHEREUM_RPC_URL": "https://rpc.example.test"},
            clear=True,
        ):
            provider = resolve_rpc_provider(RpcProviderName.DIRECT, chains=(ChainName.ETHEREUM,))
            self.assertEqual(provider.url_for(ChainName.ETHEREUM), "https://rpc.example.test")
            self.assertEqual(provider.reference_for(ChainName.ETHEREUM), "$ETHEREUM_RPC_URL")

    def test_resolve_alchemy_provider_uses_template_env_var(self) -> None:
        with patch.dict(
            os.environ,
            {"ALCHEMY_API_KEY": "test-key"},
            clear=True,
        ):
            provider = resolve_rpc_provider(RpcProviderName.ALCHEMY, chains=(ChainName.ETHEREUM,))
            self.assertEqual(
                provider.url_for(ChainName.ETHEREUM),
                "https://eth-mainnet.g.alchemy.com/v2/test-key",
            )
            self.assertEqual(
                provider.reference_for(ChainName.ETHEREUM),
                "https://eth-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY",
            )

    def test_resolve_publicnode_provider_uses_static_urls(self) -> None:
        provider = resolve_rpc_provider(
            RpcProviderName.PUBLICNODE,
            chains=(ChainName.ETHEREUM, ChainName.AVALANCHE),
        )
        self.assertEqual(
            provider.url_for(ChainName.ETHEREUM),
            "https://ethereum-rpc.publicnode.com",
        )
        self.assertEqual(
            provider.reference_for(ChainName.AVALANCHE),
            "https://avalanche-c-chain-rpc.publicnode.com",
        )

    def test_redact_sensitive_text_hides_provider_specific_values(self) -> None:
        with patch.dict(
            os.environ,
            {"ALCHEMY_API_KEY": "test-key"},
            clear=True,
        ):
            provider = resolve_rpc_provider(RpcProviderName.ALCHEMY, chains=(ChainName.ETHEREUM,))
            text = "https://eth-mainnet.g.alchemy.com/v2/test-key uses test-key"
            self.assertEqual(
                redact_sensitive_text(text, provider),
                "https://eth-mainnet.g.alchemy.com/v2/*** uses ***",
            )

    def test_resolve_acquisition_providers_uses_shared_default_when_no_overrides(self) -> None:
        providers = resolve_acquisition_providers(
            RpcProviderName.PUBLICNODE,
            chains=(ChainName.ETHEREUM,),
        )
        self.assertEqual(providers.pull.name, "publicnode")
        self.assertEqual(providers.enrich.name, "publicnode")

    def test_resolve_acquisition_providers_applies_stage_specific_overrides(self) -> None:
        with patch.dict(
            os.environ,
            {"ALCHEMY_API_KEY": "test-key"},
            clear=True,
        ):
            providers = resolve_acquisition_providers(
                RpcProviderName.PUBLICNODE,
                pull_provider_name=RpcProviderName.ALCHEMY,
                chains=(ChainName.ETHEREUM,),
            )
        self.assertEqual(providers.pull.name, "alchemy")
        self.assertEqual(providers.enrich.name, "publicnode")
