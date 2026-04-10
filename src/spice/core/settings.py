"""Environment-backed runtime settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from ..acquisition.rpc_providers import RpcProviderName
from .constants import PROJECT_ROOT


class RuntimeSettings(BaseSettings):
    """Runtime settings loaded from the project .env file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    rpc_provider: RpcProviderName = RpcProviderName.DIRECT
    ethereum_rpc_url: str | None = None
    polygon_rpc_url: str | None = None
    avalanche_rpc_url: str | None = None
    alchemy_api_key: str | None = None


@lru_cache(maxsize=1)
def load_settings() -> RuntimeSettings:
    return RuntimeSettings()
