"""Application settings loaded from environment / .env file."""

from __future__ import annotations

from functools import lru_cache
from zoneinfo import ZoneInfo

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config.parameters import SiteParameters, TariffParameters, TESParameters


class Settings(BaseSettings):
    """Global settings. Env prefix ``TES_``; nested delimiter ``__``.

    Example: ``TES_TES__CAPACITY_KWH=20`` overrides ``settings.tes.capacity_kwh``.
    """

    model_config = SettingsConfigDict(
        env_prefix="TES_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Cloud Runtime Environment
    app_env: str = Field(default="development", validation_alias=AliasChoices("APP_ENV", "TES_APP_ENV", "app_env"))
    port: int = Field(default=8000, validation_alias=AliasChoices("PORT", "TES_PORT", "port"))
    database_url: str = Field(
        default="sqlite:///./data/tes.db",
        validation_alias=AliasChoices("DATABASE_URL", "TES_DATABASE_URL", "database_url"),
    )
    run_in_process_worker: bool = Field(
        default=False,
        validation_alias=AliasChoices("RUN_IN_PROCESS_WORKER", "TES_RUN_IN_PROCESS_WORKER", "run_in_process_worker"),
    )

    # Market Providers
    market_provider: str = Field(
        default="LITGRID",
        validation_alias=AliasChoices("MARKET_PROVIDER", "TES_MARKET_PROVIDER", "market_provider"),
    )
    litgrid_api_url: str = Field(
        default="https://openapi.litgrid.eu/v1/kategorijos/elektros-energijos-kainos/801",
        validation_alias=AliasChoices("LITGRID_API_URL", "TES_LITGRID_API_URL", "litgrid_api_url"),
    )
    elering_api_url: str = Field(
        default="https://dashboard.elering.ee/api/nps/price",
        validation_alias=AliasChoices("ELERING_API_URL", "TES_ELERING_API_URL", "elering_api_url"),
    )

    # Security & Authentication
    app_secret_key: SecretStr = Field(
        default=SecretStr("virtual-tes-insecure-secret-key-change-in-production"),
        validation_alias=AliasChoices("APP_SECRET_KEY", "TES_APP_SECRET_KEY", "app_secret_key"),
    )
    auth_required: bool | None = Field(
        default=None,
        validation_alias=AliasChoices("AUTH_REQUIRED", "TES_AUTH_REQUIRED", "auth_required"),
    )
    dashboard_username: str = Field(
        default="admin",
        validation_alias=AliasChoices("DASHBOARD_USERNAME", "TES_DASHBOARD_USERNAME", "dashboard_username"),
    )
    dashboard_password_hash: str = Field(
        default="",
        validation_alias=AliasChoices("DASHBOARD_PASSWORD_HASH", "TES_DASHBOARD_PASSWORD_HASH", "dashboard_password_hash"),
    )
    dashboard_password: str = Field(
        default="admin123",
        validation_alias=AliasChoices("DASHBOARD_PASSWORD", "TES_DASHBOARD_PASSWORD", "dashboard_password"),
    )
    viewer_username: str = Field(
        default="viewer",
        validation_alias=AliasChoices("VIEWER_USERNAME", "TES_VIEWER_USERNAME", "viewer_username"),
    )
    viewer_password_hash: str = Field(
        default="",
        validation_alias=AliasChoices("VIEWER_PASSWORD_HASH", "TES_VIEWER_PASSWORD_HASH", "viewer_password_hash"),
    )
    viewer_password: str = Field(
        default="viewer123",
        validation_alias=AliasChoices("VIEWER_PASSWORD", "TES_VIEWER_PASSWORD", "viewer_password"),
    )

    timezone: str = "Europe/Vilnius"
    bidding_zone: str = "LT"
    resolution_minutes: int = Field(15, gt=0)
    log_level: str = "INFO"
    log_json: bool = False

    entsoe_api_token: SecretStr | None = None

    tes: TESParameters = TESParameters()
    site: SiteParameters = SiteParameters()
    tariff: TariffParameters = TariffParameters()

    @property
    def is_auth_enforced(self) -> bool:
        if self.auth_required is not None:
            return self.auth_required
        return self.app_env.lower() == "production"

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()
