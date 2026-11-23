"""Local-only configuration for the ASTRAL scientific API."""

from __future__ import annotations

import os

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


API_VERSION = "0.16.0"


class ApiSettings(BaseModel):
    """Validated runtime settings with restrictive development defaults."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str = "ASTRAL Scientific API"
    version: str = API_VERSION
    api_prefix: str = "/api/v1"
    cors_origins: tuple[str, ...] = (
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    )
    allowed_hosts: tuple[str, ...] = ("localhost", "127.0.0.1", "testserver")
    default_page_size: int = Field(default=25, ge=1, le=100)
    maximum_page_size: int = Field(default=100, ge=1, le=100)
    default_curve_points: int = Field(default=1000, ge=10, le=5000)
    maximum_curve_points: int = Field(default=5000, ge=10, le=5000)
    discovery_execution_enabled: bool = False
    discovery_downloads_enabled: bool = False
    maximum_campaign_downloads: int = Field(default=5, ge=0, le=10)
    deployment_mode: Literal["local", "public_read_only"] = "local"
    candidate_reviews_enabled: bool = False
    rate_limit_per_minute: int = Field(default=60, ge=10, le=600)
    maximum_request_body_bytes: int = Field(default=1_048_576, ge=1024, le=10_485_760)
    operation_key: str | None = Field(default=None, min_length=16, max_length=256)
    frontend_url: str = Field(default="http://127.0.0.1:3000", pattern=r"^https?://")

    @classmethod
    def from_environment(cls) -> "ApiSettings":
        """Load optional comma-separated local origins without a secret file."""
        origins = os.getenv("ASTRAL_API_CORS_ORIGINS")
        hosts = os.getenv("ASTRAL_API_ALLOWED_HOSTS")
        values: dict[str, object] = {}
        if origins:
            values["cors_origins"] = tuple(
                item.strip() for item in origins.split(",") if item.strip()
            )
        if hosts:
            values["allowed_hosts"] = tuple(
                item.strip() for item in hosts.split(",") if item.strip()
            )
        execution = os.getenv("ASTRAL_DISCOVERY_EXECUTION_ENABLED", "").strip().lower()
        if execution:
            values["discovery_execution_enabled"] = execution in {"1", "true", "yes", "on"}
        downloads = os.getenv("ASTRAL_DISCOVERY_DOWNLOADS_ENABLED", "").strip().lower()
        if downloads:
            values["discovery_downloads_enabled"] = downloads in {"1", "true", "yes", "on"}
        deployment_mode = os.getenv("ASTRAL_DEPLOYMENT_MODE", "").strip().lower()
        if deployment_mode:
            values["deployment_mode"] = deployment_mode
        reviews = os.getenv("ASTRAL_CANDIDATE_REVIEWS_ENABLED", "").strip().lower()
        if reviews:
            values["candidate_reviews_enabled"] = reviews in {"1", "true", "yes", "on"}
        rate_limit = os.getenv("ASTRAL_RATE_LIMIT_PER_MINUTE", "").strip()
        if rate_limit:
            values["rate_limit_per_minute"] = int(rate_limit)
        operation_key = os.getenv("ASTRAL_OPERATION_KEY", "").strip()
        if operation_key:
            values["operation_key"] = operation_key
        frontend_url = os.getenv("ASTRAL_FRONTEND_URL", "").strip()
        if frontend_url:
            values["frontend_url"] = frontend_url
        return cls(**values)

    @property
    def public_read_only(self) -> bool:
        """Return whether every scientific mutation must remain disabled."""
        return self.deployment_mode == "public_read_only"
