"""Local-only configuration for the ASTRAL scientific API."""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, Field


API_VERSION = "0.10.0"


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
        return cls(**values)
