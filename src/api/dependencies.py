"""Dependency providers for FastAPI routers."""

from __future__ import annotations

from functools import lru_cache

from src.api.config import ApiSettings
from src.api.services import ScientificService


@lru_cache(maxsize=1)
def get_settings() -> ApiSettings:
    return ApiSettings.from_environment()


@lru_cache(maxsize=1)
def get_scientific_service() -> ScientificService:
    return ScientificService()
