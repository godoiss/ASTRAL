"""Dependency providers for FastAPI routers."""

from __future__ import annotations

from functools import lru_cache

from fastapi import Request

from src.api.config import ApiSettings
from src.api.candidate_service import CandidateIntelligenceService
from src.api.discovery_service import DiscoveryJobManager, DiscoveryService
from src.api.services import ScientificService


def get_settings(request: Request) -> ApiSettings:
    """Return the settings bound to this exact application instance."""
    return request.app.state.settings


@lru_cache(maxsize=1)
def get_scientific_service() -> ScientificService:
    return ScientificService()


@lru_cache(maxsize=1)
def get_discovery_service() -> DiscoveryService:
    return DiscoveryService()


@lru_cache(maxsize=1)
def get_discovery_jobs() -> DiscoveryJobManager:
    return DiscoveryJobManager()


@lru_cache(maxsize=1)
def get_candidate_intelligence_service() -> CandidateIntelligenceService:
    return CandidateIntelligenceService()
