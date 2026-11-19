"""Dependency providers for FastAPI routers."""

from __future__ import annotations

from functools import lru_cache

from src.api.config import ApiSettings
from src.api.candidate_service import CandidateIntelligenceService
from src.api.discovery_service import DiscoveryJobManager, DiscoveryService
from src.api.services import ScientificService


@lru_cache(maxsize=1)
def get_settings() -> ApiSettings:
    return ApiSettings.from_environment()


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
