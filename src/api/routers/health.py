"""Health and immutable API identity endpoints."""

from fastapi import APIRouter, Depends

from src.api.config import ApiSettings
from src.api.dependencies import get_scientific_service, get_settings
from src.api.schemas import ApiInfoResponse, HealthResponse
from src.api.services import ScientificService


router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Check local API readiness", operation_id="getHealth")
def health(
    service: ScientificService = Depends(get_scientific_service),
    settings: ApiSettings = Depends(get_settings),
) -> HealthResponse:
    availability = service.availability()
    ready = all(availability["models"].values()) and all(availability["artifacts"].values())
    return HealthResponse(
        status="ok" if ready else "degraded",
        service=settings.title,
        version=settings.version,
        stage=16,
        local_only=not settings.public_read_only,
        **availability,
    )


@router.get("/api/v1/info", response_model=ApiInfoResponse, summary="Describe API version and scientific guardrails", operation_id="getApiInfo")
def info(settings: ApiSettings = Depends(get_settings)) -> ApiInfoResponse:
    return ApiInfoResponse(
        name=settings.title,
        version=settings.version,
        stage=16,
        api_version="v1",
        documentation={"openapi": "/openapi.json", "swagger": "/docs", "redoc": "/redoc"},
        scientific_guardrails={
            "frozen_test_accessed": False,
            "training_enabled": False,
            "downloads_enabled": settings.discovery_downloads_enabled,
            "automatic_ensemble_enabled": False,
            "kepler_tess_mixed": False,
            "campaign_execution_enabled": settings.discovery_execution_enabled and not settings.public_read_only,
            "candidate_reviews_enabled": settings.candidate_reviews_enabled and not settings.public_read_only,
            "public_read_only": settings.public_read_only,
        },
    )
