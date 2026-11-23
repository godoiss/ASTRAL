"""Stage 11 Candidate Intelligence API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse

from src.api.candidate_service import CandidateIntelligenceService
from src.api.dependencies import get_candidate_intelligence_service
from src.api.dependencies import get_settings
from src.api.config import ApiSettings
from src.api.security import authorize_mutation
from src.api.schemas import CandidateDossier, CandidateIntelligenceIndex, HumanReview, HumanReviewRequest, ScientificResponse


router = APIRouter(prefix="/api/v1/candidate-intelligence", tags=["candidate-intelligence"])

REPORT_ERROR_RESPONSES = {
    code: {
        "description": description,
        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}},
    }
    for code, description in {
        400: "Malformed or disallowed request",
        403: "Operation disabled for this deployment",
        404: "Resource is unavailable or inaccessible",
        409: "Ambiguous scientific identifier",
        422: "Request validation failed",
        500: "Unexpected local API error",
        503: "Persisted scientific artifact is unavailable",
    }.items()
}


@router.get("", response_model=ScientificResponse[CandidateIntelligenceIndex], operation_id="listCandidateIntelligence")
def candidate_index(
    service: CandidateIntelligenceService = Depends(get_candidate_intelligence_service),
) -> ScientificResponse[CandidateIntelligenceIndex]:
    return ScientificResponse[CandidateIntelligenceIndex](
        data=service.list(),
        limitations=["Review priority is not a planet probability or confirmation."],
        provenance={
            "source": "persisted Stage 11 candidate-intelligence artifacts",
            "mission": "Kepler",
            "status": "exploratory scientific review",
        },
    )


@router.get("/{kic}", response_model=ScientificResponse[CandidateDossier], operation_id="getCandidateDossier")
def candidate_dossier(
    kic: int = Path(ge=1, le=999_999_999),
    service: CandidateIntelligenceService = Depends(get_candidate_intelligence_service),
) -> ScientificResponse[CandidateDossier]:
    return ScientificResponse[CandidateDossier](
        data=service.detail(kic),
        limitations=["The dossier records screening evidence and catalog cross-checks, not a discovery claim."],
        provenance={
            "artifact": f"Stage 11 dossier KIC {kic}",
            "mission": "Kepler",
            "status": "exploratory scientific review",
        },
    )


@router.get("/{kic}/export", response_model=CandidateDossier, operation_id="exportCandidateDossier")
def export_candidate_dossier(
    kic: int = Path(ge=1, le=999_999_999),
    service: CandidateIntelligenceService = Depends(get_candidate_intelligence_service),
) -> JSONResponse:
    return JSONResponse(
        service.detail(kic),
        headers={"Content-Disposition": f'attachment; filename="astral-kic-{kic}-dossier.json"'},
    )


@router.get(
    "/{kic}/report",
    response_class=PlainTextResponse,
    operation_id="downloadCandidateReport",
    responses=REPORT_ERROR_RESPONSES,
)
def candidate_report(
    kic: int = Path(ge=1, le=999_999_999),
    service: CandidateIntelligenceService = Depends(get_candidate_intelligence_service),
) -> PlainTextResponse:
    return PlainTextResponse(
        service.markdown(kic),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="astral-kic-{kic}-report.md"'},
    )


@router.get("/{kic}/reviews", response_model=list[HumanReview], operation_id="listCandidateReviews")
def candidate_reviews(
    kic: int = Path(ge=1, le=999_999_999),
    service: CandidateIntelligenceService = Depends(get_candidate_intelligence_service),
) -> list[HumanReview]:
    return [HumanReview.model_validate(item) for item in service.reviews(kic)]


@router.post("/{kic}/reviews", response_model=HumanReview, status_code=status.HTTP_201_CREATED, operation_id="recordCandidateReview")
def record_candidate_review(
    request: HumanReviewRequest,
    http_request: Request,
    kic: int = Path(ge=1, le=999_999_999),
    service: CandidateIntelligenceService = Depends(get_candidate_intelligence_service),
    settings: ApiSettings = Depends(get_settings),
) -> HumanReview:
    authorize_mutation(
        http_request,
        settings,
        feature_enabled=settings.candidate_reviews_enabled,
        feature_name="Candidate review recording",
    )
    return HumanReview.model_validate(service.add_review(kic, request.reviewer, request.decision, request.notes))
