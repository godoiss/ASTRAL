"""Models, inference, comparison, and persisted explanation endpoints."""

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_scientific_service
from src.api.schemas import InferenceRequest, InferenceResponse, PageResponse, Pagination, ScientificResponse
from src.api.services import ScientificService


router = APIRouter(prefix="/api/v1", tags=["models"])


@router.get("/models", response_model=ScientificResponse)
def models(service: ScientificService = Depends(get_scientific_service)) -> ScientificResponse:
    return ScientificResponse(
        data={"models": service.models()},
        limitations=["Displayed metrics are validation diagnostics, not definitive performance on unseen data."],
        provenance={"cohort": "47 identical Stage 07 validation objects"},
    )


@router.get("/models/{model_id}", response_model=ScientificResponse)
def model(model_id: str, service: ScientificService = Depends(get_scientific_service)) -> ScientificResponse:
    return ScientificResponse(data=service.model(model_id), limitations=["Scores are not calibrated scientific probabilities."])


@router.get("/models/{model_id}/history", response_model=PageResponse)
def history(
    model_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    service: ScientificService = Depends(get_scientific_service),
) -> PageResponse:
    items, total = service.history(model_id, offset, limit)
    return PageResponse(items=items, pagination=Pagination(offset=offset, limit=limit, returned=len(items), total=total))


@router.post("/inference/{model_id}", response_model=InferenceResponse)
def inference(
    model_id: str,
    request: InferenceRequest,
    service: ScientificService = Depends(get_scientific_service),
) -> InferenceResponse:
    return InferenceResponse.model_validate(service.inference(model_id, request.object_id))


@router.get("/metrics/comparison", response_model=ScientificResponse)
def comparison(service: ScientificService = Depends(get_scientific_service)) -> ScientificResponse:
    report = service.comparison()
    return ScientificResponse(
        data=report,
        limitations=report.get("limitations", []),
        provenance={"artifact": "Stage 07 like-for-like model comparison"},
    )


@router.get("/explanations", response_model=ScientificResponse)
def explanations(service: ScientificService = Depends(get_scientific_service)) -> ScientificResponse:
    return ScientificResponse(
        data=service.explanation_index(),
        limitations=["Explanations measure model sensitivity and do not establish astrophysical causality."],
        provenance={"artifact": "Stage 07 persisted explainability report"},
    )


@router.get("/explanations/{object_id}", response_model=ScientificResponse)
def explanation(
    object_id: str,
    include_saliency: bool = False,
    saliency_points: int = Query(default=500, ge=50, le=2500),
    service: ScientificService = Depends(get_scientific_service),
) -> ScientificResponse:
    return ScientificResponse(
        data=service.explanation(object_id, include_saliency, saliency_points),
        limitations=["Occlusion and gradients explain model sensitivity, not astrophysical causality; influential regions are not planet confirmations."],
        provenance={"artifact": "Stage 07 persisted local explanation"},
    )
