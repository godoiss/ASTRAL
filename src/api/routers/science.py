"""Models, inference, comparison, and persisted explanation endpoints."""

from fastapi import APIRouter, Depends, Path, Query

from src.api.dependencies import get_scientific_service
from src.api.schemas import (
    EpochMetrics,
    ExplanationDetailData,
    ExplanationIndexData,
    InferenceRequest,
    InferenceResponse,
    ModelCollection,
    ModelComparisonData,
    ModelEntry,
    PageResponse,
    Pagination,
    ScientificResponse,
)
from src.api.services import ScientificService


router = APIRouter(prefix="/api/v1", tags=["models"])


@router.get("/models", response_model=ScientificResponse[ModelCollection], operation_id="listModels")
def models(service: ScientificService = Depends(get_scientific_service)) -> ScientificResponse[ModelCollection]:
    return ScientificResponse[ModelCollection](
        data={"models": service.models()},
        limitations=["Displayed metrics are validation diagnostics, not definitive performance on unseen data."],
        provenance={"cohort": "47 identical Stage 07 validation objects"},
    )


@router.get("/models/{model_id}", response_model=ScientificResponse[ModelEntry], operation_id="getModel")
def model(
    model_id: str = Path(min_length=3, max_length=32, pattern=r"^[a-z0-9-]+$"),
    service: ScientificService = Depends(get_scientific_service),
) -> ScientificResponse[ModelEntry]:
    return ScientificResponse[ModelEntry](data=service.model(model_id), limitations=["Scores are not calibrated scientific probabilities."])


@router.get("/models/{model_id}/history", response_model=PageResponse[EpochMetrics], operation_id="getModelHistory")
def history(
    model_id: str = Path(min_length=3, max_length=32, pattern=r"^[a-z0-9-]+$"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    service: ScientificService = Depends(get_scientific_service),
) -> PageResponse[EpochMetrics]:
    items, total = service.history(model_id, offset, limit)
    return PageResponse[EpochMetrics](items=items, pagination=Pagination(offset=offset, limit=limit, returned=len(items), total=total))


@router.post("/inference/{model_id}", response_model=InferenceResponse, operation_id="runInference")
def inference(
    request: InferenceRequest,
    model_id: str = Path(min_length=3, max_length=32, pattern=r"^[a-z0-9-]+$"),
    service: ScientificService = Depends(get_scientific_service),
) -> InferenceResponse:
    return InferenceResponse.model_validate(service.inference(model_id, request.object_id))


@router.get("/metrics/comparison", response_model=ScientificResponse[ModelComparisonData], operation_id="getModelComparison")
def comparison(service: ScientificService = Depends(get_scientific_service)) -> ScientificResponse[ModelComparisonData]:
    report = service.comparison()
    return ScientificResponse[ModelComparisonData](
        data=report,
        limitations=report.get("limitations", []),
        provenance={"artifact": "Stage 07 like-for-like model comparison"},
    )


@router.get("/explanations", response_model=ScientificResponse[ExplanationIndexData], operation_id="listExplanations")
def explanations(service: ScientificService = Depends(get_scientific_service)) -> ScientificResponse[ExplanationIndexData]:
    return ScientificResponse[ExplanationIndexData](
        data=service.explanation_index(),
        limitations=["Explanations measure model sensitivity and do not establish astrophysical causality."],
        provenance={"artifact": "Stage 07 persisted explainability report"},
    )


@router.get("/explanations/{object_id}", response_model=ScientificResponse[ExplanationDetailData], operation_id="getExplanation")
def explanation(
    object_id: str = Path(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_. -]+$"),
    include_saliency: bool = False,
    saliency_points: int = Query(default=500, ge=50, le=2500),
    service: ScientificService = Depends(get_scientific_service),
) -> ScientificResponse[ExplanationDetailData]:
    return ScientificResponse[ExplanationDetailData](
        data=service.explanation(object_id, include_saliency, saliency_points),
        limitations=["Occlusion and gradients explain model sensitivity, not astrophysical causality; influential regions are not planet confirmations."],
        provenance={"artifact": "Stage 07 persisted local explanation"},
    )
