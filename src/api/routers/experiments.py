"""Read-only experiment registry and isolated TESS pilot endpoints."""

from typing import Literal

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_scientific_service
from src.api.schemas import PageResponse, Pagination, ScientificResponse
from src.api.services import ScientificService


router = APIRouter(prefix="/api/v1", tags=["research"])


@router.get("/experiments", response_model=PageResponse)
def experiments(
    stage: Literal["stage06", "stage07"] = "stage07",
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=25, ge=1, le=100),
    service: ScientificService = Depends(get_scientific_service),
) -> PageResponse:
    items, total = service.list_experiments(stage, offset, limit)
    return PageResponse(items=items, pagination=Pagination(offset=offset, limit=limit, returned=len(items), total=total))


@router.get("/experiments/{experiment_id}", response_model=ScientificResponse)
def experiment(experiment_id: str, service: ScientificService = Depends(get_scientific_service)) -> ScientificResponse:
    return ScientificResponse(
        data=service.experiment(experiment_id),
        limitations=["Validation results are development diagnostics and not final unseen-test estimates."],
        provenance={"source": "local immutable experiment artifact"},
    )


@router.get("/tess/experimental", response_model=ScientificResponse)
def tess(service: ScientificService = Depends(get_scientific_service)) -> ScientificResponse:
    return ScientificResponse(
        data=service.tess(),
        limitations=["This single TESS product is a compatibility study and is not mixed with Kepler training."],
        provenance={"mission": "TESS", "status": "experimental and isolated"},
    )
