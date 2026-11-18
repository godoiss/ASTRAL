"""Dataset, object, observation, and bounded light-curve endpoints."""

from typing import Literal

from fastapi import APIRouter, Depends, Path, Query

from src.api.dependencies import get_scientific_service
from src.api.schemas import (
    DatasetCollection,
    KoiMetadataResponse,
    LightCurveData,
    ObjectSummary,
    ObservationData,
    ObservatorySummaryData,
    PageResponse,
    Pagination,
    ScientificResponse,
)
from src.api.services import ScientificService


router = APIRouter(prefix="/api/v1", tags=["astronomy"])


@router.get("/datasets", response_model=ScientificResponse[DatasetCollection], operation_id="listDatasets")
def datasets(service: ScientificService = Depends(get_scientific_service)) -> ScientificResponse[DatasetCollection]:
    return ScientificResponse[DatasetCollection](
        data={"datasets": service.datasets()},
        limitations=["Frozen test identities are excluded from catalog and inference responses."],
        provenance={"source": "local persisted ASTRAL artifacts"},
    )


@router.get("/observatory/summary", response_model=ScientificResponse[ObservatorySummaryData], operation_id="getObservatorySummary")
def observatory_summary(service: ScientificService = Depends(get_scientific_service)) -> ScientificResponse[ObservatorySummaryData]:
    return ScientificResponse[ObservatorySummaryData](
        data=service.observatory_summary(),
        limitations=["The signal cohort is Quarter 3-only, period-limited, and biased toward higher catalog SNR."],
        provenance={"artifact": "Stage 07 dataset intelligence"},
    )


@router.get("/objects", response_model=PageResponse[ObjectSummary], operation_id="listObjects")
def objects(
    query: str | None = Query(default=None, min_length=1, max_length=64),
    disposition: Literal["CONFIRMED", "FALSE POSITIVE", "CANDIDATE"] | None = None,
    observed_only: bool = False,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=25, ge=1, le=100),
    service: ScientificService = Depends(get_scientific_service),
) -> PageResponse[ObjectSummary]:
    items, total = service.list_objects(query, disposition, observed_only, offset, limit)
    return PageResponse[ObjectSummary](items=items, pagination=Pagination(offset=offset, limit=limit, returned=len(items), total=total))


@router.get("/objects/{object_id}", response_model=ScientificResponse[KoiMetadataResponse], operation_id="getObject")
def object_detail(
    object_id: str = Path(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_. -]+$"),
    service: ScientificService = Depends(get_scientific_service),
) -> ScientificResponse[KoiMetadataResponse]:
    return ScientificResponse[KoiMetadataResponse](data=service.object_metadata(object_id), provenance={"source": "NASA KOI cumulative local snapshot"})


@router.get("/objects/{object_id}/observation", response_model=ScientificResponse[ObservationData], operation_id="getObservation")
def observation(
    object_id: str = Path(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_. -]+$"),
    service: ScientificService = Depends(get_scientific_service),
) -> ScientificResponse[ObservationData]:
    return ScientificResponse[ObservationData](
        data=service.observation(object_id),
        limitations=["Only previously cached observations are served; this endpoint never contacts MAST."],
        provenance={"source": "local Kepler cache"},
    )


@router.get("/objects/{object_id}/light-curve", response_model=ScientificResponse[LightCurveData], operation_id="getLightCurve")
def light_curve(
    object_id: str = Path(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_. -]+$"),
    view: Literal["raw", "processed", "phase", "local"] = "processed",
    limit: int = Query(default=1000, ge=10, le=5000),
    service: ScientificService = Depends(get_scientific_service),
) -> ScientificResponse[LightCurveData]:
    return ScientificResponse[LightCurveData](
        data=service.light_curve(object_id, view, limit),
        limitations=["Returned points are deterministically downsampled when the requested limit is below the stored record count."],
        provenance={"source": "local persisted Kepler observation"},
    )
