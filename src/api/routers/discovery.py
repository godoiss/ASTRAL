"""Persistent Discovery Engine planning, execution, evidence, and audit endpoints."""

import secrets
from typing import Literal

from fastapi import APIRouter, Depends, Path, Query, Request, status
from fastapi.responses import JSONResponse

from src.api.config import ApiSettings
from src.api.dependencies import get_discovery_jobs, get_discovery_service
from src.api.errors import ApiError
from src.api.discovery_service import DiscoveryJobManager, DiscoveryService
from src.api.schemas import (
    DiscoveryCampaignDetail,
    DiscoveryCampaignSummary,
    DiscoveryDetection,
    DiscoveryJob,
    DiscoveryEvent,
    DiscoveryPlanData,
    DiscoveryPlanRequest,
    DiscoveryProvenanceData,
    DiscoveryRunRequest,
    DiscoverySeriesData,
    PageResponse,
    Pagination,
    ScientificResponse,
)


router = APIRouter(prefix="/api/v1/discovery", tags=["discovery"])
CampaignState = Literal["planned", "queued", "running", "completed", "failed", "cancelled", "interrupted"]


def _authorize_operation(request: Request, *, require_execution: bool = True) -> ApiSettings:
    settings: ApiSettings = request.app.state.settings
    if require_execution and not settings.discovery_execution_enabled:
        raise ApiError(403, "discovery_execution_disabled", "Campaign execution is disabled for this deployment.")
    supplied = request.headers.get("X-ASTRAL-Operation-Key")
    if settings.operation_key and (not supplied or not secrets.compare_digest(supplied, settings.operation_key)):
        raise ApiError(403, "operation_not_authorized", "A valid local operation key is required.")
    client = request.client.host if request.client else ""
    if not settings.operation_key and client not in {"127.0.0.1", "::1", "testclient"}:
        raise ApiError(403, "local_operation_only", "Operational endpoints require loopback access or an operation key.")
    return settings


@router.get("/plan", response_model=ScientificResponse[DiscoveryPlanData], operation_id="planDiscoveryCampaign")
def plan_campaign(
    request: Request,
    sample_size: int = Query(default=10, ge=1, le=25),
    service: DiscoveryService = Depends(get_discovery_service),
) -> ScientificResponse[DiscoveryPlanData]:
    plan = service.plan(sample_size)
    settings: ApiSettings = request.app.state.settings
    plan["execution_enabled"] = settings.discovery_execution_enabled
    plan["downloads_enabled"] = settings.discovery_downloads_enabled
    plan["maximum_campaign_downloads"] = settings.maximum_campaign_downloads
    return ScientificResponse[DiscoveryPlanData](
        data=plan,
        limitations=["Planning uses catalog identifiers only for selection and protected-host exclusion."],
        provenance={"source": "local KOI snapshot and Stage 04 split manifest", "mission": "Kepler"},
    )


@router.get("/campaigns", response_model=PageResponse[DiscoveryCampaignSummary], operation_id="listDiscoveryCampaigns")
def campaigns(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=10, ge=1, le=50),
    service: DiscoveryService = Depends(get_discovery_service),
) -> PageResponse[DiscoveryCampaignSummary]:
    items, total = service.campaigns(offset, limit)
    return PageResponse[DiscoveryCampaignSummary](
        items=items,
        pagination=Pagination(offset=offset, limit=limit, returned=len(items), total=total),
    )


@router.post(
    "/campaigns",
    response_model=DiscoveryJob,
    status_code=status.HTTP_201_CREATED,
    operation_id="createDiscoveryCampaignPlan",
)
def create_campaign_plan(
    payload: DiscoveryPlanRequest,
    request: Request,
    jobs: DiscoveryJobManager = Depends(get_discovery_jobs),
) -> DiscoveryJob:
    settings = _authorize_operation(request)
    if payload.allow_network:
        if not settings.discovery_downloads_enabled:
            raise ApiError(403, "scientific_downloads_disabled", "MAST acquisition is not authorized for this deployment.")
        if payload.max_new_downloads < 1 or payload.max_new_downloads > settings.maximum_campaign_downloads:
            raise ApiError(422, "download_limit_invalid", "The requested download limit is outside the deployment policy.")
    elif payload.max_new_downloads != 0:
        raise ApiError(422, "download_limit_without_network", "max_new_downloads must be zero when network access is disabled.")
    return DiscoveryJob.model_validate(
        jobs.plan(
            payload.sample_size,
            allow_network=payload.allow_network,
            max_new_downloads=payload.max_new_downloads,
        )
    )


@router.get("/campaigns/{campaign_id}", response_model=ScientificResponse[DiscoveryCampaignDetail], operation_id="getDiscoveryCampaign")
def campaign_detail(
    campaign_id: str = Path(min_length=3, max_length=96, pattern=r"^[A-Za-z0-9_-]+$"),
    service: DiscoveryService = Depends(get_discovery_service),
) -> ScientificResponse[DiscoveryCampaignDetail]:
    return ScientificResponse[DiscoveryCampaignDetail](
        data=service.detail(campaign_id),
        limitations=[
            "BLS depth SNR is a screening diagnostic, not calibrated significance.",
            "An unassociated signal is not evidence of novelty or confirmation.",
        ],
        provenance={"artifact": campaign_id, "mission": "Kepler", "status": "exploratory"},
    )


@router.get(
    "/campaigns/{campaign_id}/detections",
    response_model=PageResponse[DiscoveryDetection],
    operation_id="listDiscoveryCampaignDetections",
)
def campaign_detections(
    campaign_id: str = Path(min_length=3, max_length=96, pattern=r"^[A-Za-z0-9_-]+$"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=25, ge=1, le=100),
    service: DiscoveryService = Depends(get_discovery_service),
) -> PageResponse[DiscoveryDetection]:
    targets = service.detail(campaign_id)["targets"]
    items = targets[offset : offset + limit]
    return PageResponse[DiscoveryDetection](
        items=[DiscoveryDetection.model_validate(item) for item in items],
        pagination=Pagination(offset=offset, limit=limit, returned=len(items), total=len(targets)),
    )


@router.get(
    "/campaigns/{campaign_id}/export",
    response_model=DiscoveryCampaignDetail,
    operation_id="exportDiscoveryCampaign",
)
def export_campaign(
    campaign_id: str = Path(min_length=3, max_length=96, pattern=r"^[A-Za-z0-9_-]+$"),
    service: DiscoveryService = Depends(get_discovery_service),
) -> JSONResponse:
    return JSONResponse(
        service.detail(campaign_id),
        headers={"Content-Disposition": f'attachment; filename="astral-{campaign_id}.json"'},
    )


@router.get(
    "/campaigns/{campaign_id}/targets/{kic}/series",
    response_model=ScientificResponse[DiscoverySeriesData],
    operation_id="getDiscoveryTargetSeries",
)
def target_series(
    campaign_id: str = Path(min_length=3, max_length=96, pattern=r"^[A-Za-z0-9_-]+$"),
    kic: int = Path(ge=1, le=999999999),
    limit: int = Query(default=1600, ge=100, le=5000),
    service: DiscoveryService = Depends(get_discovery_service),
) -> ScientificResponse[DiscoverySeriesData]:
    return ScientificResponse[DiscoverySeriesData](
        data=service.series(campaign_id, kic, limit),
        limitations=["Series may be deterministically downsampled for browser rendering."],
        provenance={"artifact": campaign_id, "mission": "Kepler"},
    )


@router.post(
    "/jobs",
    response_model=DiscoveryJob,
    status_code=status.HTTP_202_ACCEPTED,
    operation_id="startDiscoveryCampaign",
)
def start_campaign(
    request: DiscoveryRunRequest,
    http_request: Request,
    jobs: DiscoveryJobManager = Depends(get_discovery_jobs),
) -> DiscoveryJob:
    _authorize_operation(http_request)
    return DiscoveryJob.model_validate(jobs.start(request.sample_size))


@router.get("/jobs", response_model=PageResponse[DiscoveryJob], operation_id="listDiscoveryJobs")
def discovery_jobs(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    campaign_status: list[CampaignState] = Query(default=[]),
    jobs: DiscoveryJobManager = Depends(get_discovery_jobs),
) -> PageResponse[DiscoveryJob]:
    items, total = jobs.list(offset, limit, tuple(campaign_status))
    return PageResponse[DiscoveryJob](
        items=[DiscoveryJob.model_validate(item) for item in items],
        pagination=Pagination(offset=offset, limit=limit, returned=len(items), total=total),
    )


@router.get("/jobs/{job_id}", response_model=DiscoveryJob, operation_id="getDiscoveryJob")
def discovery_job(
    job_id: str = Path(min_length=36, max_length=36),
    jobs: DiscoveryJobManager = Depends(get_discovery_jobs),
) -> DiscoveryJob:
    return DiscoveryJob.model_validate(jobs.get(job_id))


@router.post("/jobs/{job_id}/start", response_model=DiscoveryJob, status_code=status.HTTP_202_ACCEPTED, operation_id="startPlannedDiscoveryCampaign")
def start_planned_campaign(
    request: Request,
    job_id: str = Path(min_length=36, max_length=36),
    jobs: DiscoveryJobManager = Depends(get_discovery_jobs),
) -> DiscoveryJob:
    _authorize_operation(request)
    return DiscoveryJob.model_validate(jobs.start_planned(job_id))


@router.post("/jobs/{job_id}/cancel", response_model=DiscoveryJob, operation_id="cancelDiscoveryCampaign")
def cancel_campaign(
    request: Request,
    job_id: str = Path(min_length=36, max_length=36),
    jobs: DiscoveryJobManager = Depends(get_discovery_jobs),
) -> DiscoveryJob:
    _authorize_operation(request, require_execution=False)
    return DiscoveryJob.model_validate(jobs.cancel(job_id))


@router.post("/jobs/{job_id}/resume", response_model=DiscoveryJob, status_code=status.HTTP_202_ACCEPTED, operation_id="resumeDiscoveryCampaign")
def resume_campaign(
    request: Request,
    job_id: str = Path(min_length=36, max_length=36),
    jobs: DiscoveryJobManager = Depends(get_discovery_jobs),
) -> DiscoveryJob:
    _authorize_operation(request)
    return DiscoveryJob.model_validate(jobs.resume(job_id))


@router.get("/jobs/{job_id}/events", response_model=PageResponse[DiscoveryEvent], operation_id="listDiscoveryJobEvents")
def discovery_job_events(
    job_id: str = Path(min_length=36, max_length=36),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    jobs: DiscoveryJobManager = Depends(get_discovery_jobs),
) -> PageResponse[DiscoveryEvent]:
    items, total = jobs.events(job_id, offset, limit)
    return PageResponse[DiscoveryEvent](
        items=[DiscoveryEvent.model_validate(item) for item in items],
        pagination=Pagination(offset=offset, limit=limit, returned=len(items), total=total),
    )


@router.get(
    "/jobs/{job_id}/provenance",
    response_model=ScientificResponse[DiscoveryProvenanceData],
    operation_id="getDiscoveryJobProvenance",
)
def discovery_job_provenance(
    job_id: str = Path(min_length=36, max_length=36),
    jobs: DiscoveryJobManager = Depends(get_discovery_jobs),
) -> ScientificResponse[DiscoveryProvenanceData]:
    return ScientificResponse[DiscoveryProvenanceData](
        data=DiscoveryProvenanceData.model_validate(jobs.provenance(job_id)),
        limitations=["Operational metadata does not replace the immutable scientific campaign artifact."],
        provenance={"source": "local Stage 12 SQLite operational store", "mission": "Kepler"},
    )
