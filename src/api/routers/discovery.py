"""Stage 10 discovery planning, evidence, and cache-only execution endpoints."""

from fastapi import APIRouter, Depends, Path, Query, Request, status

from src.api.config import ApiSettings
from src.api.dependencies import get_discovery_jobs, get_discovery_service
from src.api.errors import ApiError
from src.api.discovery_service import DiscoveryJobManager, DiscoveryService
from src.api.schemas import (
    DiscoveryCampaignDetail,
    DiscoveryCampaignSummary,
    DiscoveryJob,
    DiscoveryPlanData,
    DiscoveryRunRequest,
    DiscoverySeriesData,
    PageResponse,
    Pagination,
    ScientificResponse,
)


router = APIRouter(prefix="/api/v1/discovery", tags=["discovery"])


@router.get("/plan", response_model=ScientificResponse[DiscoveryPlanData], operation_id="planDiscoveryCampaign")
def plan_campaign(
    request: Request,
    sample_size: int = Query(default=10, ge=1, le=25),
    service: DiscoveryService = Depends(get_discovery_service),
) -> ScientificResponse[DiscoveryPlanData]:
    plan = service.plan(sample_size)
    settings: ApiSettings = request.app.state.settings
    plan["execution_enabled"] = settings.discovery_execution_enabled
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
    settings: ApiSettings = http_request.app.state.settings
    if not settings.discovery_execution_enabled:
        raise ApiError(
            403,
            "discovery_execution_disabled",
            "Campaign execution is disabled for this deployment.",
        )
    return DiscoveryJob.model_validate(jobs.start(request.sample_size))


@router.get("/jobs/{job_id}", response_model=DiscoveryJob, operation_id="getDiscoveryJob")
def discovery_job(
    job_id: str = Path(min_length=36, max_length=36),
    jobs: DiscoveryJobManager = Depends(get_discovery_jobs),
) -> DiscoveryJob:
    return DiscoveryJob.model_validate(jobs.get(job_id))
