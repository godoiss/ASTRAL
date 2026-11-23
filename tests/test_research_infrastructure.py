"""Persistent Stage 12 research infrastructure and API tests."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

import src.discovery.campaign as campaign_module
from src.api.app import create_app
from src.api.config import ApiSettings
from src.api.dependencies import get_discovery_jobs
from src.api.discovery_service import DiscoveryJobManager
from src.discovery.config import DiscoveryConfig
from src.research.campaign_store import CampaignStore


def _create_plan(store: CampaignStore, campaign_id: str = "stage12-store-test") -> dict:
    config = DiscoveryConfig()
    return store.create_plan(
        campaign_id=campaign_id,
        mission="Kepler",
        selection_strategy=config.campaign.strategy,
        total=2,
        allow_network=False,
        max_new_downloads=0,
        campaign_config=config.to_dict()["campaign"],
        detector_config=config.to_dict()["detection"],
        provenance={"frozen_test_records_accessed": False},
        resources={"worker_concurrency_limit": 1},
    )


def _wait(manager: DiscoveryJobManager, job_id: str, states: set[str]) -> dict:
    for _ in range(200):
        job = manager.get(job_id)
        if job["status"] in states:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job did not enter one of {states}")


def test_sqlite_store_recovers_orphaned_job_as_interrupted(tmp_path: Path) -> None:
    database = tmp_path / "research.db"
    store = CampaignStore(database)
    job = _create_plan(store)
    store.transition(job["job_id"], "queued", stage="queued", message="queued")
    store.transition(job["job_id"], "running", stage="processing", message="running")
    store.progress(
        job["job_id"], stage="processing", current=1, total=2, message="KIC 10811496", kic=10811496, elapsed_seconds=1.2
    )

    restarted = CampaignStore(database)
    recovered = restarted.get(job["job_id"])
    events, total = restarted.events(job["job_id"], offset=0, limit=100)

    assert recovered["status"] == "interrupted"
    assert recovered["current_kic"] == 10811496
    assert total == len(events)
    assert events[-1]["event_type"] == "worker_interrupted"


def test_manager_cancels_at_safe_boundary_and_resumes(tmp_path: Path) -> None:
    calls = 0

    def controlled_runner(config, *, campaign_id, progress_callback, cancellation_callback, resume):
        nonlocal calls
        calls += 1
        progress_callback("processing", 1, config.campaign.sample_size, "KIC 10811496")
        if not resume:
            for _ in range(200):
                if cancellation_callback():
                    return {"campaign_id": campaign_id, "status": "cancelled", "summary": {"investigated_stars": 0}}
                time.sleep(0.005)
            raise AssertionError("cancellation was not observed")
        return {"campaign_id": campaign_id, "status": "completed", "summary": {"investigated_stars": 2}}

    manager = DiscoveryJobManager(
        store=CampaignStore(tmp_path / "operations.db"),
        runner=controlled_runner,
    )
    planned = manager.plan(2)
    manager.start_planned(planned["job_id"])
    _wait(manager, planned["job_id"], {"running"})
    manager.cancel(planned["job_id"])
    cancelled = _wait(manager, planned["job_id"], {"cancelled"})
    assert cancelled["cancel_requested"] is True

    manager.resume(planned["job_id"])
    completed = _wait(manager, planned["job_id"], {"completed"})
    assert completed["cancel_requested"] is False
    assert completed["result_summary"]["investigated_stars"] == 2
    assert calls == 2


def test_research_api_persists_plan_events_provenance_and_results(tmp_path: Path) -> None:
    def fast_runner(config, *, campaign_id, progress_callback, cancellation_callback, resume):
        assert not config.campaign.allow_network
        assert not cancellation_callback()
        progress_callback("processing", 1, config.campaign.sample_size, "KIC 10811496")
        return {"campaign_id": campaign_id, "status": "completed", "summary": {"investigated_stars": 1}}

    manager = DiscoveryJobManager(store=CampaignStore(tmp_path / "api.db"), runner=fast_runner)
    app = create_app(ApiSettings(discovery_execution_enabled=True))
    app.dependency_overrides[get_discovery_jobs] = lambda: manager

    with TestClient(app) as client:
        planned = client.post(
            "/api/v1/discovery/campaigns",
            json={"mission": "Kepler", "sample_size": 1, "allow_network": False, "max_new_downloads": 0},
        )
        assert planned.status_code == 201
        job_id = planned.json()["job_id"]
        started = client.post(f"/api/v1/discovery/jobs/{job_id}/start")
        assert started.status_code == 202
        _wait(manager, job_id, {"completed"})

        page = client.get("/api/v1/discovery/jobs")
        events = client.get(f"/api/v1/discovery/jobs/{job_id}/events")
        provenance = client.get(f"/api/v1/discovery/jobs/{job_id}/provenance")
        denied_download = client.post(
            "/api/v1/discovery/campaigns",
            json={"mission": "Kepler", "sample_size": 1, "allow_network": True, "max_new_downloads": 1},
        )

    assert page.status_code == events.status_code == provenance.status_code == 200
    assert page.json()["pagination"]["total"] == 1
    assert events.json()["pagination"]["total"] >= 4
    assert provenance.json()["data"]["scientific_guardrails"]["frozen_test_records_accessed"] is False
    assert denied_download.status_code == 403


def test_real_cached_campaign_can_cancel_then_resume_without_recomputing_completed_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(campaign_module, "CAMPAIGN_ROOT", tmp_path / "campaigns")
    monkeypatch.setattr(campaign_module, "LATEST_CAMPAIGN", tmp_path / "latest.json")
    monkeypatch.setattr(campaign_module, "_relative", lambda path: str(path))
    base = DiscoveryConfig.load(Path("configs/discovery_engine.json"))
    config = replace(
        base,
        campaign=replace(base.campaign, sample_size=1, allow_network=False, max_new_downloads=0),
    )

    cancelled = campaign_module.run_campaign(
        config,
        campaign_id="stage12-real-cache-resume",
        cancellation_callback=lambda: True,
    )
    resumed = campaign_module.run_campaign(
        config,
        campaign_id="stage12-real-cache-resume",
        cancellation_callback=lambda: False,
        resume=True,
    )

    assert cancelled["status"] == "cancelled"
    assert resumed["status"] == "completed"
    assert resumed["summary"]["investigated_stars"] == 1
    assert resumed["summary"]["new_downloads_started"] == 0
    assert resumed["summary"]["frozen_test_records_accessed"] is False
    assert resumed["resume_count"] == 1
