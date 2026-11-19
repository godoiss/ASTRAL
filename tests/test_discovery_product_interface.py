"""Product-interface and scientific API integration tests for Stage 10."""

from __future__ import annotations

import time
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.config import ApiSettings
from src.api.dependencies import get_discovery_jobs, get_discovery_service


def test_product_root_serves_atlas_interface() -> None:
    with TestClient(create_app()) as client:
        root = client.get("/", follow_redirects=False)
        interface = client.get("/app/")

    assert root.status_code in {302, 307}
    assert root.headers["location"] == "/app/"
    assert interface.status_code == 200
    assert "Discovery Engine operational" in interface.text
    assert "Local prototype" not in interface.text
    assert interface.headers["x-frame-options"] == "DENY"
    assert interface.headers["x-content-type-options"] == "nosniff"
    assert "connect-src 'self'" in interface.headers["content-security-policy"]


def test_discovery_plan_is_unlabeled_cached_and_protected() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/discovery/plan", params={"sample_size": 10})

    assert response.status_code == 200
    plan = response.json()["data"]
    assert plan["sample_size"] == 10
    assert plan["cached_targets"] == 10
    assert plan["network_enabled"] is False
    assert plan["execution_enabled"] is False
    assert plan["protected_host_overlap"] == []
    assert {target["label"] for target in plan["targets"]} == {-1}


def test_persisted_campaign_and_bounded_series_are_real() -> None:
    with TestClient(create_app()) as client:
        page = client.get("/api/v1/discovery/campaigns", params={"limit": 1})
        campaign_id = page.json()["items"][0]["campaign_id"]
        detail = client.get(f"/api/v1/discovery/campaigns/{campaign_id}")
        first = detail.json()["data"]["targets"][0]
        series = client.get(
            f"/api/v1/discovery/campaigns/{campaign_id}/targets/{first['kic']}/series",
            params={"limit": 500},
        )

    assert page.status_code == detail.status_code == series.status_code == 200
    assert detail.json()["data"]["campaign"]["confirmed_discoveries"] == 0
    assert first["interpretation"] in {
        "catalog_associated_signal",
        "unassociated_inconclusive_signal",
        "no_signal_above_screening_threshold",
    }
    arrays = series.json()["data"]
    assert 100 <= arrays["records_returned"] <= 500
    assert len(arrays["time_bkjd"]) == len(arrays["detrended_flux"])
    assert len(arrays["periods"]) == len(arrays["powers"])


def test_browser_job_is_cache_only_and_reports_real_progress_contract() -> None:
    def fake_campaign(config, *, campaign_id, progress_callback):
        assert config.campaign.allow_network is False
        assert config.campaign.max_new_downloads == 0
        progress_callback("processing", 1, config.campaign.sample_size, "KIC 10811496")
        return {"campaign_id": campaign_id}

    get_discovery_jobs.cache_clear()
    with patch("src.api.discovery_service.run_campaign", side_effect=fake_campaign):
        with TestClient(create_app(ApiSettings(discovery_execution_enabled=True))) as client:
            started = client.post("/api/v1/discovery/jobs", json={"sample_size": 5})
            assert started.status_code == 202
            job_id = started.json()["job_id"]
            for _ in range(50):
                job = client.get(f"/api/v1/discovery/jobs/{job_id}")
                if job.json()["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.01)

    get_discovery_jobs.cache_clear()
    assert job.status_code == 200
    assert job.json()["status"] == "completed"
    assert job.json()["current"] == 5
    assert job.json()["campaign_id"].startswith("web-")


def test_discovery_errors_are_bounded_and_structured() -> None:
    with TestClient(create_app()) as client:
        invalid_sample = client.get("/api/v1/discovery/plan", params={"sample_size": 26})
        missing = client.get("/api/v1/discovery/campaigns/not-a-real-campaign")
        traversal = client.get("/api/v1/discovery/campaigns/%2E%2E%5Csecret")

    assert invalid_sample.status_code == 422
    assert missing.status_code == 404
    assert traversal.status_code in {404, 422}
    assert missing.json()["error"]["code"] == "campaign_not_found"
