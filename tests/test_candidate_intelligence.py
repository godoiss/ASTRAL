"""Scientific, API, and persistence tests for Stage 11 candidate intelligence."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.candidate_service import CandidateIntelligenceService
from src.api.config import ApiSettings
from src.api.dependencies import get_candidate_intelligence_service
from src.candidate_intelligence.catalogs import ExoplanetArchiveClient
from src.candidate_intelligence.engine import PROJECT_ROOT, build_candidate_intelligence


class PersistedCatalogClient:
    def verify(self, kics: list[int]) -> dict[str, object]:
        payload = json.loads((PROJECT_ROOT / "experiments" / "stage11" / "catalog_verification.json").read_text(encoding="utf-8"))
        assert sorted(kics) == sorted(payload["kics"])
        return payload


def test_official_catalog_client_is_bounded_and_uses_whitelisted_tables() -> None:
    urls: list[str] = []

    def fetch(url: str, timeout: float) -> bytes:
        urls.append(url)
        assert timeout == 7.0
        if "cumulative" in url:
            return b'[{"kepid": 42, "kepoi_name": "K00042.01"}]'
        return b'[{"kepid": 42, "tce_plnt_num": 1}]'

    result = ExoplanetArchiveClient(timeout_seconds=7.0, fetch=fetch).verify([42, 42])

    assert len(urls) == 2
    assert result["kics"] == [42]
    assert result["koi_records"][0]["kepid"] == 42
    assert result["tce_records"][0]["tce_plnt_num"] == 1
    assert all("kepid+in+%2842%29" in url for url in urls)


def test_stage11_builder_reproduces_all_eight_real_screening_dossiers(tmp_path: Path) -> None:
    report = build_candidate_intelligence(
        output_root=tmp_path,
        catalog_client=PersistedCatalogClient(),
        generated_at_utc="2026-11-19T20:00:00+00:00",
    )

    assert report["summary"] == {
        "investigated_signals": 8,
        "catalog_associated": 3,
        "unassociated_inconclusive": 5,
        "possible_eclipsing_binary": 1,
        "confirmed_discoveries": 0,
        "neural_models_invoked": False,
        "frozen_test_accessed": False,
    }
    assert len(report["dossiers"]) == 8
    assert all(dossier["priority"]["meaning"] == "human_review_priority_not_planet_probability" for dossier in report["dossiers"])
    assert all(dossier["neural_compatibility"]["models_invoked"] is False for dossier in report["dossiers"])
    assert len(list((tmp_path / "dossiers").glob("*.json"))) == 8
    assert len(list((tmp_path / "dossiers").glob("*.md"))) == 8


def test_candidate_api_serves_exports_and_all_real_series_without_side_effects() -> None:
    with TestClient(create_app()) as client:
        index = client.get("/api/v1/candidate-intelligence")
        assert index.status_code == 200
        dossiers = index.json()["data"]["dossiers"]
        assert len(dossiers) == 8
        for summary in dossiers:
            kic = summary["kic"]
            detail = client.get(f"/api/v1/candidate-intelligence/{kic}")
            export = client.get(f"/api/v1/candidate-intelligence/{kic}/export")
            report = client.get(f"/api/v1/candidate-intelligence/{kic}/report")
            series = client.get(
                f"/api/v1/discovery/campaigns/stage10-kepler-reference-final/targets/{kic}/series",
                params={"limit": 200},
            )
            assert detail.status_code == export.status_code == report.status_code == series.status_code == 200
            assert detail.json()["data"]["neural_compatibility"]["models_invoked"] is False
            assert detail.json()["data"]["catalog_verification"]["novelty_established"] is False
            assert "attachment" in export.headers["content-disposition"]
            assert report.headers["content-type"].startswith("text/markdown")
            assert series.json()["data"]["records_returned"] <= 200


def test_human_review_is_validated_and_persisted_locally(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "experiments" / "stage11"
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    service = CandidateIntelligenceService(tmp_path)
    app = create_app(ApiSettings(candidate_reviews_enabled=True))
    app.dependency_overrides[get_candidate_intelligence_service] = lambda: service

    with TestClient(app) as client:
        invalid = client.post(
            "/api/v1/candidate-intelligence/10811496/reviews",
            json={"reviewer": "x", "decision": "confirmed", "notes": "no"},
        )
        recorded = client.post(
            "/api/v1/candidate-intelligence/10811496/reviews",
            json={"reviewer": "ASTRAL reviewer", "decision": "needs_follow_up", "notes": "Independent quarter analysis remains required."},
        )
        history = client.get("/api/v1/candidate-intelligence/10811496/reviews")

    assert invalid.status_code == 422
    assert recorded.status_code == 201
    assert history.status_code == 200
    assert history.json()[0]["decision"] == "needs_follow_up"
    persisted = json.loads((tmp_path / "human_reviews.json").read_text(encoding="utf-8"))
    assert persisted[0]["review_id"] == recorded.json()["review_id"]
