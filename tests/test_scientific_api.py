"""Integration and safety tests for the local Stage 08 FastAPI backend."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.config import ApiSettings
from src.api.dependencies import get_scientific_service
from src.api.services import PROJECT_ROOT, public_payload
from src.observatory.persistence import artifact_is_cached


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Use the real local artifacts while keeping every endpoint read-only."""
    get_scientific_service.cache_clear()
    settings = ApiSettings(
        cors_origins=("http://localhost:3000",),
        allowed_hosts=("testserver", "localhost", "127.0.0.1"),
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client
    get_scientific_service.cache_clear()


def _assert_no_internal_path(payload: object) -> None:
    serialized = json.dumps(payload, ensure_ascii=False)
    assert str(PROJECT_ROOT).lower() not in serialized.lower()
    assert "D:\\OneDrive".lower() not in serialized.lower()


def test_health_version_and_restrictive_cors(client: TestClient) -> None:
    health = client.get("/health")
    info = client.get("/api/v1/info")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["version"] == "0.16.0"
    assert health.json()["stage"] == 16
    assert info.json()["scientific_guardrails"] == {
        "frozen_test_accessed": False,
        "training_enabled": False,
        "downloads_enabled": False,
        "automatic_ensemble_enabled": False,
        "kepler_tess_mixed": False,
        "campaign_execution_enabled": False,
        "candidate_reviews_enabled": False,
        "public_read_only": False,
    }
    assert health.headers["cache-control"] == "no-store"
    assert health.headers["x-request-id"]

    allowed = client.options(
        "/api/v1/objects",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    denied = client.options(
        "/api/v1/objects",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert denied.status_code == 400


def test_public_profile_blocks_scientific_mutations() -> None:
    settings = ApiSettings(
        deployment_mode="public_read_only",
        discovery_execution_enabled=True,
        candidate_reviews_enabled=True,
        allowed_hosts=("testserver",),
    )
    with TestClient(create_app(settings)) as public:
        health = public.get("/health")
        review = public.post(
            "/api/v1/candidate-intelligence/10811496/reviews",
            json={"reviewer": "Anonymous", "decision": "needs_follow_up", "notes": "Must not be persisted."},
        )
        campaign = public.post(
            "/api/v1/discovery/campaigns",
            json={"sample_size": 1, "allow_network": False, "max_new_downloads": 0},
        )

    assert health.json()["local_only"] is False
    assert review.status_code == 403
    assert campaign.status_code == 403
    assert review.json()["error"]["code"] == "operation_disabled"


def test_request_size_and_mutation_rate_are_bounded() -> None:
    settings = ApiSettings(
        allowed_hosts=("testserver",),
        maximum_request_body_bytes=1024,
        rate_limit_per_minute=10,
    )
    with TestClient(create_app(settings)) as bounded:
        oversized = bounded.post(
            "/api/v1/inference/ann-01",
            content=b"x" * 1025,
            headers={"Content-Type": "application/json"},
        )
        responses = [bounded.post("/route-that-does-not-exist") for _ in range(11)]

    assert oversized.status_code == 413
    assert oversized.json()["error"]["code"] == "request_too_large"
    assert responses[-1].status_code == 429
    assert responses[-1].headers["retry-after"] == "60"


def test_openapi_exposes_persistent_campaign_control_without_training_or_download_routes(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]

    operations = [
        operation
        for path_item in paths.values()
        for method, operation in path_item.items()
        if method.lower() in {"get", "post", "put", "patch", "delete"}
    ]

    assert "/api/v1/inference/{model_id}" in paths
    assert len(operations) == 40
    assert len({operation["operationId"] for operation in operations}) == 40
    assert "/api/v1/candidate-intelligence" in paths
    assert "/api/v1/candidate-intelligence/{kic}" in paths
    assert "/api/v1/candidate-intelligence/{kic}/export" in paths
    assert "/api/v1/candidate-intelligence/{kic}/report" in paths
    assert "/api/v1/candidate-intelligence/{kic}/reviews" in paths
    assert not any("train" in path.lower() or "download" in path.lower() for path in paths)
    write_operations = [
        (path, method)
        for path, operations in paths.items()
        for method in operations
        if method.lower() in {"post", "put", "patch", "delete"}
    ]
    assert write_operations == [
        ("/api/v1/inference/{model_id}", "post"),
        ("/api/v1/discovery/campaigns", "post"),
        ("/api/v1/discovery/jobs", "post"),
        ("/api/v1/discovery/jobs/{job_id}/start", "post"),
        ("/api/v1/discovery/jobs/{job_id}/cancel", "post"),
        ("/api/v1/discovery/jobs/{job_id}/resume", "post"),
        ("/api/v1/candidate-intelligence/{kic}/reviews", "post"),
    ]

    for operation in operations:
        success = operation["responses"].get("200") or operation["responses"].get("201") or operation["responses"].get("202")
        content = success["content"]
        assert content
        assert all(media["schema"] for media in content.values())
        assert operation["responses"]["422"]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/ErrorResponse"
        }


def test_all_eighteen_operations_have_a_real_success_path(client: TestClient) -> None:
    experiment_page = client.get("/api/v1/experiments", params={"stage": "stage07", "limit": 1})
    experiment_id = experiment_page.json()["items"][0]["experiment_id"]
    requests = [
        ("get", "/health", None),
        ("get", "/api/v1/info", None),
        ("get", "/api/v1/datasets", None),
        ("get", "/api/v1/observatory/summary", None),
        ("get", "/api/v1/objects?query=K00753.01&limit=1", None),
        ("get", "/api/v1/objects/K00753.01", None),
        ("get", "/api/v1/objects/K00753.01/observation", None),
        ("get", "/api/v1/objects/K00753.01/light-curve?view=phase&limit=50", None),
        ("get", "/api/v1/models", None),
        ("get", "/api/v1/models/ann-02", None),
        ("get", "/api/v1/models/ann-02/history?limit=5", None),
        ("post", "/api/v1/inference/ann-02", {"object_id": "K00753.01"}),
        ("get", "/api/v1/metrics/comparison", None),
        ("get", "/api/v1/explanations", None),
        ("get", "/api/v1/explanations/K00753.01?include_saliency=true&saliency_points=50", None),
        ("get", "/api/v1/experiments?stage=stage07&limit=1", None),
        ("get", f"/api/v1/experiments/{experiment_id}", None),
        ("get", "/api/v1/tess/experimental", None),
    ]

    assert len(requests) == 18
    for method, url, body in requests:
        response = client.request(method, url, json=body)
        assert response.status_code == 200, (method, url, response.text)
        _assert_no_internal_path(response.json())


def test_dataset_and_observatory_summaries_are_sanitized(client: TestClient) -> None:
    datasets = client.get("/api/v1/datasets")
    observatory = client.get("/api/v1/observatory/summary")

    assert datasets.status_code == observatory.status_code == 200
    items = datasets.json()["data"]["datasets"]
    assert {item["mission"] for item in items} == {"Kepler", "TESS"}
    ann02 = next(item for item in items if item["id"] == "ann02-kepler-q3")
    assert ann02["accepted"] == 194
    assert ann02["frozen_test_excluded"] is True
    assert observatory.json()["data"]["guardrails"]["frozen_test_accessed"] is False
    _assert_no_internal_path(datasets.json())
    _assert_no_internal_path(observatory.json())


def test_object_catalog_is_paginated_and_validated(client: TestClient) -> None:
    response = client.get("/api/v1/objects", params={"limit": 3, "offset": 2})
    invalid = client.get("/api/v1/objects", params={"limit": 101})

    assert response.status_code == 200
    assert len(response.json()["items"]) == 3
    assert response.json()["pagination"]["offset"] == 2
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "validation_error"


def test_frozen_test_identity_is_not_served_or_inferred(client: TestClient) -> None:
    manifest = pd.read_csv(PROJECT_ROOT / "experiments/stage04/split_manifest.csv")
    test_object = str(manifest.loc[manifest["split"].eq("test"), "kepoi_name"].iloc[0])

    detail = client.get(f"/api/v1/objects/{test_object}")
    ann01 = client.post("/api/v1/inference/ann-01", json={"object_id": test_object})
    ann02 = client.post("/api/v1/inference/ann-02", json={"object_id": test_object})

    assert detail.status_code == ann01.status_code == ann02.status_code == 404
    assert detail.json()["error"]["code"] == "object_not_found"


def test_real_candidate_metadata_observation_and_bounded_curves(client: TestClient) -> None:
    detail = client.get("/api/v1/objects/K00753.01")
    observation = client.get("/api/v1/objects/K00753.01/observation")

    assert detail.status_code == observation.status_code == 200
    assert detail.json()["data"]["disposition"] == "CANDIDATE"
    assert detail.json()["data"]["split"] == "candidate"
    assert detail.json()["data"]["ra_deg"] == pytest.approx(297.00482)
    assert detail.json()["data"]["dec_deg"] == pytest.approx(48.134129)
    assert observation.json()["data"]["mission"] == "Kepler"
    for view in ("raw", "processed", "phase", "local"):
        curve = client.get(
            "/api/v1/objects/K00753.01/light-curve",
            params={"view": view, "limit": 25},
        )
        assert curve.status_code == 200
        assert curve.json()["data"]["records_returned"] <= 25
        assert curve.json()["data"]["view"] == view
        _assert_no_internal_path(curve.json())


def test_uncached_observation_returns_structured_not_found(client: TestClient) -> None:
    service = get_scientific_service()
    object_id = next(
        str(value)
        for value in service.catalog["kepoi_name"]
        if not artifact_is_cached(str(value))
    )
    response = client.get(f"/api/v1/objects/{object_id}/observation")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "observation_not_found"
    assert str(PROJECT_ROOT) not in response.text


def test_model_catalog_pins_stage07_selected_ann02(client: TestClient) -> None:
    response = client.get("/api/v1/models")

    assert response.status_code == 200
    models = {item["id"]: item for item in response.json()["data"]["models"]}
    assert models["ann-01"]["shared_validation_metrics"]["accuracy"] == pytest.approx(0.8723404253)
    assert models["ann-02"]["experiment_id"] == "ann02-20261118T180835Z"
    assert models["ann-02"]["served_checkpoint"] == "seed_stability"
    assert models["ann-02"]["seed"] == 97
    assert models["ann-02"]["shared_validation_metrics"]["accuracy"] == pytest.approx(0.7659574468)


def test_real_ann01_and_ann02_candidate_inference(client: TestClient) -> None:
    ann01 = client.post("/api/v1/inference/ann-01", json={"object_id": "K00753.01"})
    ann02 = client.post("/api/v1/inference/ann-02", json={"object_id": "K00753.01"})

    assert ann01.status_code == ann02.status_code == 200
    assert ann01.json()["original_disposition"] == "CANDIDATE"
    assert ann02.json()["model_experiment_id"] == "ann02-20261118T180835Z"
    assert 0.0 <= ann01.json()["score"] <= 1.0
    assert 0.0 <= ann02.json()["score"] <= 1.0
    assert "not exoplanet confirmations" in ann02.json()["scientific_notice"]
    assert ann02.json()["details"]["feature_map_shapes"] == {
        "global.relu_2": [16, 500],
        "local.relu_2": [16, 100],
    }


def test_real_neural_visualization_exposes_bounded_model_state(client: TestClient) -> None:
    ann01 = client.get(
        "/api/v1/models/ann-01/visualization",
        params={"object_id": "K00753.01"},
    )
    ann02 = client.get(
        "/api/v1/models/ann-02/visualization",
        params={"object_id": "K00753.01"},
    )

    assert ann01.status_code == ann02.status_code == 200
    assert ann01.json()["data"]["object_id"] == "K00753.01"
    assert sum(len(layer["values"]) for layer in ann01.json()["data"]["parameters"]) > 0
    maps = {item["name"]: item for item in ann02.json()["data"]["activations"]}
    assert maps["global.relu_2"]["shape"] == [16, 500]
    assert maps["local.relu_2"]["shape"] == [16, 100]
    assert maps["global.relu_2"]["role"] == "feature_map"
    _assert_no_internal_path(ann01.json())
    _assert_no_internal_path(ann02.json())


def test_inference_errors_are_consistent(client: TestClient) -> None:
    unknown_model = client.post("/api/v1/inference/ann-99", json={"object_id": "K00753.01"})
    invalid_body = client.post("/api/v1/inference/ann-01", json={"object_id": "../secret"})
    missing = client.get("/api/v1/objects/DOES-NOT-EXIST")
    unknown_route = client.get("/api/v1/not-a-route")
    invalid_path = client.get("/api/v1/objects/%2E%2E%5Csecret")

    assert unknown_model.status_code == 404
    assert unknown_model.json()["error"]["code"] == "model_not_found"
    assert invalid_body.status_code == 422
    assert invalid_body.json()["error"]["code"] == "validation_error"
    assert missing.status_code == 404
    assert unknown_route.status_code == 404
    assert unknown_route.json()["error"]["code"] == "route_not_found"
    assert invalid_path.status_code == 422
    assert invalid_path.json()["error"]["code"] == "validation_error"


def test_metrics_explanations_and_saliency_are_persisted_and_bounded(client: TestClient) -> None:
    comparison = client.get("/api/v1/metrics/comparison")
    index = client.get("/api/v1/explanations")
    explained_id = index.json()["data"]["objects"][0]["object_id"]
    explanation = client.get(
        f"/api/v1/explanations/{explained_id}",
        params={"include_saliency": True, "saliency_points": 50},
    )

    assert comparison.status_code == index.status_code == explanation.status_code == 200
    assert comparison.json()["data"]["cohort"]["records"] == 47
    assert comparison.json()["data"]["agreement"]["records"] == 32
    saliency = explanation.json()["data"]["ann02"]["saliency"]
    assert all(item["returned_points"] <= 50 for item in saliency.values())
    assert "causality" in explanation.json()["limitations"][0]


def test_experiments_histories_and_tess_remain_read_only(client: TestClient) -> None:
    experiments = client.get("/api/v1/experiments", params={"stage": "stage07"})
    history = client.get("/api/v1/models/ann-02/history", params={"limit": 5})
    tess = client.get("/api/v1/tess/experimental")

    assert experiments.status_code == history.status_code == tess.status_code == 200
    assert experiments.json()["pagination"]["total"] >= 6
    assert len(history.json()["items"]) == 5
    assert tess.json()["data"]["target"] == "TIC 441420236"
    assert tess.json()["data"]["guardrails"]["mixed_with_kepler_training"] is False
    _assert_no_internal_path(tess.json())


def test_experiment_registry_has_unique_semantic_identities(client: TestClient) -> None:
    response = client.get("/api/v1/experiments", params={"stage": "stage06", "limit": 100})

    assert response.status_code == 200
    experiments = response.json()["items"]
    identities = [(item["stage"], item["experiment_id"]) for item in experiments]
    assert len(identities) == len(set(identities))

    pilot = [item for item in experiments if item["experiment_id"] == "ann02-20261118T124527Z"]
    assert len(pilot) == 1
    assert pilot[0]["generated_at_utc"] is not None


def test_endpoint_reads_never_call_training_or_downloads(client: TestClient) -> None:
    forbidden = RuntimeError("forbidden scientific side effect")
    with (
        patch("src.observatory.mast.search_and_download", side_effect=forbidden) as download,
        patch("src.deep_learning.training.train_ann02", side_effect=forbidden) as ann02_train,
        patch("src.lab.services.train_developer_model", side_effect=forbidden) as ann01_train,
    ):
        assert client.get("/api/v1/datasets").status_code == 200
        assert client.get("/api/v1/objects/K00753.01/observation").status_code == 200
        assert client.post("/api/v1/inference/ann-02", json={"object_id": "K00753.01"}).status_code == 200
    download.assert_not_called()
    ann02_train.assert_not_called()
    ann01_train.assert_not_called()


def test_public_payload_removes_nested_absolute_paths() -> None:
    payload = public_payload(
        {
            "safe": "Kepler",
            "path": str(PROJECT_ROOT / "secret.json"),
            "nested": {"value": str(PROJECT_ROOT / "models/model.pt")},
        }
    )

    assert payload == {"safe": "Kepler", "nested": {"value": None}}
