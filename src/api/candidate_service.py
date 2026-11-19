"""Read-only scientific dossiers plus explicit local human-review records."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.api.errors import ApiError
from src.candidate_intelligence.engine import STAGE11_ROOT


class CandidateIntelligenceService:
    """Serve persisted Stage 11 evidence without running catalogs or models."""

    def __init__(self, root: Path = STAGE11_ROOT) -> None:
        self.root = root
        self.index_path = root / "candidate_intelligence.json"
        self.reviews_path = root / "human_reviews.json"
        self._lock = threading.Lock()

    def _index(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ApiError(404, "candidate_intelligence_unavailable", "No persisted Stage 11 candidate intelligence artifact is available.") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ApiError(503, "candidate_intelligence_unreadable", "The persisted candidate intelligence artifact cannot be read.") from exc
        if payload.get("stage") != "stage11" or not isinstance(payload.get("dossiers"), list):
            raise ApiError(503, "candidate_intelligence_invalid", "The persisted candidate intelligence artifact is invalid.")
        return payload

    def _reviews(self) -> list[dict[str, Any]]:
        if not self.reviews_path.is_file():
            return []
        try:
            payload = json.loads(self.reviews_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ApiError(503, "review_history_unreadable", "The local review history cannot be read.") from exc
        return payload if isinstance(payload, list) else []

    def list(self) -> dict[str, Any]:
        payload = self._index()
        reviews = self._reviews()
        review_counts: dict[int, int] = {}
        for review in reviews:
            kic = int(review.get("kic", -1))
            review_counts[kic] = review_counts.get(kic, 0) + 1
        return {
            "campaign_id": payload["campaign_id"],
            "generated_at_utc": payload["generated_at_utc"],
            "artifact_sha256": payload.get("artifact_sha256"),
            "summary": payload["summary"],
            "ranking_method": payload["ranking_method"],
            "catalog_sources": payload["catalog_sources"],
            "scientific_boundaries": payload["scientific_boundaries"],
            "dossiers": [
                {
                    "dossier_id": dossier["dossier_id"],
                    "kic": dossier["kic"],
                    "selection_reference": dossier.get("selection_reference"),
                    "investigation_state": dossier["investigation_state"],
                    "period_days": dossier["measured_event"]["period_days"],
                    "duration_hours": dossier["measured_event"]["duration_hours"],
                    "depth_ppm": dossier["measured_event"]["depth_ppm"],
                    "depth_snr": dossier["measured_event"]["depth_snr"],
                    "observed_events": dossier["measured_event"]["observed_events"],
                    "event_coverage": dossier["measured_event"]["event_coverage"],
                    "possible_eclipsing_binary": dossier["diagnostics"]["possible_eclipsing_binary"],
                    "priority": dossier["priority"],
                    "review_count": review_counts.get(int(dossier["kic"]), 0),
                }
                for dossier in payload["dossiers"]
            ],
        }

    def detail(self, kic: int) -> dict[str, Any]:
        payload = self._index()
        dossier = next((item for item in payload["dossiers"] if int(item["kic"]) == int(kic)), None)
        if dossier is None:
            raise ApiError(404, "dossier_not_found", "No Stage 11 dossier matches this KIC identifier.")
        result = dict(dossier)
        result["review_history"] = [review for review in self._reviews() if int(review.get("kic", -1)) == int(kic)]
        return result

    def markdown(self, kic: int) -> str:
        self.detail(kic)
        path = self.root / "dossiers" / f"kic_{int(kic):09d}.md"
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ApiError(503, "report_unreadable", "The scientific Markdown report cannot be read.") from exc

    def reviews(self, kic: int) -> list[dict[str, Any]]:
        self.detail(kic)
        return [review for review in self._reviews() if int(review.get("kic", -1)) == int(kic)]

    def add_review(self, kic: int, reviewer: str, decision: str, notes: str) -> dict[str, Any]:
        self.detail(kic)
        review = {
            "review_id": str(uuid.uuid4()),
            "kic": int(kic),
            "reviewer": reviewer.strip(),
            "decision": decision,
            "notes": notes.strip(),
            "created_at_utc": datetime.now(UTC).isoformat(),
        }
        with self._lock:
            reviews = self._reviews()
            reviews.append(review)
            self.root.mkdir(parents=True, exist_ok=True)
            temporary = self.reviews_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(reviews, indent=2, ensure_ascii=False), encoding="utf-8")
            temporary.replace(self.reviews_path)
        return review
