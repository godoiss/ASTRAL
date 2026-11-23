"""SQLite-backed operational metadata and event history for research campaigns."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE = PROJECT_ROOT / "data" / "research" / "campaigns.sqlite3"
CAMPAIGN_STATES = {
    "planned",
    "queued",
    "running",
    "completed",
    "failed",
    "cancelled",
    "interrupted",
}
ACTIVE_STATES = {"queued", "running"}
TRANSITIONS = {
    "planned": {"queued", "cancelled"},
    "queued": {"running", "failed", "cancelled", "interrupted"},
    "running": {"completed", "failed", "cancelled", "interrupted"},
    "failed": {"queued", "cancelled"},
    "cancelled": {"queued"},
    "interrupted": {"queued", "cancelled"},
    "completed": set(),
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class CampaignStore:
    """Persist campaign state transitions and append-only events atomically."""

    def __init__(self, path: Path = DEFAULT_DATABASE, *, recover_active: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()
        if recover_active:
            self.recover_interrupted()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS campaign_jobs (
                    job_id TEXT PRIMARY KEY,
                    campaign_id TEXT NOT NULL UNIQUE,
                    mission TEXT NOT NULL CHECK (mission IN ('Kepler')),
                    selection_strategy TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    current INTEGER NOT NULL DEFAULT 0 CHECK (current >= 0),
                    total INTEGER NOT NULL CHECK (total >= 1),
                    current_kic INTEGER,
                    message TEXT NOT NULL,
                    error_code TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK (cancel_requested IN (0, 1)),
                    allow_network INTEGER NOT NULL DEFAULT 0 CHECK (allow_network IN (0, 1)),
                    max_new_downloads INTEGER NOT NULL DEFAULT 0 CHECK (max_new_downloads >= 0),
                    campaign_config_json TEXT NOT NULL,
                    detector_config_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    resource_json TEXT NOT NULL,
                    result_summary_json TEXT,
                    artifact_path TEXT,
                    created_at_utc TEXT NOT NULL,
                    queued_at_utc TEXT,
                    started_at_utc TEXT,
                    completed_at_utc TEXT,
                    updated_at_utc TEXT NOT NULL,
                    CHECK (status IN ('planned','queued','running','completed','failed','cancelled','interrupted'))
                );

                CREATE TABLE IF NOT EXISTS campaign_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL REFERENCES campaign_jobs(job_id) ON DELETE CASCADE,
                    created_at_utc TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    level TEXT NOT NULL CHECK (level IN ('info','warning','error')),
                    stage TEXT NOT NULL,
                    current INTEGER NOT NULL DEFAULT 0,
                    total INTEGER NOT NULL DEFAULT 0,
                    kic INTEGER,
                    message TEXT NOT NULL,
                    elapsed_seconds REAL,
                    details_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_campaign_jobs_status_updated
                    ON campaign_jobs(status, updated_at_utc DESC);
                CREATE INDEX IF NOT EXISTS idx_campaign_events_job_id
                    ON campaign_events(job_id, event_id);
                """
            )

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        for key in (
            "campaign_config_json",
            "detector_config_json",
            "provenance_json",
            "resource_json",
            "result_summary_json",
        ):
            name = key.removesuffix("_json")
            raw = result.pop(key, None)
            result[name] = json.loads(raw) if raw else None
        result["cancel_requested"] = bool(result["cancel_requested"])
        result["allow_network"] = bool(result["allow_network"])
        return result

    @staticmethod
    def _decode_event(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["details"] = json.loads(result.pop("details_json"))
        return result

    def create_plan(
        self,
        *,
        campaign_id: str,
        mission: str,
        selection_strategy: str,
        total: int,
        allow_network: bool,
        max_new_downloads: int,
        campaign_config: dict[str, Any],
        detector_config: dict[str, Any],
        provenance: dict[str, Any],
        resources: dict[str, Any],
    ) -> dict[str, Any]:
        if mission != "Kepler":
            raise ValueError("research campaigns currently support Kepler only")
        job_id = str(uuid.uuid4())
        created = _now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO campaign_jobs (
                    job_id, campaign_id, mission, selection_strategy, status, stage,
                    current, total, message, allow_network, max_new_downloads,
                    campaign_config_json, detector_config_json, provenance_json,
                    resource_json, created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, 'planned', 'planned', 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    campaign_id,
                    mission,
                    selection_strategy,
                    total,
                    "Campaign plan persisted; execution has not started.",
                    int(allow_network),
                    max_new_downloads,
                    _json(campaign_config),
                    _json(detector_config),
                    _json(provenance),
                    _json(resources),
                    created,
                    created,
                ),
            )
            self._insert_event(
                connection,
                job_id=job_id,
                event_type="campaign_planned",
                level="info",
                stage="planned",
                current=0,
                total=total,
                message="Campaign plan validated and persisted.",
                details={"mission": mission, "allow_network": allow_network},
            )
            connection.commit()
        return self.get(job_id)

    def _insert_event(
        self,
        connection: sqlite3.Connection,
        *,
        job_id: str,
        event_type: str,
        level: str,
        stage: str,
        current: int,
        total: int,
        message: str,
        kic: int | None = None,
        elapsed_seconds: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO campaign_events (
                job_id, created_at_utc, event_type, level, stage, current, total,
                kic, message, elapsed_seconds, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                _now(),
                event_type,
                level,
                stage,
                current,
                total,
                kic,
                message[:500],
                elapsed_seconds,
                _json(details or {}),
            ),
        )

    def append_event(self, job_id: str, **event: Any) -> None:
        with self._lock, self._connect() as connection:
            self._insert_event(connection, job_id=job_id, **event)
            connection.commit()

    def get(self, job_id: str) -> dict[str, Any]:
        try:
            uuid.UUID(job_id)
        except ValueError as exc:
            raise ValueError("invalid job identifier") from exc
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM campaign_jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._decode(row)

    def get_by_campaign(self, campaign_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM campaign_jobs WHERE campaign_id = ?", (campaign_id,)
            ).fetchone()
        if row is None:
            raise KeyError(campaign_id)
        return self._decode(row)

    def list(self, *, offset: int, limit: int, statuses: Iterable[str] | None = None) -> tuple[list[dict[str, Any]], int]:
        values = tuple(statuses or ())
        if any(value not in CAMPAIGN_STATES for value in values):
            raise ValueError("invalid campaign status filter")
        where = ""
        parameters: list[Any] = []
        if values:
            where = " WHERE status IN (" + ",".join("?" for _ in values) + ")"
            parameters.extend(values)
        with self._connect() as connection:
            total = int(connection.execute(f"SELECT COUNT(*) FROM campaign_jobs{where}", parameters).fetchone()[0])
            rows = connection.execute(
                f"SELECT * FROM campaign_jobs{where} ORDER BY created_at_utc DESC LIMIT ? OFFSET ?",
                [*parameters, limit, offset],
            ).fetchall()
        return [self._decode(row) for row in rows], total

    def events(self, job_id: str, *, offset: int, limit: int) -> tuple[list[dict[str, Any]], int]:
        self.get(job_id)
        with self._connect() as connection:
            total = int(
                connection.execute("SELECT COUNT(*) FROM campaign_events WHERE job_id = ?", (job_id,)).fetchone()[0]
            )
            rows = connection.execute(
                "SELECT * FROM campaign_events WHERE job_id = ? ORDER BY event_id LIMIT ? OFFSET ?",
                (job_id, limit, offset),
            ).fetchall()
        return [self._decode_event(row) for row in rows], total

    def transition(
        self,
        job_id: str,
        status: str,
        *,
        stage: str,
        message: str,
        error_code: str | None = None,
        result_summary: dict[str, Any] | None = None,
        artifact_path: str | None = None,
        event_type: str | None = None,
        level: str = "info",
    ) -> dict[str, Any]:
        if status not in CAMPAIGN_STATES:
            raise ValueError("invalid campaign status")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM campaign_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            previous = str(row["status"])
            if status != previous and status not in TRANSITIONS[previous]:
                raise ValueError(f"invalid state transition: {previous} -> {status}")
            now = _now()
            queued = now if status == "queued" else row["queued_at_utc"]
            started = now if status == "running" and not row["started_at_utc"] else row["started_at_utc"]
            completed = now if status in {"completed", "failed", "cancelled"} else None
            connection.execute(
                """
                UPDATE campaign_jobs SET status = ?, stage = ?, message = ?, error_code = ?,
                    queued_at_utc = ?, started_at_utc = ?, completed_at_utc = ?, updated_at_utc = ?,
                    cancel_requested = CASE WHEN ? = 'queued' THEN 0 ELSE cancel_requested END,
                    result_summary_json = COALESCE(?, result_summary_json),
                    artifact_path = COALESCE(?, artifact_path)
                WHERE job_id = ?
                """,
                (
                    status,
                    stage,
                    message[:500],
                    error_code,
                    queued,
                    started,
                    completed,
                    now,
                    status,
                    _json(result_summary) if result_summary is not None else None,
                    artifact_path,
                    job_id,
                ),
            )
            self._insert_event(
                connection,
                job_id=job_id,
                event_type=event_type or f"campaign_{status}",
                level=level,
                stage=stage,
                current=int(row["current"]),
                total=int(row["total"]),
                kic=row["current_kic"],
                message=message,
                details={"previous_status": previous, "status": status},
            )
            connection.commit()
        return self.get(job_id)

    def progress(
        self,
        job_id: str,
        *,
        stage: str,
        current: int,
        total: int,
        message: str,
        kic: int | None,
        elapsed_seconds: float | None,
    ) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT status FROM campaign_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            if row["status"] not in ACTIVE_STATES:
                connection.rollback()
                return self.get(job_id)
            connection.execute(
                """
                UPDATE campaign_jobs SET status = 'running', stage = ?, current = ?, total = ?,
                    current_kic = ?, message = ?, started_at_utc = COALESCE(started_at_utc, ?),
                    updated_at_utc = ? WHERE job_id = ?
                """,
                (stage, current, total, kic, message[:500], _now(), _now(), job_id),
            )
            self._insert_event(
                connection,
                job_id=job_id,
                event_type="stage_progress",
                level="info",
                stage=stage,
                current=current,
                total=total,
                kic=kic,
                message=message,
                elapsed_seconds=elapsed_seconds,
                details={},
            )
            connection.commit()
        return self.get(job_id)

    def request_cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM campaign_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            if row["status"] not in {"planned", "queued", "running"}:
                raise ValueError(f"cannot cancel campaign in {row['status']} state")
            if row["status"] == "planned":
                connection.rollback()
                return self.transition(
                    job_id,
                    "cancelled",
                    stage="cancelled",
                    message="Planned campaign cancelled before execution.",
                )
            connection.execute(
                "UPDATE campaign_jobs SET cancel_requested = 1, message = ?, updated_at_utc = ? WHERE job_id = ?",
                ("Cancellation requested; the active scientific stage will finish safely.", _now(), job_id),
            )
            self._insert_event(
                connection,
                job_id=job_id,
                event_type="cancellation_requested",
                level="warning",
                stage=str(row["stage"]),
                current=int(row["current"]),
                total=int(row["total"]),
                kic=row["current_kic"],
                message="Cancellation requested; waiting for a safe scientific boundary.",
                details={},
            )
            connection.commit()
        return self.get(job_id)

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM campaign_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return bool(row and row[0])

    def recover_interrupted(self) -> int:
        """Mark orphaned active jobs interrupted after a process restart."""
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT * FROM campaign_jobs WHERE status IN ('queued','running')"
            ).fetchall()
            for row in rows:
                connection.execute(
                    """
                    UPDATE campaign_jobs SET status = 'interrupted', stage = 'interrupted',
                        message = ?, updated_at_utc = ? WHERE job_id = ?
                    """,
                    (
                        "Local worker stopped before a terminal state; persisted artifacts remain available for safe resume.",
                        _now(),
                        row["job_id"],
                    ),
                )
                self._insert_event(
                    connection,
                    job_id=row["job_id"],
                    event_type="worker_interrupted",
                    level="warning",
                    stage="interrupted",
                    current=int(row["current"]),
                    total=int(row["total"]),
                    kic=row["current_kic"],
                    message="Active state recovered as interrupted after service initialization.",
                    details={"previous_status": row["status"]},
                )
            connection.commit()
        return len(rows)
