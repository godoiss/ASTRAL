# ASTRAL Stage 12 — Discovery API & Research Infrastructure

## Outcome

Stage 12 turns the Stage 10 discovery workflow into a persistent, restart-safe local research operation. The scientific detector and its immutable JSON/NPZ artifacts remain independent from HTTP. FastAPI now manages bounded campaign plans and execution state, while the official Next.js Atlas interface exposes Mission Control in **Discover** and a provenance-aware campaign archive in **Research**.

No neural model was retrained, no frozen-test object was analyzed, Kepler and TESS were not mixed, and the validated Stage 10/11 artifacts were not rewritten. Scientific downloads remain disabled unless both deployment configuration and an individual campaign explicitly authorize them.

## Pre-implementation audit

The repository already contained:

- a synchronous, deterministic `run_campaign` scientific workflow with progress callbacks;
- atomic Stage 10 campaign, target, processed-series, detection, provenance, and report artifacts;
- protected-host exclusion based on the existing split manifest;
- resource limits for targets, products, disk use, and per-star processing;
- a FastAPI discovery service and an in-memory single-worker job manager;
- Atlas Discover and Research routes in the publication Next.js application.

The operational gap was the in-memory manager. Jobs disappeared on restart, exposed no append-only events, had no formal transition model, and could not be cooperatively cancelled or resumed. The UI could start a cache-only task but could not create and review a durable plan separately from execution.

## Architecture

```text
Next.js Discover / Research
          │ typed HTTP contracts
          ▼
FastAPI v1 discovery router
          │ orchestration only
          ▼
DiscoveryJobManager ───── CampaignStore (SQLite/WAL)
          │                     ├── campaign_jobs
          │                     └── campaign_events (append-only)
          ▼
run_campaign (scientific layer)
          │
          └── versioned JSON + NPZ campaign artifacts
```

SQLite stores only operational metadata, transitions, progress, timestamps, bounded resource facts, provenance references, and event history. Scientific arrays and measurements remain in the versioned campaign files. Every public artifact path is project-relative; internal filesystem paths are not returned.

## State model and recovery

Supported states are `planned`, `queued`, `running`, `completed`, `failed`, `cancelled`, and `interrupted`. `CampaignStore` validates every transition in a transaction. Completed jobs are terminal. Failed, cancelled, and interrupted jobs may return to `queued` only through the explicit resume operation.

At service startup, orphaned `queued` or `running` jobs become `interrupted` with a persisted recovery event. Resume validates the stored normalized configuration and target identities, reopens the existing scientific campaign, skips targets already persisted as completed or failed, and continues the remaining targets. Cancellation is cooperative at target, acquisition, processing, and BLS boundaries; partial results and the cancelled campaign summary remain inspectable.

SQLite uses WAL, foreign keys, a busy timeout, indexes, parameterized statements, and short explicit transactions. The operational database is local and ignored by Git at `data/research/campaigns.sqlite3`; the schema is recreated automatically and records survive application restarts.

## API contracts

The API is version `0.12.0` and exposes 40 documented OpenAPI operations. Stage 12 adds:

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/discovery/campaigns` | Validate and persist a plan without starting it |
| `GET` | `/api/v1/discovery/jobs` | Paginated/filterable operational history |
| `GET` | `/api/v1/discovery/jobs/{job_id}` | Current durable state |
| `POST` | `/api/v1/discovery/jobs/{job_id}/start` | Start a reviewed plan |
| `POST` | `/api/v1/discovery/jobs/{job_id}/cancel` | Request safe cancellation |
| `POST` | `/api/v1/discovery/jobs/{job_id}/resume` | Resume a recoverable campaign |
| `GET` | `/api/v1/discovery/jobs/{job_id}/events` | Paginated append-only event stream |
| `GET` | `/api/v1/discovery/jobs/{job_id}/provenance` | Config, source, resources, paths, and guardrails |
| `GET` | `/api/v1/discovery/campaigns/{id}/detections` | Persisted detections only |
| `GET` | `/api/v1/discovery/campaigns/{id}/export` | Bounded portable campaign projection |

Existing plan, campaign detail/series, dossier, experiment, model, observation, inference, and candidate-intelligence contracts remain available. The earlier `POST /discovery/jobs` route is retained for compatibility but now uses the same persistent store.

Mutating endpoints reject execution when `ASTRAL_DISCOVERY_EXECUTION_ENABLED` is false. If `ASTRAL_OPERATION_KEY` is configured, `X-ASTRAL-Operation-Key` is required with constant-time comparison; without a key, mutation is limited to local/test clients. Network acquisition additionally requires `ASTRAL_DISCOVERY_DOWNLOADS_ENABLED=true`, per-plan `allow_network=true`, and a bounded `max_new_downloads` not exceeding the deployment limit. This lightweight key is a local-stage control, not a substitute for identity-aware authentication, TLS, authorization policy, or audit forwarding in a public deployment.

## Official Atlas experience

Discover now acts as Mission Control rather than issuing an opaque background request. It provides:

- real-coordinate stellar atlas and active-target highlight;
- bounded campaign parameters and explicit cache/network authority;
- persisted plan review before a separate start action;
- live stage, current KIC, progress, event telemetry, cancel, and resume controls;
- real persisted detections and operational campaign history;
- recoverable errors and a secondary audit table.

Research now indexes both operational campaigns and immutable scientific evidence. Users can select a campaign, inspect state, timestamps, target position, artifact presence, event chronology, provenance boundary, candidate dossiers, and pre-existing ANN experiment evidence. Tables remain secondary audit views.

## Real cache-only validation

Campaign `research-20261119T195423Z-f2b49a1b` was planned and started through the Stage 12 HTTP contracts with `allow_network=false`, `max_new_downloads=0`, and three cached unlabeled Kepler targets:

- KIC 9772531 (`K00950.01`)
- KIC 10155029 (`K03208.01`)
- KIC 8626325 (`K04775.01`)

It completed 3/3 targets in 1.264 seconds, recorded 13 operational events, produced three screening detections (one locally catalog-associated and two unassociated/inconclusive), started zero downloads, invoked no neural model, and reported zero confirmed discoveries. Its scientific artifact contains the exact KOI snapshot and split-manifest SHA-256 hashes plus NumPy, Pandas, Astropy, and Lightkurve versions. After a real FastAPI process restart, the same job and completed artifact were retrieved from the SQLite store.

Cancellation and resume are also covered by a deterministic orchestration integration test and by a real cached scientific campaign test: cancellation preserves a partial artifact, resume skips finished targets, and completion retains the same campaign identity. No test performs a network request.

## Validation evidence

- Python: `92 passed`, with two non-blocking dependency warnings.
- Stage 12 focused backend/discovery suite: `16 passed` during implementation.
- Next.js component suite: `5 passed`.
- TypeScript: `tsc --noEmit` passed.
- Next.js production build: seven publication routes compiled and eight static pages generated.
- OpenAPI: 40 unique operation IDs; schemas and standardized 422 responses validated.
- Browser: Discover and Research rendered successfully at 1600-pixel desktop widths with the real completed campaign visible.
- Restart: operational state and artifact reference survived FastAPI termination and startup.

The warnings are unchanged and non-blocking: Lightkurve's optional `tpfmodel` helper lacks `oktopus`, and Starlette's compatibility TestClient announces a future `httpx2` migration.

## Local execution

```powershell
$env:ASTRAL_DISCOVERY_EXECUTION_ENABLED = "true"
$env:ASTRAL_DISCOVERY_DOWNLOADS_ENABLED = "false"  # recommended cache-only default
# Optional local mutation key (minimum 16 characters):
# $env:ASTRAL_OPERATION_KEY = "replace-with-a-local-secret"

.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000

cd web
npm run dev -- --hostname 127.0.0.1 --port 3000
```

Open `http://127.0.0.1:3000/discover`, use **Persist reviewed plan**, then **Start expedition**. Research history is available at `http://127.0.0.1:3000/research`; Swagger remains at `http://127.0.0.1:8000/docs`.

## Limitations and next safeguards

- Execution remains a single local worker by design; this prevents CPU and disk contention but is not a distributed scheduler.
- Cooperative cancellation cannot interrupt a native library call in the middle of that call; it takes effect at the next safe scientific boundary.
- Resume is target-granular, not instruction-granular within a partially processed target.
- SQLite is suitable for this local workstation stage, not multi-host execution.
- The operation key provides local mutation gating only. A public deployment requires authenticated identities, role/claim authorization, TLS termination, secret management, rate limiting, and centralized audit retention.
- Screening detections remain investigative evidence, never discoveries or calibrated planet probabilities.
- TESS remains isolated and no mission data is mixed.

Stage 13 was not started.
