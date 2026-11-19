# ASTRAL Stage 10 — Atlas Product Interface

> Historical note: this buildless client has been superseded by the publication
> Next.js application in `web/`. FastAPI `/` and `/app/` now redirect to the
> Next.js interface. The document remains as Stage 10 implementation provenance.

## Purpose

The approved Atlas direction is now the primary user-facing ASTRAL interface. It is served by FastAPI at `/app/` and consumes the versioned scientific API on the same origin. Streamlit remains an internal developer instrument; it is no longer the product surface used to evaluate Stage 10.

The implementation is buildless HTML, CSS, and JavaScript so it remains easy to audit and publish. It does not embed invented observations, scores, or campaign metrics. Every dynamic value is requested from persisted ASTRAL artifacts through typed Pydantic contracts.

## Product areas connected to real services

- **Discover:** protected campaign planning, cache-only execution, real progress, campaign ledger, measured target diagnostics, BLS periodograms, detrended luminosity, and phase-folded evidence.
- **Observatory:** cached KOI objects, original NASA disposition, ANN-01/ANN-02 inference, and processed Kepler luminosity from the API.
- **Candidates:** only unassociated BLS screening signals from the selected persisted campaign, always labeled inconclusive and never as discoveries.
- **Neural Lab:** real inference scores and model metadata; topology remains explicitly explanatory rather than a simulated activation view.
- **Research:** like-for-like ANN metrics loaded from the persisted 47-object validation comparison.

## Discovery API

The following Stage 10 endpoints extend the existing API:

- `GET /api/v1/discovery/plan`
- `GET /api/v1/discovery/campaigns`
- `GET /api/v1/discovery/campaigns/{campaign_id}`
- `GET /api/v1/discovery/campaigns/{campaign_id}/targets/{kic}/series`
- `POST /api/v1/discovery/jobs`
- `GET /api/v1/discovery/jobs/{job_id}`

Campaign execution from HTTP is cache-only. The service forces `allow_network=false` and `max_new_downloads=0`; it cannot contact MAST, train a model, access frozen-test records, or confirm a detection. Only one campaign can run at a time.

For publication safety, execution is disabled by default. Local research sessions may enable it explicitly:

```powershell
$env:ASTRAL_DISCOVERY_EXECUTION_ENABLED = "true"
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/app/`. A public deployment without authentication must leave the execution flag disabled; persisted evidence remains readable while the launch button is unavailable.

## Real end-to-end validation

The product endpoint was tested through the running HTTP service. A 12-target campaign was submitted through `POST /api/v1/discovery/jobs`, progressed through the background worker, and completed as `web-20261119T154446Z-6b45fe59`:

- 12/12 targets investigated;
- 8 BLS screening detections;
- 3 local catalog-period associations;
- 5 unassociated, inconclusive screens;
- 0 confirmed discoveries;
- 0 new downloads;
- no neural training or frozen-test record access.

The result and bounded numerical series were then read back through the public product endpoints. Edge headless QA verified the live Discover, Candidates, and Observatory surfaces at 1440 px, including real API status, catalog objects, inference scores, and the five inconclusive signals.

The final repository-wide automated suite completed with `82 passed` in 26.78 seconds. JavaScript syntax validation also completed successfully with `node --check`.

## Scientific and deployment boundaries

- Detector SNR is not calibrated significance.
- A missing local period association is not novelty evidence.
- The standalone TCE catalog is not queried by this interface.
- Browser campaigns cannot download data.
- Public write access requires future authentication, rate limiting, and durable job infrastructure.
- The frontend and API are locally integrated, but no public deployment was performed in Stage 10.
- No Next.js dependency was introduced; the Atlas client can be migrated without changing the scientific API contracts.
