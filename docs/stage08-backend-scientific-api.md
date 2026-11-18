# Stage 08 — Backend & Scientific API

## Scope

Stage 08 introduces a local, read-only FastAPI boundary over the scientific
components already implemented by ASTRAL. It prepares a stable versioned contract
for a future Next.js client without moving scientific logic into HTTP handlers.

The API does not expose training, acquisition, model activation, artifact writes,
or automatic ensemble operations. It does not materialize or serve frozen test
identities. All responses are produced from existing local models, cached
observations, and persisted experiment artifacts.

## Architecture

```text
api.py                         ASGI entry point
src/api/
├── app.py                     FastAPI factory, middleware, error handlers
├── config.py                  Pydantic-validated local settings
├── dependencies.py            Cached dependency providers
├── errors.py                  Stable public error type
├── schemas.py                 Pydantic HTTP contracts
├── services.py                Read-only scientific application facade
└── routers/
    ├── health.py              Health and API identity
    ├── catalog.py             Datasets, objects, observations, curves
    ├── science.py             Models, inference, comparison, explanations
    └── experiments.py         Experiment registry and isolated TESS pilot
```

Routers only validate HTTP inputs and map service results into Pydantic response
models. `ScientificService` reuses `src.lab.services`, `src.observatory`,
`src.deep_learning`, and persisted Stage 07 JSON/NPZ artifacts. No competing
neural or signal-processing implementation was added.

ANN-02 serving is pinned to the immutable Stage 07 campaign selection
`ann02-20261118T180835Z` (`seed_stability`, seed 97). It is resolved from the
campaign registry and not from the mutable Developer Lab active-model pointer.

## Endpoints

All scientific endpoints are versioned under `/api/v1`.

| Method | Endpoint | Resource |
|---|---|---|
| GET | `/health` | Local readiness and artifact availability |
| GET | `/api/v1/info` | API version, documentation, guardrails |
| GET | `/api/v1/datasets` | Kepler catalog, ANN-02 cohort, isolated TESS pilot |
| GET | `/api/v1/observatory/summary` | Cohort, quality, astronomy, bias and exclusions |
| GET | `/api/v1/objects` | Paginated safe catalog; optional query/class/cache filters |
| GET | `/api/v1/objects/{object_id}` | KOI identity and physical/catalog metadata |
| GET | `/api/v1/objects/{object_id}/observation` | Cached observation and provenance |
| GET | `/api/v1/objects/{object_id}/light-curve` | Bounded raw/processed/phase/local arrays |
| GET | `/api/v1/models` | ANN-01 and selected ANN-02 catalog |
| GET | `/api/v1/models/{model_id}` | One model and shared-cohort metrics |
| GET | `/api/v1/models/{model_id}/history` | Paginated real training history |
| POST | `/api/v1/inference/{model_id}` | Read-only inference for an existing compatible KOI |
| GET | `/api/v1/metrics/comparison` | Persisted like-for-like comparison |
| GET | `/api/v1/explanations` | ANN-01 importance and persisted explanation index |
| GET | `/api/v1/explanations/{object_id}` | Persisted local ANN-01/ANN-02 explanation |
| GET | `/api/v1/experiments` | Paginated Stage 06/07 experiment registry |
| GET | `/api/v1/experiments/{experiment_id}` | Sanitized persisted experiment document |
| GET | `/api/v1/tess/experimental` | Clearly isolated TESS compatibility result |

Interactive OpenAPI documentation is available at `/docs`, ReDoc at `/redoc`,
and the machine-readable schema at `/openapi.json`.

## Request examples

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health

Invoke-RestMethod `
  'http://127.0.0.1:8000/api/v1/objects?query=K00753.01&limit=10'

Invoke-RestMethod `
  'http://127.0.0.1:8000/api/v1/objects/K00753.01/light-curve?view=local&limit=500'

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/v1/inference/ann-02 `
  -ContentType application/json `
  -Body '{"object_id":"K00753.01"}'
```

Inference scores are uncalibrated model outputs. `CONFIRMED-LIKE` does not alter
the NASA catalog disposition and does not represent a discovery or confirmation.

## Local execution and configuration

Install the pinned project dependencies and start Uvicorn from the repository
root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn api:app `
  --host 127.0.0.1 `
  --port 8000
```

The server is intentionally bound to loopback. Development CORS permits only
`http://localhost:3000` and `http://127.0.0.1:3000` by default. Optional local
overrides use comma-separated environment variables:

```powershell
$env:ASTRAL_API_CORS_ORIGINS='http://localhost:3000,http://127.0.0.1:3000'
$env:ASTRAL_API_ALLOWED_HOSTS='localhost,127.0.0.1'
```

The API does not load `.env` automatically and has no credentials or cloud
configuration.

## Reliability and security controls

- Pydantic validates request bodies, pagination, filters, view names, and limits.
- Object lists return at most 100 rows; curves return at most 5,000 points.
- Frozen test identities are removed from the service catalog and rejected by
  object and inference endpoints.
- Curve and inference endpoints consume only local cached artifacts.
- Internal filesystem paths and executable locations are recursively removed.
- Errors have stable codes and never return tracebacks or local paths.
- CORS origins and host headers are restricted for local development.
- Responses include request IDs, processing time, and `Cache-Control: no-store`.
- Kepler and TESS remain distinct resources; no cross-mission training occurs.

## Testing protocol

`tests/test_scientific_api.py` validates endpoint contracts, OpenAPI methods,
pagination, validation failures, CORS, path sanitization, frozen-test rejection,
real ANN-01/ANN-02 inference, cached curves, experiment artifacts, persisted
explanations, TESS isolation, and absence of training/download calls.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

Tests patch acquisition and training entry points to fail if an API read invokes
them. No test downloads scientific data, trains a model, activates a checkpoint,
or evaluates the frozen test partition.

## Known limitations

- The service is local and single-process; authentication, rate limiting, public
  TLS, deployment manifests, and multi-user authorization are intentionally out
  of scope.
- Model files and astronomical caches are not committed to Git, so a fresh clone
  must restore the documented local artifacts before model endpoints are ready.
- ANN-02 results remain development validation results on a biased Q3 cohort.
- Only three representative local explanations were persisted in Stage 07.
- The TESS endpoint exposes one isolated compatibility pilot, not a TESS model or
  cross-mission validation result.
- JSON light curves are downsampled for browser transport; bulk scientific export
  is not part of this API stage.
