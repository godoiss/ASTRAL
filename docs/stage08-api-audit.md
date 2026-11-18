# Stage 08 API Audit

Audit date: 2026-11-18

## Conclusion

The 18 documented operations were exercised against real local artifacts and
returned successful responses. ANN-01 and ANN-02 inference was verified only for
catalog objects available to the API. The frozen test partition remained
inaccessible, no training or acquisition function was invoked, and the protected
datasets and checkpoints retained their original SHA-256 hashes and timestamps.

The audit identified one integration risk: most OpenAPI response bodies were
previously represented by generic dictionaries. The HTTP boundary now uses
strict, resource-specific Pydantic models and stable operation identifiers.

## Endpoint verification

| Area | Operations exercised | Result |
|---|---:|---|
| Health and identity | 2 | 200 |
| Datasets and observatory | 2 | 200 |
| Objects, observations, and curves | 4 | 200 |
| Models, histories, and inference | 4 | 200 |
| Comparison and explainability | 3 | 200 |
| Experiments and isolated TESS pilot | 3 | 200 |
| **Total** | **18** | **All successful** |

The requests covered pagination, class and text filters, all four light-curve
views, bounded saliency, experiment detail, and the isolated TESS resource.
Invalid offsets, limits, filters, views, identifiers, unknown routes, unknown
models, and unsupported HTTP methods returned structured 4xx responses.

## Real inference checks

Three accessible KOIs representing `CANDIDATE`, `CONFIRMED`, and
`FALSE POSITIVE` catalog dispositions were passed to both models. The following
outputs are uncalibrated model scores and are recorded only as execution checks:

| Object | Catalog disposition | ANN-01 score | ANN-02 score |
|---|---|---:|---:|
| K00753.01 | CANDIDATE | 0.4147668955 | 0.7588245273 |
| K00097.01 | CONFIRMED | 0.8280777101 | 0.6251904964 |
| K07977.01 | FALSE POSITIVE | 6.4357056e-25 | 0.0589553453 |

These checks do not change NASA dispositions and are not scientific
confirmations. ANN-02 remained pinned to experiment
`ann02-20261118T180835Z`, checkpoint `seed_stability`, seed 97.

## Scientific integrity

- The catalog exposed by the service excludes every row whose split is `test`.
- A frozen-test identity was rejected with HTTP 404 by object lookup, ANN-01
  inference, and ANN-02 inference.
- The OpenAPI document contains no training or download operation. Its only POST
  is read-only inference.
- Automated tests replace ANN-01 training, ANN-02 training, and MAST download
  entry points with failing sentinels and verify that API calls never invoke
  them.
- Kepler and TESS remain separate resources. No ensemble is computed.

Protected artifact hashes were identical before and after endpoint and test
execution:

| Artifact | SHA-256 |
|---|---|
| `data/raw/kepler_koi.csv` | `A85348F20082D24FB14D4A05A9A44C644B801998FAD7CEC36F8980EB1F0A69AB` |
| `data/ann02/ann02_dataset.npz` | `6992AE2A0A1652C7BA4E6C021EBA92CC1644609EDB6BB21CA70145BA8475F14B` |
| `data/ann02/manifest.json` | `F50E1E9A9E03830F6304915D222394CDD4F6C1434E2FDF2A4EAD68FFB19ABFFA` |
| ANN-01 selected model | `B3E333F614735710EB66C7B1F034249FE38D5593CCAF8CC70AD5A7BB9B41FFA0` |
| ANN-02 Stage 07 selected model | `774DA392C1688AA1D9608BA03C965914BCBD4F6E34CD2BCC8A2F871C640783C2` |

## Corrections made

- Replaced generic `dict[str, Any]` response envelopes with strict typed schemas
  for catalogs, curves, models, metrics, inference, explanations, experiments,
  and TESS compatibility data.
- Added unique stable `operationId` values to all 18 operations for predictable
  future TypeScript client generation.
- Documented the uniform error envelope for HTTP 400, 404, 409, 422, 500, and
  503 in OpenAPI.
- Added length and character validation to object, model, and experiment path
  parameters.
- Normalized experiment-detail responses instead of exposing arbitrary persisted
  JSON documents.
- Cached the expensive catalog checksum per process and indexed complete
  observation triples. The observation index refreshes when cache directories
  change, preserving compatibility with the Developer Lab.
- Extended integration tests to exercise all 18 operations and both-model frozen
  test rejection.

Some intentionally variable scientific structures remain dictionaries inside
otherwise typed resources: persisted architecture/configuration metadata,
observation provenance, and the experimental TESS common contract. Their outer
contracts and stable fields are fully typed.

## Validation evidence

```text
API tests:       15 passed
Complete suite:  65 passed
Dependency check: No broken requirements found
Live API:        HTTP 200 on 127.0.0.1:8000/health
OpenAPI:         18 operations, all with concrete 200 response schemas
Developer Lab:   HTTP 200 on 127.0.0.1:8501
```

Two non-blocking dependency warnings remain: Lightkurve reports that optional
`oktopus` support is absent, and FastAPI's compatibility TestClient reports that
its current `httpx` adapter is deprecated in favor of `httpx2`. Neither affects
the audited API paths or the current test results.

## Residual limitations

- The server is local and single-process. Authentication, TLS, public deployment,
  distributed rate limiting, and multi-user authorization remain out of scope.
- Request result volumes are bounded, but there is no reverse-proxy request-size
  or rate policy because the service is loopback-only.
- Model and observation artifacts must exist locally; they are intentionally not
  embedded in API responses or downloaded on demand.
- Validation metrics remain development diagnostics on a limited, biased cohort,
  not final unseen-test estimates.
- Only the explanations persisted during Stage 07 are available.

