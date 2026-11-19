# ASTRAL Stage 11 — Candidate Intelligence & Scientific Verification

## Status and scientific boundary

Stage 11 converts the eight real screening detections from the immutable `stage10-kepler-reference-final` campaign into reproducible scientific-review dossiers. It does not confirm planets, establish novelty, change NASA catalog dispositions, retrain a model, access the frozen test cohort, or interpret a neural score as scientific evidence.

The review-priority score orders human effort. It is a transparent composite of signal quality, observational retention, repeatability, event and phase coverage, processing robustness, transit consistency, false-positive evidence, unresolved catalog value, and a period-search-boundary penalty. It is **not** a planet probability, confidence interval, or false-alarm probability.

## Reproducible architecture

- `src/candidate_intelligence/catalogs.py` performs bounded, column-explicit NASA Exoplanet Archive TAP queries.
- `src/candidate_intelligence/engine.py` joins official catalog records only after detection, computes review evidence, and writes immutable JSON/Markdown dossiers.
- `src/candidate_intelligence/cli.py` rebuilds the persisted Stage 11 evidence from the Stage 10 reference campaign.
- `src/api/candidate_service.py` reads the local artifacts and stores only explicit human review decisions.
- `src/api/routers/candidate_intelligence.py` exposes typed, versioned API resources without scientific downloads or model execution.
- `web/app/candidates/page.tsx` is the publication-interface investigation center.

The API does not query NASA at request time. Catalog acquisition is an explicit CLI workflow, while normal product navigation is served entirely from persisted, auditable artifacts.

## Official catalog verification

The bounded verification run queried the official NASA Exoplanet Archive `cumulative` KOI table and `q1_q17_dr25_tce` table for exactly eight KIC identifiers. It returned eight KOI records and nine DR25 TCE records. Query text, source URLs, retrieval time, selected fields, and returned records are preserved in `experiments/stage11/catalog_verification.json`.

The Exoplanet Archive documents an eclipsing-binary catalog, but no compatible table was available through the TAP schema used by this workflow. Stage 11 therefore records that source as `not_machine_queried`, with the reason, rather than treating the absence of a query result as evidence of no match.

Official references:

- [Kepler Objects of Interest documentation](https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html)
- [KOI column definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html)
- [Kepler Threshold-Crossing Events](https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_TCEs.html)
- [Kepler eclipsing-binary documentation](https://exoplanetarchive.ipac.caltech.edu/docs/eclbin.html)
- [Programmatic API documentation](https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html)

## Real investigation results

| KIC | BLS period (days) | Review priority | Catalog relationship | Scientific state |
|---:|---:|---:|---|---|
| 10811496 | 19.898822 | 91.547 | KOI/TCE period match near 19.8991 d | Catalog associated |
| 10351947 | 21.446192 | 91.185 | Host KOI/TCE near 260.45 d; no simple period relation | Unassociated, inconclusive |
| 9140402 | 2.495268 | 84.928 | KOI/TCE period match near 2.49417 d | Catalog associated |
| 8527297 | 21.670882 | 84.386 | Host KOI and three TCEs near 0.587 d; no simple relation | Unassociated, inconclusive |
| 11340218 | 5.002919 | 83.664 | Host KOI near 111.149 d; no DR25 TCE | Unassociated, inconclusive |
| 8626325 | 10.166949 | 82.624 | Host KOI/TCE near 16.4491 d; no simple relation | Unassociated, inconclusive |
| 9772531 | 10.396915 | 65.226 | One-third relation to KOI/TCE near 31.2017 d | Catalog associated; EB-compatible |
| 10155029 | 30.000000 | 63.817 | Host KOI/TCE near 2.13165 d; BLS peak on search boundary | Unassociated, inconclusive |

The run produced three catalog-associated signals, five unassociated but inconclusive signals, one eclipsing-binary-compatible diagnostic, and zero confirmed discoveries. All eight returned KOI records have catalog disposition `CANDIDATE`; no official KOI false-positive flags were set in the selected records. Empty automated TCE classification values are retained as unavailable, not interpreted.

## Evidence and model compatibility

Each dossier contains the independent BLS measurement, observational coverage, event depths, alias candidates, odd/even comparison, phase-0.5 secondary-event diagnostic, catalog period relationships, favorable/contrary/inconclusive evidence, instrumental hypotheses, limitations, provenance hashes, and human-review history.

ANN-01 was not invoked because an arbitrary detected event lacks independently fitted planet radius, equilibrium temperature, and compatible catalog model SNR. Substituting the selection KOI values would classify the catalog object rather than the newly measured event. ANN-02 was not invoked because Discovery Engine arrays are not the validated catalog-centered ANN-02 representation and no out-of-distribution validation supports that use. Both decisions are persisted per dossier.

## Publication interface and workflow

The Atlas application exposes all eight signals at `/candidates`. The primary visual workflow is:

`Discover → Candidate Intelligence → Observatory → Research`

The investigation center provides real time series, BLS periodogram, phase fold, event-depth sequence, aliases, evidence matrix, official KOI/TCE records, transparent ranking components, neural compatibility, exports, and an append-only human review form. Observatory handoff keeps the independently detected BLS period visibly separate from catalog-centered ephemerides. Research indexes the Stage 11 archive and review state.

API resources:

- `GET /api/v1/candidate-intelligence`
- `GET /api/v1/candidate-intelligence/{kic}`
- `GET /api/v1/candidate-intelligence/{kic}/export`
- `GET /api/v1/candidate-intelligence/{kic}/report`
- `GET /api/v1/candidate-intelligence/{kic}/reviews`
- `POST /api/v1/candidate-intelligence/{kic}/reviews`

## Reproduction

```powershell
# Explicit scientific rebuild; this performs the bounded official catalog query.
.\.venv\Scripts\python.exe -m src.candidate_intelligence.cli

# Product services; normal navigation remains local and performs no NASA query.
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
cd web
npm run dev
```

Open `http://127.0.0.1:3000/candidates`.

## Limitations

- Only Quarter 3 long-cadence observations are analyzed.
- The Stage 10 period search is limited to 1–30 days; one signal lies exactly on that boundary.
- No multi-quarter repeatability, centroid motion, pixel-level contamination, nearby-source, spectroscopy, or independent stellar validation was performed.
- The BLS threshold has no calibrated false-alarm probability from bootstrap or injection/recovery experiments.
- A simple period relationship does not prove that two events share the same astrophysical origin.
- No catalog relationship does not establish novelty.
- The current eclipsing-binary source status is explicitly unresolved at machine-query level.
- Human decisions are local research annotations, not NASA classifications.

Stage 12 was not started.
