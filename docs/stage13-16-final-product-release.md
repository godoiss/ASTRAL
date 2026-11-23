# ASTRAL Stages 13–16 — Final Product and Release Report

**Status:** implemented and locally validated  
**Application surface:** Next.js Atlas (`web/`)  
**Scientific service:** FastAPI (`src/api/`)  
**Validation date:** 2026-11-23  
**Release profile:** public read-only by default

## 1. Scope and audited baseline

This delivery consolidates the existing scientific stack instead of replacing it. The retained evidence baseline is:

- `stage10-kepler-reference-final`: 12 Kepler host stars, eight screening detections, three local KOI-period associations, five unassociated/inconclusive screens, and zero claimed discoveries;
- eight persisted Stage 11 Candidate Intelligence dossiers;
- ANN-01 NumPy and ANN-02 PyTorch checkpoints and their existing scientific-validation records;
- Stage 12 durable campaign execution, SQLite state, append-only events, cancellation, resume, and recovery;
- a frozen test partition that remains outside campaign execution, product inference selection, and this release work.

No model was retrained, no scientific result was rewritten, and no observation was fabricated during Stages 13–16.

## 2. Final architecture

```text
Browser
  └─ Next.js Atlas publication client (:3000)
       ├─ Discover / persistent Mission Control
       ├─ Observatory / real cached Kepler signals
       ├─ Candidates / eight scientific dossiers
       ├─ Neural Lab / retained ANN states
       └─ Research / campaigns and experiments
            │ same-origin /api proxy
            ▼
       FastAPI scientific boundary (:8000)
            ├─ typed read services
            ├─ inference and persisted explanations
            ├─ protected scientific mutations
            └─ durable campaign coordinator
                 ├─ SQLite operational state
                 └─ versioned JSON/NPZ evidence
```

The frontend contains presentation and interaction logic only. Detection, signal processing, inference, provenance, campaign state transitions, and persistence remain in the Python scientific services.

## 3. Stage 13 — final product experience

### Public landing page

The root route is now a real product entry point rather than a redirect. It communicates the platform's purpose, its three scientific engines, the five-instrument investigation path, persisted reference outcomes, and the scientific boundary between a screening signal and a discovery. Open Graph, Twitter metadata, canonical metadata, a generated icon, sitemap, and robots policy are included.

### Connected scientific areas

- **Discover** retains the real-coordinate Kepler atlas, selection, filtering, plan validation, protected execution, progress, events, cancellation/resume, and persisted results.
- **Observatory** retains raw and processed luminosity, phase-folded and transit-local views, physical/catalog context, orbital representation, and real ANN-01 inference where input compatibility exists.
- **Candidates** exposes all eight Stage 11 dossiers, measured diagnostics, official catalog cross-checks, alternatives, uncertainty, provenance, export, and a protected human-review capability.
- **Neural Lab** exposes real ANN-01 weights, biases, activations and feature importance plus ANN-02 architecture, retained filters, feature maps, signal sensitivity and local explanations. Unavailable state is labeled rather than synthesized.
- **Research** combines model history, comparable-cohort metrics, experiment evidence, immutable campaign artifacts, event histories, and reproducibility exports.

All five areas remain linked by real object, candidate, and campaign identifiers.

## 4. Stage 14 — scientific visualization

Atlas remains a purpose-built scientific interface rather than a generic administrative dashboard. The implementation preserves:

- stellar cartography grounded in catalog coordinates where those coordinates are available;
- interactive Plotly light curves, periodograms, phase-folded views, model metrics, filters, and feature maps;
- accessible text/table alternatives for detailed inspection and audit;
- explicit labels for schematic orbital or neural representations;
- restrained depth, motion, cyan/violet focus, scientific typography, and non-color status cues.

Plotly is loaded only when a scientific figure approaches the viewport. This reduces the initial JavaScript and memory cost without replacing real measurements with placeholders.

## 5. Stage 15 — mobile experience

The mobile shell uses a one-handed five-instrument dock, a focusable navigation drawer, a dismissible scrim, Escape handling, body-scroll locking, safe-area insets, touch-sized controls, and route-aware labels. Charts remain responsive and their containers use explicit touch behavior to avoid accidental page overflow.

Automated browser validation covered all six public routes at 320, 360, 390, 430, 768, 1024, 1440, and 1920 pixels. That is 48 route/viewport checks with no horizontal overflow. Touch emulation was used through 768 px. Physical-device testing was not available and is not claimed.

The application honors `prefers-reduced-motion`. Heavy scientific plots are lazy-loaded and do not initialize off-screen.

## 6. Stage 16 — security and production readiness

### Default safety boundary

`ASTRAL_DEPLOYMENT_MODE=public_read_only` is the publication default. In this profile:

- campaign creation, start, cancel and resume are rejected;
- human-review writes are rejected;
- MAST downloads remain disabled;
- retained scientific evidence, curves, dossiers, models, experiments and explanations remain readable;
- inference uses existing compatible data and retained checkpoints only; it does not train a model or alter a dataset.

Controlled local mutation requires all relevant feature flags plus either a loopback client or a valid `X-ASTRAL-Operation-Key`. A public read-only profile cannot be overridden by the key.

### Additional controls

- restrictive CORS and `TrustedHostMiddleware`;
- Pydantic bounds for page sizes, curve points and campaign limits;
- a 1 MiB request-body ceiling by default;
- bounded in-process mutation rate limiting;
- CSP and related browser security headers from Next.js;
- `Permissions-Policy`, cross-origin opener policy and HTTPS-only HSTS from FastAPI;
- opaque public errors without internal filesystem paths;
- secrets supplied through environment variables, never committed;
- non-root runtime users in both Docker images.

The in-process limiter is appropriate for this single-instance reference deployment. A multi-instance public deployment must add a shared upstream limiter and identity-aware authorization.

## 7. Persistence, backup, and recovery

The Docker reference profile is intentionally read-only and binds services to loopback. A real host must mount durable storage for:

- `data/research/` — SQLite operational state and journals;
- `experiments/` — scientific reports, campaign artifacts, manifests and exports;
- the required local model/checkpoint and observation caches, according to the deployment policy.

Recommended backup procedure:

1. stop campaign mutation or run in public read-only mode;
2. create a consistent SQLite backup with the SQLite backup API or `.backup`, not a live file copy;
3. copy `experiments/` and required immutable scientific artifacts with checksums;
4. record the Git commit, API version and environment profile alongside the backup;
5. restore into a staging instance and verify `/health`, `/api/v1/info`, campaign event history, and dossier exports before promotion.

SQLite is sufficient for the bounded single-worker deployment. Horizontal workers require a shared transactional database and distributed task ownership; that migration is deliberately not claimed here.

## 8. Reproducible execution

### Safe local production build

```powershell
$env:ASTRAL_DEPLOYMENT_MODE = "public_read_only"
$env:ASTRAL_DISCOVERY_EXECUTION_ENABLED = "false"
$env:ASTRAL_DISCOVERY_DOWNLOADS_ENABLED = "false"
$env:ASTRAL_CANDIDATE_REVIEWS_ENABLED = "false"
\.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000

cd web
$env:ASTRAL_API_ORIGIN = "http://127.0.0.1:8000"
npm ci
npm run build
npm run start
```

Open `http://127.0.0.1:3000`. OpenAPI remains available locally at `http://127.0.0.1:8000/docs`.

### Containers

```powershell
docker compose config
docker compose up --build
```

The supplied Compose profile is a local release reference, not an internet edge. For a public host, terminate TLS at a maintained reverse proxy, set the real origin/hosts, mount durable volumes, provide centralized logs and backups, and place authenticated administrative execution on a separate protected surface.

## 9. Verification evidence

Final checks completed in the local CPU environment:

| Gate | Result |
|---|---:|
| Python/FastAPI suite | 94 passed |
| Clean-checkout CI scientific suite | 37 passed |
| TypeScript typecheck | passed |
| ESLint | passed, zero warnings |
| Vitest components | 6 passed |
| Next.js production build | passed |
| Responsive browser checks | 48 passed |
| Browser interactions, controlled profile | 7 passed |
| Browser console errors | 0 |
| `npm audit --omit=dev --audit-level=high` | 0 vulnerabilities |
| `pip check` | no broken requirements |
| `pip-audit -r requirements.txt` | no known vulnerabilities |
| `docker compose config` | valid |

The full 94-test scientific suite is intentionally a local, artifact-backed gate because raw NASA data, observation caches and trained checkpoints are not committed. GitHub Actions runs a 37-test portable suite on a clean checkout, covering neural mathematics, preprocessing, acquisition guards, scientific-validation logic, environment contracts, persisted candidate dossiers and training primitives. This distinction prevents either uploading large scientific data or making data-dependent tests pass with fabricated fixtures.

The controlled browser pass started and observed a real cache-only campaign (`research-20261123T122834Z-18273fe4`): 5/5 targets completed, no network acquisition, persisted evidence, and a completed terminal state. The public read-only pass separately verifies that the publication profile exposes evidence without exposing mutation.

## 10. Scientific and engineering limitations

- Reference detections are screening results, not new confirmed exoplanets.
- ANN scores are not calibrated discovery probabilities.
- ANN validation cohorts are limited and are not definitive performance estimates on novel mission data.
- The Stage 10 reference campaign is intentionally small and partially cache constrained.
- TESS remains an isolated experiment and is never merged automatically with Kepler training evidence.
- Browser checks used desktop browser emulation; no physical mobile hardware result is claimed.
- Plotly basic distribution is used; advanced rendering remains bounded by client memory and CPU.
- Public authentication, shared rate limiting, object storage, managed database, domain and TLS depend on external infrastructure not present in this workspace.

## 11. Publication status

The source, production builds, local safe profile, containers, CI, security boundary, documentation and browser validation are complete. No public URL was created because the workspace has no selected hosting account, domain, TLS configuration, persistent-volume target or deployment credentials. Publishing without those decisions would be insecure and could create cost.

The exact remaining release action is to select and authorize infrastructure, provision persistent storage and HTTPS, configure the environment from `.env.example`, deploy both images, and rerun the same smoke/security checks against the public URL.

