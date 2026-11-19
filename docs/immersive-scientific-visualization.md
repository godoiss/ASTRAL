# ASTRAL — Immersive Scientific Visualization

## Outcome

ASTRAL now has a publication-oriented Next.js interface under `web/`. The Atlas
design system is no longer presented as a conventional administrative dashboard:
the primary interaction is visual exploration, while tables remain available as
secondary audit instruments. Streamlit is unchanged and remains an internal
engineering laboratory.

The client contains no scientific implementation. It consumes the versioned
FastAPI boundary, which remains responsible for catalog filtering, frozen-test
exclusion, light-curve serialization, model loading, inference, explanations,
campaign execution, and artifact provenance.

## Product routes

- `/discover` — interactive equatorial stellar atlas using real KOI `ra` and
  `dec` fields, object selection, protected expedition planning, real job
  progress, persisted campaign summaries, and a secondary campaign ledger.
- `/observatory` — raw and detrended luminosity, phase-folded and local transit
  views, catalog context, ANN-01/ANN-02 readings, and an explicitly illustrative
  orbital projection whose numeric labels come from catalog parameters.
- `/candidates` — investigative dossiers for unassociated BLS screens, with
  luminosity, periodogram, phase fold, coverage, eclipsing-binary diagnostics,
  alternative hypotheses, limitations, and explicit inconclusive status.
- `/neural` — actual ANN-01 weights and retained neuron activations, ANN-02
  PyTorch module topology, checkpoint parameter distributions, real feature
  maps, saliency, occlusion regions, and local ANN-01 contributions.
- `/research` — persisted experiment chronology, native learning histories,
  like-for-like validation metrics, model agreement/divergence space, protocol
  boundaries, and reproducibility register.

## Scientific API additions

The API remains backward compatible and adds:

- optional `ra_deg` and `dec_deg` fields to object summaries and details, read
  from the NASA KOI cumulative snapshot;
- `GET /api/v1/models/{model_id}/visualization?object_id=...`, returning typed,
  bounded, real model parameters and inference activations;
- FastAPI `/` and `/app/` redirects to the configured publication client through
  `ASTRAL_FRONTEND_URL` (default `http://127.0.0.1:3000`).

ANN-01 visualization values come from the persisted Stage 04 checkpoint and a
retained forward pass for the selected object. ANN-02 weights come from the
selected `seed_stability` checkpoint, seed 97, and feature maps are captured from
that object's real PyTorch forward pass. No neural state is simulated.

## Visualization decisions

The sky atlas uses an efficient two-dimensional Canvas projection rather than a
permanent Three.js scene. The available KOI field is small and the primary task
is coordinate selection; Canvas offers clearer equatorial axes, lower CPU/GPU
cost, pointer zoom/pan, and an accessible catalog fallback. Real RA/Dec values are
shown in degrees and the projection is identified as equatorial catalog data.

The atlas camera is fitted to the actual Kepler catalog extent instead of the
entire celestial sphere, preventing the narrow Kepler field from collapsing into
an unreadable cluster. Its graticule, boundary, density glow, and optional screen-
proximity mesh are all derived from displayed catalog coordinates. The mesh is
explicitly labeled as a visual index and never as a physical stellar association.
Confirmed, candidate, and false-positive records use different glyph shapes in
addition to color. Zoom is pointer-anchored, pan uses pointer capture, and the
selected object can be focused without altering its catalog coordinates.

Plotly's basic distribution is loaded only on pages that render scientific
graphs. ScatterGL is used for dense luminosity and phase measurements. Vector
SVG is used for ANN-01 topology and orbital context because both require sharp,
auditable labels and modest geometry rather than a decorative 3D scene.

The orbital diagram is explicitly schematic: period, radius, and semi-major-axis
labels are real when present, but inclination, projected phase, perspective, and
body scale are not claimed as measured geometry.

### Atlas experience refinement

The publication client uses a dedicated `experience.css` layer over the reusable
scientific primitives. This separates decorative atmosphere from scientific
encoding and introduces:

- a floating translucent mission rail and responsive mobile command bar;
- an editorial Newsreader display face paired with Manrope for instrument UI;
- a procedural star field and orbital geometry that never encode measurements;
- higher-contrast instrument surfaces, restrained spectral light, and clearer
  visual hierarchy for plots, dossiers, neural state, and experiment records;
- responsive Plotly, Canvas, and SVG containers without permanent GPU-heavy 3D
  rendering.

The experience layer changes presentation only. Catalog values, observation
arrays, inference outputs, explanations, and experiment artifacts continue to
come from the existing FastAPI contracts.

## Accessibility and performance

- keyboard-focus indicators and 44 px minimum interactive targets;
- semantic navigation, headings, labels, tables, and plot descriptions;
- disposition is communicated by text as well as color;
- responsive layouts at desktop, tablet, and 390 CSS px;
- reduced-motion support;
- Canvas pixel ratio capped at 2;
- no permanent 3D renderer or animation loop;
- Plotly imported dynamically and only in the browser;
- all pages statically prerendered by Next.js, with scientific data fetched at
  runtime from FastAPI.

## Local execution

Terminal 1:

```powershell
$env:ASTRAL_DISCOVERY_EXECUTION_ENABLED = "true"  # optional local write gate
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```powershell
cd web
npm install
npm run dev
```

Open `http://127.0.0.1:3000`. The frontend rewrite target may be changed with
`ASTRAL_API_ORIGIN`; the FastAPI redirect target may be changed with
`ASTRAL_FRONTEND_URL`.

## Integrity boundaries

- no frozen-test object is exposed;
- no model is trained or modified;
- browser discovery jobs remain cache-only and deployment-gated;
- no new observation download is triggered by the interface;
- neural scores are not calibrated probabilities;
- unassociated signals remain inconclusive and are not discovery claims;
- Kepler and TESS remain distinct;
- the original Streamlit Developer Lab is preserved but is not the publication
  product surface.

## Validation record

Validated locally on 2026-11-19 against the persisted scientific artifacts:

- Python suite: `83 passed`;
- focused scientific API suite: `21 passed`;
- Next.js component suite: `3 passed`;
- TypeScript: no type errors;
- Next.js production build: seven routes prerendered successfully;
- route smoke test: `/discover`, `/observatory`, `/candidates`, `/neural`, and
  `/research` returned HTTP 200;
- API proxy smoke test: real `K00753.01` coordinates and ANN-02 visualization
  tensors were returned through the Next.js origin;
- mobile audit: a real Chromium viewport at 390 x 844 reported equal viewport,
  document, and scroll widths, with responsive Plotly resizing enabled;
- dependency audit: zero known npm vulnerabilities.

Visual evidence is stored in `visualization/immersive-discover.png`,
`visualization/immersive-neural.png`, and `visualization/immersive-mobile.png`.
