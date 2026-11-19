# ASTRAL Stage 09 — Branding & Discovery-first UX

Status: **Atlas direction approved; superseded by the live Stage 10 integration**
Package version: `0.9.1-evaluation`

## 1. Product north star

**Find what hasn’t been found.**

ASTRAL should mature into an instrument for planning surveys of real public
observations, detecting transit-like signals, reviewing compatible neural
assessments, cross-checking catalogs, and prioritizing plausible candidates.
That ambition does not change the current scientific truth:

- a classifier score is neither a calibrated probability nor a discovery;
- a known KOI is not a newly detected object;
- a ranked signal is not a confirmed exoplanet;
- Kepler and TESS remain separate scientific contexts;
- the frozen test partition remains inaccessible;
- absent services are named as roadmap work, not visually simulated.

The intended primary journey is `Discover → Observatory → Candidates → Neural
Lab → Research`. A user can begin with a research question rather than already
knowing a KOI identifier.

## 2. Creative directions

All three directions use the same information architecture. The prototype
switches tokens live so reviewers can compare visual character without
confounding it with different layouts.

### Atlas — recommended

**Narrative.** A contemporary stellar atlas for planning and reviewing
scientific expeditions. Editorial typography, sparse meridians, open space, and
precise instrument labels balance wonder with restraint.

- Void `#070D17`, Ion `#70DBF7`, Signal `#EFB681`.
- Georgia/system serif display, Segoe UI interface, Consolas data.
- Illustrative constellation artwork is always explicitly marked as brand art.
- Compact star/orbit mark; open geometric icons; thin strokes.
- Strengths: discoverability, memorable editorial voice, long-session
  legibility, and strong separation of narrative from data.
- Risk: illustrative elements must never be mistaken for observations.

### Signal Room

**Narrative.** A spectral analysis workspace that foregrounds signal morphology
and neural interpretation.

- Violet `#D4B7FA`, Teal `#70C9C0`, near-black violet canvas.
- Denser grid language and rectangular instrument surfaces.
- Especially suitable for curves, feature maps, saliency, and XAI.
- Strengths: precise and technically expressive.
- Risk: can feel too specialized and intimidating as a default entry point.

### Quiet Orbit

**Narrative.** A calm academic observatory for sustained research sessions.

- Sage `#B5D3C6`, Warm `#D5A780`, dark mineral-green canvas.
- Subtle archival warmth and restrained surface elevation.
- Strengths: calm reading, report review, and reduced visual fatigue.
- Risk: less distinctive and less immediate as a portfolio identity.

| Criterion | Atlas | Signal Room | Quiet Orbit |
|---|---:|---:|---:|
| Originality | 5/5 | 4/5 | 4/5 |
| Immediate impact | 5/5 | 4/5 | 3/5 |
| Scientific legibility | 5/5 | 4/5 | 5/5 |
| Accessibility resilience | 5/5 | 4/5 | 5/5 |
| Discovery-first fit | 5/5 | 3/5 | 4/5 |

**Recommendation:** continue with Atlas. Signal Room can inform analytical
canvases; Quiet Orbit can inform long-form reports. They should remain token
influences rather than fragmenting the product into separate themes.

## 3. Identity

Use `ASTRAL` in uppercase and `Astronomical Neural Network` only as the formal
descriptor. The voice is precise, curious, understated, and exploratory. Favor
expedition, observation, signal, candidate, evidence, and research. Never use
“planet found” for a model result.

The proposed mark combines a central star with two observation/orbital paths.
It is original SVG geometry, works at 20 px, and avoids rockets, mission-patch
styling, and the NASA insignia. Decorative artwork may create a sense of
wonder, but its caption must say that it is illustrative.

## 4. Design foundations

The machine-readable source is
`design/stage09/design_tokens.json`.

### Typography

- Display: Georgia, with Times New Roman fallback; 48–76 px responsive.
- Interface: Segoe UI, Arial fallback; 16 px body.
- Data: Consolas, Cascadia Mono fallback, tabular numerals.
- UI labels are at least 14 px. Twelve-pixel text is limited to eyebrows,
  secondary metadata, and nonessential telemetry.
- Section headings are approximately 26–28 px; line length targets 45–75
  characters.

No font files are distributed. These system font stacks add no bundled font
license. Future production substitution requires a recorded license review.

### Space, radius, and grid

- Spacing: 4, 8, 12, 16, 24, 32, 48, 64 px.
- Radius: 6, 8, 10, 14, 20 px.
- Twelve-column desktop, six-column tablet, four-column mobile.
- 1,680 px content maximum; 252 px desktop navigation rail.
- 44 px minimum interactive target.
- Dense scientific tables scroll horizontally instead of hiding fields.

### Color semantics

Brand accents and scientific meaning are independent. Ion cyan and Signal
amber express hierarchy and interaction; they do not encode classification.
Confirmed, false-positive, and candidate states each use a dedicated color,
text label, and symbol. Plot colors identify signal representations, not NASA
catalog dispositions. A score must not inherit catalog-status color.

### Components

- Shell: `AppRail`, `ContextBar`, `DirectionStudy`, `ScientificBoundary`.
- Discovery: `ExpeditionPlanner`, `PlanReview`, `CampaignState`,
  `UnavailableService`.
- Astronomy: `ObjectRecord`, `CatalogDisposition`, `ObservationWorkspace`,
  `ProvenanceDrawer`, `LightCurvePlot`.
- Candidates: `CandidateRegistry`, `DetectionEvidence`, `RejectionReason`,
  `CatalogCrossCheck`, `VettingState`.
- Neural: `ModelSwitch`, `TopologySchematic`, `ScoreReadout`,
  `ExplanationMethod`, `FeatureMap`, `SensitivityRegion`.
- Research: `ComparableCohort`, `MetricRecord`, `ExperimentHistory`,
  `CheckpointIdentity`, `LimitationCallout`.
- Primitives: button, labeled field, select, table, badge, dialog, notice,
  skeleton, error, empty, and unavailable state.

### States

- Loading states identify the literal operation; they never cycle fake values.
- Empty states distinguish no result from unavailable capability.
- Unavailable states name the missing service and roadmap stage.
- Errors expose safe error codes and recovery guidance, not internal paths.
- Experimental TESS states retain mission identity and isolation.
- Frozen states explain why access is intentionally denied.

### Motion and accessibility

Motion may communicate navigation, focus, or arrival of real data. Do not
animate stars, coordinates, activations, weights, or signals in ways that imply
measurement. Honor `prefers-reduced-motion`. Target WCAG 2.2 AA with visible
focus, semantic landmarks, keyboard-operable controls, text-plus-symbol status,
table captions, and nonvisual summaries for future scientific plots.

## 5. Screen contracts

### Discover

The default landing explains why ASTRAL exists before requesting identifiers.
Its planner captures mission, sample, strategy, and detection method. “Review
expedition” opens a summary and explicitly performs no search, download,
training, or persistence. The Discovery Engine is labeled unavailable and
assigned to Stage 10.

Future campaign execution must expose progress, observation coverage, method
configuration, rejection reasons, catalog cross-checks, reproducible artifacts,
and failure recovery.

### Observatory

The design artifact includes three authorized, audited records:
`K00753.01`, `K00097.01`, and `K07977.01`. Their catalog dispositions and stored
Stage 08 ANN scores are displayed as separate concepts. The light-curve canvas
is an unavailable state because the prototype intentionally has no API
connection and no fictional series.

Production dependencies: object search/detail, observation metadata, and
processed light-curve endpoints.

### Candidates

The default state states: **No newly found candidates have been demonstrated.**
Future work needs campaign, detection, catalog, candidate-registry, and vetting
services before it can truthfully populate this area.

### Neural Lab

The ANN-01/ANN-02 switch changes an explicitly conceptual schematic. Nodes
describe roles on hover or keyboard focus. The map is not recovered topology
and contains no live activations, gradients, or weights. Audited object scores
come from the Stage 08 record.

Implemented explanation approaches are named accurately: ANN-01 permutation
importance and local feature differences; ANN-02 gradient × input, occlusion
sensitivity, and feature maps. Production dependencies include model detail,
explanation, and inference endpoints.

### Research

The shared 47-object validation cohort displays ANN-01 accuracy `0.8723` and
balanced accuracy `0.8696`; ANN-02 accuracy `0.7660` and balanced accuracy
`0.7681`. ANN-01 validation ROC-AUC `1.0000` is explicitly non-definitive. No
ensemble exists and the frozen test remains isolated.

## 6. UX architecture and flows

Primary navigation contains exactly five areas:

1. Discover — question, expedition plan, campaign contract.
2. Observatory — known objects, observations, catalog truth, provenance.
3. Candidates — future detections, cross-checks, vetting, rejection evidence.
4. Neural Lab — models, scores, explanations, conceptual structure.
5. Research — comparable cohorts, metrics, histories, experiments, limits.

Key future flows:

- Plan survey: Discover → review contract → future campaign execution →
  Observatory.
- Audit signal: Observatory → observation evidence → catalog cross-check →
  Candidates.
- Question score: Observatory/Candidates → Neural Lab → explanation → model and
  experiment provenance.
- Review rigor: Research → comparable cohort → model history → limitations.

## 7. Planned Next.js structure

This is a Stage 13 implementation plan, not current code:

```text
app/
  discover/
  observatory/
  observatory/[objectId]/
  candidates/
  candidates/[candidateId]/
  neural-lab/
  neural-lab/[modelId]/
  research/
components/
  shell/          AppRail ContextBar ScientificBoundary
  discovery/      ExpeditionPlanner PlanReview CampaignState
  astronomy/      ObjectRecord ObservationWorkspace ProvenanceDrawer
  candidates/     DetectionEvidence CatalogCrossCheck VettingState
  neural/         ModelSwitch TopologySchematic ScoreReadout
  explainability/ FeatureImportance SensitivityRegion FeatureMap
  research/       ComparableCohort MetricRecord ExperimentHistory
  primitives/     Button Field Select Table Dialog StatePanel
lib/
  api/            generated Stage 08 client; future discovery client
  contracts/      response adapters and scientific boundaries
  formatting/     units precision identifiers dates
styles/
  design-tokens.css typography.css scientific-colors.css motion.css
```

Stage 10 is reserved for the Discovery Engine, Stage 11 for Candidate
Intelligence, Stage 12 for Discovery API/research infrastructure, Stage 13 for
Next.js, Stage 14 for advanced visualizations, and Stage 15 for release after
audits.

## 8. Assets and licensing

- Preview mark and inline icons: original project SVG geometry; no external icon
  package bundled.
- Georgia, Segoe UI, and Consolas: referenced as operating-system font stacks;
  no font binaries redistributed. Production availability and substitution need
  review on each target OS.
- No external photography is bundled. Future mission imagery must store source,
  mission/creator, asset identifier, usage note, and credit.
- NASA names and datasets may be cited factually; do not imitate NASA insignia
  or imply endorsement.

## 9. Implementation and review report

Implemented in this package:

- Atlas discovery-first high-fidelity prototype;
- live Signal Room and Quiet Orbit token comparison;
- five-view navigation and responsive shell;
- non-executing expedition review dialog;
- honest missing Discovery Engine, candidate-registry, and light-curve states;
- three audited object snapshots and shared-cohort research diagnostics;
- model/object switches, keyboard-focusable conceptual nodes, and real
  explanation-method descriptions;
- machine-readable tokens and planned future component architecture.

The artifact makes no API call, external download, model inference, training,
or scientific computation. It does not modify models, datasets, checkpoints,
backend, Streamlit, or frozen-test artifacts.

Atlas was subsequently approved as the product direction. Stage 10 connects the
approved buildless client to typed FastAPI services while preserving this
document as the historical design specification. See
`docs/stage10-atlas-product-interface.md` for the live integration and its
scientific and deployment boundaries.
