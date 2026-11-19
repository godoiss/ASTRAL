# ASTRAL Stage 10 — Discovery Engine

## Status and scientific boundary

Stage 10 adds an independent, reproducible search for periodic transit-like signals in real Kepler light curves. It is an exploratory screening instrument, not a discovery-validation pipeline. ASTRAL does not call ANN-01 or ANN-02 to confirm a signal, does not retrain either network, does not change a NASA disposition, and never interprets an unmatched peak as a new exoplanet.

The reference campaign completed on 2026-11-19 with 12 unlabeled host stars. All labeled Stage 04 train, validation, and frozen-test hosts were excluded before acquisition. The measured target intersection with the complete protected host set was empty. Only host identifiers from the split manifest were read to enforce that exclusion; no frozen-test feature, observation, label, prediction, or metric entered the campaign.

## Architecture

The implementation is separated into independently testable modules:

- `src/discovery/config.py`: validated search, acquisition, and resource limits.
- `src/discovery/acquisition.py`: KIC-first local-cache access and explicit, bounded MAST acquisition.
- `src/discovery/processing.py`: quality filtering, segment normalization, robust-noise estimation, and catalog-independent detrending.
- `src/discovery/detection.py`: Astropy Box Least Squares search and transparent diagnostics.
- `src/discovery/catalog.py`: protected target selection and post-detection KOI/TCE-linked cross-checking.
- `src/discovery/campaign.py`: immutable campaign directories, progress, hashes, failures, and machine-readable results.
- `src/discovery/cli.py`: planning, execution, and artifact inspection.
- `src/lab/discovery.py`: real Streamlit/Plotly campaign controls and evidence views.

The detector accepts a processed light-curve record and a validated detection configuration. Acquisition, processing, detection, and catalog interpretation remain separate, so a later Transit Least Squares adapter can reuse the campaign and evidence layers without changing the BLS implementation or scientific records.

## Data independence and target protection

The deterministic selection strategy is `unlabeled_candidates_cache_first`:

1. The local KOI snapshot supplies unique KIC identifiers whose current catalog disposition is `CANDIDATE`.
2. Every KIC present in the Stage 04 `train`, `validation`, or `test` split is excluded.
3. Targets retain label `-1`; candidates are never converted into labeled examples.
4. Cached observations are preferred, and remaining eligible stars are deterministically shuffled with seed `10103`.
5. Only KIC and raw observed cadences enter processing and BLS. Catalog period, duration, epoch, and disposition do not enter the detector.
6. KOI period and KOI-linked TCE fields are read only after a period has been measured.

The campaign report records the SHA-256 of both protection inputs:

- KOI snapshot: `a85348f20082d24fb14d4a05a9a44c644b801998fad7cec36f8980eb1f0a69ab`
- Stage 04 split manifest: `467011389b02f4bd034bec88c731454b0deb7dcff43dec25fe03dfeee0cfc445`

Existing Observatory products are reused only from their original raw arrays. Catalog-centered processed products and transit masks are deliberately not consumed.

## Signal processing and BLS method

The processor removes non-finite measurements and Kepler quality flags selected by bitmask `1130799`. Each observed segment is divided by its median. A centered two-day rolling median estimates the slow trend at existing cadences only; temporal gaps are never filled with synthetic signal. Flux errors are conservatively floored by a robust residual-noise estimate,

`sigma_robust = 1.4826 * median(|x - median(x)|)`.

Astropy `BoxLeastSquares` evaluates 4,000 uniformly spaced frequencies corresponding to periods from 1 to 30 days and trial durations of 1.5, 2.5, 4, and 6 hours. The selected grid point maximizes the BLS depth-SNR objective. The screen then requires positive depth, depth SNR at least 6, at least two individually covered events, and event coverage of at least 0.5.

For every target the engine persists:

- measured period, transit epoch, duration, depth, depth error, and screening SNR;
- per-event depths, expected/observed events, temporal event coverage, and phase-bin coverage;
- competing local BLS peaks and simple half/third/double/triple-period aliases;
- odd/even depth and phase-0.5 secondary-eclipse diagnostics;
- an eclipsing-binary compatibility flag;
- grid/noise uncertainty approximations and their limitations;
- raw provenance, processed arrays, periodogram, phase-folded arrays, hashes, and post-detection catalog associations.

The ranking score is a transparent diagnostic derived from depth SNR, coverage, alias penalty, and eclipsing-binary penalty. It is not a calibrated probability, false-alarm probability, or scientific confidence.

## Resource and failure controls

Network access defaults to disabled. A campaign requires explicit opt-in, downloads at most the configured number of stars, selects at most one Quarter 3 long-cadence FITS product per star, applies a per-star timeout/retry policy, and stops new acquisition at the configured local discovery-cache limit. Raw FITS files remain in the existing local Lightkurve cache; compact discovery arrays and provenance remain in `data/discovery/`, which is excluded from Git except for directory placeholders.

The first 10-target expedition recorded two honest MAST exclusions (`KIC 11183752` and `KIC 11547869`, no Kepler light curve found) and eight completed analyses. A second bounded campaign reused those eight local caches and allowed only four additional downloads. It completed 12/12 analyses with no failures. Both campaign reports are preserved.

## Controlled algorithm tests

Synthetic series are used only inside automated tests to validate the algorithm, never as product data or interface results. Tests verify recovery of an injected 5.25-day box transit, rejection under a deliberately high threshold, and identification of alternating eclipse depths as an eclipsing-binary diagnostic. Safety tests verify frozen-host exclusion, post-detection period relationships, cache-only real access, and network/resource defaults.

## Real Kepler reference campaign

Acquisition campaign: `stage10-kepler-campaign-12`  
Final cache-reproducibility campaign: `stage10-kepler-reference-final`

- Requested/investigated: 12/12
- New MAST downloads in the acquisition campaign: 4; eight observations came from the local cache
- New MAST downloads in the final reproducibility run: 0
- Screening detections: 8
- Period-associated with local KOI records: 3
- Unassociated and explicitly inconclusive: 5
- Below the configured screen: 4
- Confirmed discoveries: 0
- Frozen-test access: false
- Neural-model invocation: false
- Acquisition-campaign runtime: 27.54 seconds on the local CPU
- Final cache-only reproducibility runtime: 5.79 seconds on the local CPU

| KIC | BLS period (d) | Depth SNR | Events | Coverage | Post-detection interpretation |
|---:|---:|---:|---:|---:|---|
| 9140402 | 2.495268 | 12.09 | 33 | 0.92 | KOI period association |
| 4366323 | 2.613498 | 5.03 | 31 | 0.91 | Below screen |
| 8527297 | 21.670882 | 6.12 | 5 | 1.00 | Unassociated, inconclusive |
| 10155029 | 30.000000 | 6.44 | 3 | 1.00 | Unassociated, inconclusive; peak is on search boundary |
| 10811496 | 19.898822 | 33.25 | 4 | 1.00 | KOI period association |
| 9005321 | 2.024024 | 5.32 | 43 | 0.96 | Below screen |
| 12109267 | 1.134361 | 5.82 | 75 | 0.95 | Below screen |
| 6043562 | 2.955144 | 5.46 | 28 | 0.93 | Below screen |
| 8626325 | 10.166949 | 6.02 | 7 | 0.78 | Unassociated, inconclusive |
| 10351947 | 21.446192 | 9.05 | 5 | 1.00 | Unassociated, inconclusive |
| 9772531 | 10.396915 | 99.28 | 8 | 1.00 | One-third KOI-period association; eclipsing-binary compatible |
| 11340218 | 5.002919 | 7.11 | 17 | 0.94 | Unassociated, inconclusive |

The exact values above come from persisted real campaign results. In particular, the five unmatched screens are not discovery claims. The local KOI snapshot contains KOI-linked TCE fields, but no standalone TCE catalog was available or queried, so absence of an association cannot establish novelty.

## Discovery Lab

The Developer Lab now includes a **Discovery Lab** tab. It exposes the protected expedition plan, explicit network permission, bounded downloads, real progress stages, persisted campaign selection, target-level diagnostics, and interactive Plotly views of:

- original observed luminosity;
- independently detrended luminosity;
- the complete BLS periodogram and selected period;
- phase-folded cadences and observed median bins;
- catalog associations, aliases, evidence, hashes, and limitations.

The interface always labels the output as screening evidence and displays zero confirmed discoveries.

## Execution

```powershell
# Inspect the deterministic plan; no network request
.\.venv\Scripts\python.exe -m src.discovery.cli plan --sample-size 10

# Run an explicit bounded campaign
.\.venv\Scripts\python.exe -m src.discovery.cli run `
  --sample-size 10 `
  --max-new-downloads 5 `
  --allow-network

# Inspect the latest complete machine-readable report
.\.venv\Scripts\python.exe -m src.discovery.cli inspect

# Open the integrated local laboratory
.\.venv\Scripts\python.exe -m streamlit run developer_lab.py
```

## Validation evidence

The final local suite completed with `77 passed` in 38.96 seconds. The focused Stage 09/10 and Streamlit suite completed with `17 passed`. The Streamlit AppTest loaded all 12 preserved tabs, including **Discovery Lab**, against the persisted final campaign. The local server health endpoint returned HTTP 200 with `ok` on `127.0.0.1:8501`.

Two dependency warnings remain non-blocking: Lightkurve reports that its optional `tpfmodel` helper lacks `oktopus`, and FastAPI/Starlette reports a future TestClient migration to `httpx2`. Neither warning changes the BLS light-curve workflow or its results.

## Limitations and next research questions

- Quarter 3 alone provides roughly 89 days and introduces selection and coverage bias.
- The 1–30 day grid excludes longer-period events and a boundary peak is not reliable evidence.
- A rolling median can attenuate long or broad astrophysical variability.
- BLS assumes a periodic box and is also sensitive to eclipsing binaries, stellar variability, contamination, and systematics.
- The current SNR threshold has not been calibrated to a false-alarm probability through injection/recovery or bootstrap studies.
- Uncertainties are grid/noise approximations rather than posterior intervals.
- No centroid, pixel-level contamination, multi-quarter consistency, stellar-vetting, or independent TCE-catalog validation was performed.
- ANN-01 needs six catalog features unavailable for an arbitrary new signal. ANN-02 was trained on a small catalog-derived Q3 distribution. Compatibility and out-of-distribution validation are required before either model can be applied responsibly.

Stage 11 was not started.
