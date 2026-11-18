# ASTRAL Stage 05 — Deep Space Observatory

## Scope

Stage 05 adds bounded access to real Kepler stellar-luminosity measurements and prepares reusable, mask-aware signal representations for a future ANN-02. It does not implement ANN-02, modify ANN-01, change the Stage 04 split, or overwrite `data/raw/kepler_koi.csv`.

## Scientific sources

- [MAST Kepler mission archive](https://archive.stsci.edu/missions-and-data/kepler)
- [MAST TESS mission archive](https://archive.stsci.edu/missions-and-data/tess)
- [Lightkurve search and download documentation](https://lightkurve.github.io/lightkurve/reference/search.html)
- [Lightkurve folding documentation](https://lightkurve.github.io/lightkurve/reference/api/lightkurve.LightCurve.fold.html)
- [NASA Exoplanet Archive KOI column definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html)

The KOI catalog defines `koi_time0bk` as the transit epoch in BKJD, where `BKJD = BJD - 2,454,833.0`. ASTRAL records both time systems in provenance and uses BKJD consistently with Kepler light-curve times.

## Acquisition and cache

`src.observatory.mast` searches MAST using an explicit `KIC <identifier>` target, mission, author, cadence, and optional quarter. Downloads are bounded to 1–8 products by validated configuration; the reference configuration permits one long-cadence Kepler Quarter 3 product. Requests have configurable timeouts, retries, structured errors, and progress callbacks.

Lightkurve stores original FITS products under `data/observatory/cache/lightkurve/`. ASTRAL additionally persists, in separate files:

- original time, flux, flux error, quality flags, and segment identity;
- cleaned, normalized, detrended, phase-folded, and local-view arrays;
- scientific provenance, configuration, source products, hashes, and software versions;
- fixed-size global/local vectors and observation masks.

The original cumulative KOI CSV remains unchanged. An observatory-only metadata view containing `koi_time0bk` and physical parameters is written to `data/observatory/catalog/`.

## Signal-processing protocol

1. Read `pdcsap_flux`, time, errors, and quality flags from the cached FITS product.
2. Preserve all raw cadences before filtering.
3. Remove non-finite samples and cadences matching the configured Kepler quality bitmask (`1130799`).
4. Normalize every observing segment independently using Lightkurve.
5. Estimate low-frequency trends with a Savitzky–Golay window of 101 samples and polynomial order 2. Catalog transit windows are excluded from trend estimation to reduce transit attenuation.
6. Unite retained observation segments and sort by BKJD.
7. Fold with the catalog period and epoch:

   `phase = ((BKJD - epoch + period / 2) mod period) / period - 0.5`

8. Produce a transit-centered local view in hours.

No gap is filled and no missing observation is synthesized.

## ANN-02 dataset preparation

The reusable preparation layer produces:

- a 2,001-bin global orbital view;
- a 201-bin local transit view;
- Boolean masks identifying bins that contain real observations;
- KOI and KIC identity;
- Stage 04 split assignment;
- class `1` for `CONFIRMED`, `0` for `FALSE POSITIVE`, and `-1` for unlabelled candidates.

Empty bins remain `NaN` and have a `False` mask. The group-integrity validator rejects labelled examples where one KIC appears in multiple partitions. No test record is used to tune processing or a model.

## Bounded validation sample

Validation ran on 16 November 2026 with one Quarter 3 long-cadence FITS file per host star. All four labelled examples belong to the frozen Stage 04 training partition.

| Object | KIC | NASA disposition | Split | Raw | Processed | Global observed bins | Local observed bins |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| K00752.01 | 10797460 | CONFIRMED | train | 4,228 | 4,134 | 1,933 / 2,001 | 161 / 201 |
| K00755.01 | 10854555 | CONFIRMED | train | 4,228 | 4,134 | 1,858 / 2,001 | 201 / 201 |
| K00754.01 | 10848459 | FALSE POSITIVE | train | 4,228 | 4,134 | 829 / 2,001 | 104 / 201 |
| K00114.01 | 6721123 | FALSE POSITIVE | train | 4,228 | 4,134 | 1,996 / 2,001 | 201 / 201 |
| K00753.01 | 10811496 | CANDIDATE | candidate | 4,228 | 4,134 | 1,810 / 2,001 | 87 / 201 |

The five FITS products total approximately 2.33 MB. Their median cadence is approximately 1,765.4 seconds (29.42 minutes). KOI-753.01 spans BKJD 260.22494–349.49622 in this bounded observation.

Machine-readable results are stored in `experiments/stage05/deep_space_validation.json`. Local per-object provenance remains beside cached artifacts and includes source-file SHA-256 hashes.

## Developer Lab integration

The existing Streamlit application now includes a sixth tab, **Deep Space Observatory**, with:

- KOI, Kepler-name, and KIC search;
- bounded MAST acquisition and progress feedback;
- cache status and observation metadata;
- interactive Plotly original, processed, global phase, and local transit views;
- catalog identity and physical parameters;
- ANN-01 catalog-feature inference shown separately from light-curve analysis;
- vector/mask coverage and downloadable provenance.

The Neural Observatory now requires an explicit activation sample index for multi-record caches. A table always represents the actual 19 neurons of one selected record, rather than flattening batch rows into fictitious neurons.

## Limitations

1. The reference validation uses one quarter per star, not the complete Kepler baseline.
2. Catalog ephemerides may accumulate timing error away from their reference epoch.
3. Detrending parameters are a conservative infrastructure default, not a scientifically optimized choice.
4. PDCSAP corrections and catalog dispositions carry mission-pipeline assumptions and provenance bias.
5. A phase-centered dip does not establish a planetary origin; false positives and stellar eclipses can produce similar signals.
6. Fixed vectors are infrastructure for future work and have not been used to train ANN-02.

## Execution

Acquire one object:

```powershell
python -m src.observatory.cli fetch K00753.01
```

Repeat the bounded reference validation, using the persistent cache when present:

```powershell
python -m src.observatory.cli validate-sample
```

Launch the integrated Lab:

```powershell
python -m streamlit run developer_lab.py
```
