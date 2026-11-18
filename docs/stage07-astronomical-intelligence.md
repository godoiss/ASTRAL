# Stage 07 — Astronomical Intelligence

## Scope and scientific protocol

Stage 07 expands ASTRAL's scientific-data diagnostics, controlled ANN-02
experimentation, model-faithful explainability, and like-for-like ANN-01/ANN-02
comparison. The frozen Stage 04 test partition remains unavailable to development
workflows. Validation results guide checkpoints and experiments but are not
reported as final performance on unseen data.

No candidate is converted into a labelled example. No Kepler and TESS samples are
mixed automatically. No ensemble decision rule is implemented.

## Dataset expansion protocol

The initial target is 200 labelled KOIs plus the separately retained KOI-753.01
candidate. Selection remains grouped by `kepid`, balanced by catalog disposition,
and restricted to the frozen Stage 04 training and validation assignments.

Acquisition is incremental and resumable:

- one Kepler Quarter 3 long-cadence product per host star;
- no more than 40 new FITS downloads per invocation;
- 250 MB observatory storage ceiling;
- persistent Lightkurve cache and per-object checkpoint;
- 120-second hard timeout for each uncached target;
- explicit global/local coverage and observed-transit exclusions;
- no materialization of frozen test records.

This design is intentionally bounded but biased. Quarter 3 alone cannot measure
quarter-to-quarter robustness. Periods above 30 days are excluded, favoring
systems with several transits in one quarter. Cached and higher catalog-SNR targets
are preferred, producing a viability cohort rather than a prevalence-representative
sample.

## Explainability methods

### ANN-01

Local explanations use central finite differences of the real ANN-01 sigmoid score
with respect to each normalized input feature. A first-order local contribution is
computed relative to the training-feature mean. Cohort importance uses repeated
permutation of one validation feature at a time and measures the increase in binary
cross-entropy.

These values describe model sensitivity, not causal astronomical effects. Feature
correlation can redistribute apparent importance.

### ANN-02

Signal explanations combine two calculations on the real CNN:

1. input-gradient × input saliency for every global and local flux bin;
2. bounded region occlusion, replacing one region with normalized median flux and
   measuring the resulting score delta.

The strongest region is the region most influential to this model under this
perturbation. It is not automatically a transit, physical explanation, or planet
confirmation.

## Like-for-like model comparison

ANN-01 and ANN-02 metrics are recomputed from the same KOI identities in the
ANN-02 validation split. The comparison records individual scores, thresholded
classes, agreement, divergence, and errors. ANN-01 catalog-feature scores and
ANN-02 morphology scores are not assumed to be calibrated or interchangeable.

Because ANN-02 checkpoint selection uses this validation cohort, the comparison is
diagnostic rather than a final test. An automatic ensemble is deliberately absent.

## TESS technical study

Lightkurve exposes Kepler and TESS products through the same search and
`LightCurve` abstractions, and both missions distribute FITS time-series products
through MAST. Compatibility at the container/API level does not imply a shared
training distribution.

Important domain differences include KIC/KOI versus TIC/TOI identities, quarter
versus sector segmentation, BKJD (`BJD - 2454833`) versus BTJD
(`BJD - 2457000`), cadence modes, observing baseline, pixel scale, blending, and
mission-specific systematics. The experimental TESS download is therefore stored
under `data/tess_experimental` and never enters the Kepler ANN-02 dataset.

Official technical references:

- [Lightkurve search_lightcurve](https://lightkurve.github.io/lightkurve/reference/api/lightkurve.search_lightcurve.html)
- [Lightkurve light-curve product tutorial](https://lightkurve.github.io/lightkurve/tutorials/1-getting-started/using-light-curve-file-products.html)
- [MAST TESS archive](https://archive.stsci.edu/missions-and-data/tess)
- [MAST TESS data-product overview](https://outerspace.stsci.edu/spaces/TESS/pages/14563420/2.0%2B-%2BData%2BProduct%2BOverview)

## Reproducible commands

```powershell
python -m src.deep_learning.cli --config configs/ann02.json build
python -m src.intelligence.cli dataset-report
python -m src.intelligence.cli compare
python -m src.intelligence.cli campaign
python -m src.intelligence.cli explain
python -m src.intelligence.cli tess-compatibility
python -m src.intelligence.cli tess-pilot
python -m pytest -q
python -m streamlit run developer_lab.py
```

## Results and limitations

The bounded acquisition requested 200 labelled KOIs and one separately retained
candidate. It materialized **194 real objects from 194 distinct host stars**:

- training: 146 objects (75 false positives, 71 confirmed);
- validation: 47 objects (24 false positives, 23 confirmed);
- candidate: KOI-753.01, retained without a training label;
- test: zero materialized records and zero overlap with frozen test stars.

Six selected KICs had no Kepler Quarter 3 light-curve product available through
the bounded MAST query. One further object failed the minimum local-coverage rule.
No replacement was selected after observing these outcomes. The final observatory
cache occupies 123.98 MiB, below the 250 MiB ceiling.

The selected cohort has a median orbital period of 6.10 days and median catalog
model SNR of 2857.7. By comparison, all eligible short-period labelled records
have median model SNR 35.5. The 30-day period ceiling removes 25.09% of otherwise
eligible labelled catalog records. Together with Quarter 3-only coverage and
high-SNR preference, this is a strong selection bias and limits generalization.

### Controlled ANN-02 campaign

Five versioned CPU trials used 146 training objects and the same 47 validation
objects. Early stopping and checkpoint selection used validation BCE only:

| Trial | Best epoch | Accuracy | Balanced accuracy | F1 | ROC-AUC | PR-AUC | BCE |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 33 | 0.7234 | 0.7237 | 0.7234 | 0.7790 | 0.7334 | 0.579122 |
| low learning rate | 59 | 0.6596 | 0.6585 | 0.6364 | 0.7373 | 0.7048 | 0.602840 |
| wider branches | 19 | 0.6809 | 0.6812 | 0.6809 | 0.7500 | 0.6768 | 0.585901 |
| stronger regularization | 34 | 0.7021 | 0.7029 | 0.7083 | 0.7826 | 0.7368 | 0.562325 |
| seed 97 | 27 | **0.7660** | **0.7681** | **0.7843** | 0.7663 | 0.6988 | **0.541733** |

The seed-97 checkpoint was selected by the declared BCE rule. Different seeds and
configurations produce materially different results. Repeated use of this small
validation partition can overfit model selection, so none of these values is a
claim of final performance on unseen data.

### Like-for-like comparison and explainability

On exactly the same 47 validation objects, ANN-01 reached accuracy 0.8723,
balanced accuracy 0.8696, ROC-AUC 1.0000, and BCE 0.241167. ANN-02 reached
accuracy 0.7660, balanced accuracy 0.7681, ROC-AUC 0.7663, and BCE 0.541733.
The models agreed on 32 of 47 objects (68.09%). The cohort was used in ANN-02
selection, the input modalities differ, and neither score is treated as a
calibrated probability; an automatic ensemble is therefore not justified.

ANN-01 validation permutation importance was led by `koi_depth` (mean BCE increase
7.1453), followed by `koi_teq` (0.1770) and `koi_duration` (0.1212). These are
model sensitivities in a biased, small cohort, not causal astrophysical evidence.
The persisted explanation report also contains finite-difference local ANN-01
sensitivities and ANN-02 gradient/occlusion maps for one confirmed object, one
false positive, and the retained candidate.

### Isolated TESS pilot

One official SPOC light curve was acquired for TIC 441420236, Sector 27. It has
16,812 cadences, median cadence 120.003 seconds, 16,781 finite flux records, 45
quality-flagged records, and a 24.37-day span. It remains under
`data/tess_experimental`, is ignored by Git, and was not mixed into ANN-02.

Machine-readable evidence is stored under `experiments/stage07/`, including the
dataset report, trial campaign, individual training histories, shared-cohort model
comparison, explanation values, and TESS provenance. The frozen test partition
remains unopened throughout Stage 07.
