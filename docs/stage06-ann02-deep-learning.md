# Stage 06 — ANN-02 Deep Learning Technical Report

## Scope and scientific reference

Stage 06 adds ANN-02, a local PyTorch one-dimensional convolutional network that classifies real Kepler phase-folded signals. ANN-01, its frozen Stage 04 evaluation, the Developer Lab, and the Deep Space Observatory were not modified mathematically.

The methodological reference is Christopher Shallue and Andrew Vanderburg, *Identifying Exoplanets with Deep Learning* ([arXiv:1712.05044](https://arxiv.org/abs/1712.05044)). ASTRAL adopts the paper's key idea of independent global and local phase views, but this implementation is not a reproduction of AstroNet.

## Differences from the reference methodology

| Aspect | Shallue & Vanderburg | ASTRAL ANN-02 pilot |
|---|---|---|
| Population | DR24 TCE labels: 3,600 PC, 9,596 AFP, 2,541 NTP | NASA cumulative KOIs: 8 labelled pilot records plus one reserved candidate |
| Observation span | Q1–Q17, up to four years | One bounded Quarter 3 long-cadence segment per pilot star |
| Views | Global 2,001 and local 201 bins | Global 2,001 and local 201 bins |
| Input channels | Flux | Median-centered flux and explicit availability mask |
| Missing bins | No separate mask channel | Source vectors retain `NaN`; tensors use neutral zero only with mask zero |
| Signal scaling | Median 0 and minimum −1 | Median 0 and robust 95th-percentile absolute scale, clipped to ±8 |
| Detrending | Per-light-curve basis spline spacing selected using BIC | Transit-masked Savitzky–Golay flattening from Stage 05 |
| Architecture | Deeper TensorFlow CNN selected by a large Vizier search | Lightweight two-branch PyTorch CNN with 2,033 parameters |
| Optimization | Adam, batch 64, 50 epochs, learning rate 1e-5 for the best model | Adam, batch 8, up to 40 epochs, learning rate 1e-3, early-stopping support |
| Regularization | Best reported configuration did not use dropout | Dropout 0.20 and weight decay 1e-4 |
| Augmentation | Random horizontal reflections | No augmentation in the pilot |
| Model selection | Thousands of trials and model averaging | One explicitly configured local pilot; no hyperparameter claim |
| Partitioning | Random 80/10/10 | Frozen Stage 04 KIC-grouped partitions; test records not acquired |

These differences materially limit metric comparability. The ASTRAL pilot is deliberately small, high-SNR-biased, short-period-biased, and cache-biased.

## Dataset pipeline

`src/deep_learning/dataset.py` selects a deterministic class-balanced cohort from the immutable Stage 04 split manifest. It never selects labelled `test` records. Candidates have label `-1` and cannot enter optimization.

Acquisition is explicit, incremental, and bounded:

- default pilot: 8 labelled KOIs and KOI-753.01 as a candidate;
- one Quarter 3 FITS product per star;
- at most five new downloads and 250 MB local observatory storage;
- persistent Lightkurve cache and per-object JSON checkpoint;
- isolated subprocess with a hard 120-second timeout per uncached object;
- configurable expansion up to 200 labelled objects and eight products per star;
- quality exclusions based on global coverage, local coverage, and observed transits;
- original cadences, processed arrays, provenance, masks, KOI, KIC, disposition, and frozen split remain linked.

The completed pilot accepted all nine requested records with no failures or exclusions:

| Split | Confirmed | False positive | Candidate | Total |
|---|---:|---:|---:|---:|
| Train | 3 | 3 | 0 | 6 |
| Validation | 1 | 1 | 0 | 2 |
| Candidate | 0 | 0 | 1 | 1 |

Every record contains 4,228 raw and 4,134 retained Q3 cadences. Global coverage ranges from 0.4143 to 0.9975, local coverage from 0.4328 to 1.0000, and observed catalog-centered transits from 4 to 102. The observatory tree occupied 7,509,271 bytes after this run. Dataset SHA-256: `847b9b70160da8d1283ae3c4dab9412265ade2da94695f55318a7db3752a4349`.

## Architecture

Both signal views have transformed-flux and binary-mask channels.

```text
Global [2 × 2001] ─ Conv1D(2→8) ─ ReLU ─ MaxPool(4) ─ Conv1D(8→16) ─ ReLU ─ AdaptiveAvgPool ┐
                                                                                               ├─ Concat(32) ─ FC(16) ─ ReLU ─ Dropout ─ Logit
Local  [2 ×  201] ─ Conv1D(2→8) ─ ReLU ─ MaxPool(2) ─ Conv1D(8→16) ─ ReLU ─ AdaptiveAvgPool ──┘
```

The model has exactly 2,033 trainable parameters. `BCEWithLogitsLoss` combines binary cross-entropy and a stable logit transformation. Adam performs mini-batch updates. CUDA is selected automatically when exposed by the installed PyTorch build; otherwise execution falls back to CPU.

## Pilot training results

The reference pilot used Python 3.14.4, PyTorch 2.14.1+cpu, seed 2026, and an Intel CPU. It completed 40 epochs in 7.29 seconds. The best checkpoint was epoch 40.

| Metric | Validation result |
|---|---:|
| Records | 2 |
| Accuracy | 0.5000 |
| Balanced accuracy | 0.5000 |
| Precision | 0.5000 |
| Recall | 1.0000 |
| F1 | 0.6667 |
| ROC-AUC | 1.0000 |
| PR-AUC | 1.0000 |
| Binary cross-entropy | 0.6792 |
| Confusion matrix | `[[0, 1], [0, 1]]` |

Validation loss decreased from 0.6941 to 0.6792. Accuracy remained 0.50. ROC-AUC and PR-AUC are numerically 1.0 because the positive record received a higher score than the negative record, but with one record per class these values have no statistical reliability. This demonstrates an operational end-to-end learning path, not scientific performance.

The frozen Stage 04 test partition was not downloaded, evaluated, or used for any decision. A future ANN-01/ANN-02 comparison requires a larger common cohort of stars untouched by both training processes.

## Candidate inference

For KOI-753.01 / KIC 10811496, ANN-02 produced an uncalibrated score of `0.514728` and ANN-01 produced `0.414767`. The threshold labels disagree. Neither score is a scientific probability, catalog reclassification, confirmation, or discovery claim.

## Neural observability and Developer Lab

The trained bundle persists the state dictionary, architecture, exact parameter count, filters, biases, real intermediate activations and feature maps, gradients from a genuine backward pass, epoch history, metrics, time, device, dataset hash, and guardrails.

The Developer Lab's **ANN-02 — Signal Intelligence** tab exposes dataset status, bounded acquisition controls, live training, architecture, metrics, source and tensor curves, individual inference, ANN-01 comparison, filters, and feature maps. It never triggers downloads automatically.

## Commands

```powershell
python -m src.deep_learning.cli hardware
python -m src.deep_learning.cli build
python -m src.deep_learning.cli status
python -m src.deep_learning.cli train
python -m src.deep_learning.cli infer K00753.01
python -m pytest -q
python -m streamlit run developer_lab.py
```

## Hardware and CUDA note

This IdeaPad exposed no NVIDIA device, so the verified distribution is CPU-only. On the GTX 1650 computer, first check `nvidia-smi`, then use the official PyTorch selector for a wheel compatible with that driver, Python version, Windows, and supported CUDA runtime. Verify with `python -m src.deep_learning.cli hardware`. Do not install a CUDA-specific wheel blindly.

## Limitations and next work within Stage 06

- Expand the labelled cohort before interpreting metrics.
- Add more quarters and quantify quarter-to-quarter consistency.
- Pre-register a larger common frozen test cohort for ANN-01/ANN-02 comparison.
- Measure seed stability and tune only on train/validation after sample growth.
- Investigate availability bias, low-SNR behavior, secondary eclipses, centroid information, and calibration.

TESS, model combination, FastAPI, Next.js, and Stage 07 remain outside this implementation.

## Expanded-dataset experiment (Stage 06 continuation)

Experiment `ann02-20261118T134320Z` used dataset SHA-256
`4ccb35e47b786023d8ee5e7fd5c02b286f50e781cbad8f5c82054fe6ed098142`.
The immutable collection contains 41 distinct KOIs hosted by 41 distinct KIC stars:

| Split | Confirmed | False positive | Unlabelled candidate | Total |
|---|---:|---:|---:|---:|
| Train | 15 | 15 | 0 | 30 |
| Validation | 5 | 5 | 0 | 10 |
| Candidate | 0 | 0 | 1 | 1 |
| Test | 0 | 0 | 0 | 0 |

Only the 30 training objects produced optimizer updates. The 10 validation objects
were evaluated with gradients disabled and selected the lowest-validation-loss
checkpoint. KOI-753.01 retained `class_label=-1` and was not supplied to either
loader. No KIC occurs in more than one split, and the materialized KIC set has zero
intersection with the 1,005 host stars in the frozen Stage 04 test partition.

Training used the unchanged 2,033-parameter architecture, seed 2026, Adam,
learning rate 0.001, batch size 8, weight decay 0.0001, and at most 40 epochs on
CPU. It completed 40 epochs in 6.672 seconds. Epoch 39 was selected; epoch 40 was
not persisted because its validation loss was higher.

| Metric | Original pilot (2 validation) | Expanded run (10 validation) |
|---|---:|---:|
| Accuracy | 0.5000 | 0.8000 |
| Balanced accuracy | 0.5000 | 0.8000 |
| Precision | 0.5000 | 0.8000 |
| Recall | 1.0000 | 0.8000 |
| F1 | 0.6667 | 0.8000 |
| ROC-AUC | 1.0000 | 0.7600 |
| PR-AUC | 1.0000 | 0.8600 |
| Binary cross-entropy | 0.6792 | 0.6355 |
| Confusion matrix | `[[0, 1], [0, 1]]` | `[[4, 1], [1, 4]]` |

The apparent improvement is descriptive, not a definitive validation: the
validation set has only five examples per class, the cohorts differ in size, and
selection remains balanced, short-period-, Q3-, cache-, quality-, and SNR-biased.
The pilot's perfect ROC/PR ranking arose from one example per class and is not
statistically meaningful. No test-set result is reported because that partition
remains unopened.

On the reserved KOI-753.01 candidate, the expanded model emitted an uncalibrated
score of `0.341610`, compared with `0.514728` from the original pilot. This change
is not evidence for or against planetary status: the candidate supplied no label,
did not participate in optimization or checkpoint selection, and neither score is
a calibrated probability.

The expanded checkpoint is stored separately under
`models/ann02/runs/ann02-20261118T134320Z`. The original pilot hash remains
`39fb14c1a47d23b5ec61f1de107c3e35587149b51909c16c7c407c116df71afa`; the
expanded model hash is
`7863c249588cc16b8ef60cacaf609b6717ce434f2dec898b4e023a609f444e45`.
The Developer Lab resolves the active-model pointer, renders the expanded learning
curves and metrics, and compares them with the preserved pilot. Its training button
now states that optimization uses training records only.
