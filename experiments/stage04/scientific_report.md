# ASTRAL Stage 04 — Scientific Validation Report

## Executive summary

Campaign `stage04-scientific-validation-v1` evaluated ASTRAL on the real Kepler cumulative KOI dataset using a frozen host-star-grouped split. Hyperparameters and initialization seed were selected exclusively on training and validation data. The campaign test partition was opened only after selection.

The selected configuration was `stability-004-seed-2026` with architecture `6-8-4-1`, learning rate `0.03`, batch size `64`, `120` epochs, and seed `2026`. Final held-out PR-AUC was `0.827992` and balanced accuracy was `0.819532`.

## Methodology

- Dataset version: `sha256:a85348f20082d24f`.
- Dataset SHA-256: `a85348f20082d24fb14d4a05a9a44c644b801998fad7cec36f8980eb1f0a69ab`.
- Grouped split seed: `104729`.
- Group key: `kepid`; no host star is shared by train, validation, and test.
- Missing values: training median; scaling: training mean and population standard deviation.
- Selection metric: validation PR-AUC (average precision), pre-registered before execution.
- Decision threshold: 0.5 for threshold-based metrics; ROC-AUC and PR-AUC use continuous scores.
- ANN training: NumPy implementation only; Scikit-learn is restricted to metrics and comparison models.

### Frozen split

| Partition | Records | Stars | Confirmed | False Positive |
| --- | --- | --- | --- | --- |
| train | 5311 | 4626 | 1918 | 3393 |
| validation | 1139 | 1010 | 420 | 719 |
| test | 1137 | 1005 | 410 | 727 |
| candidates | 1977 | 1861 | N/A | N/A |

## Neural-engine audit

- Forward outputs finite and in `[0, 1]`: `True`.
- ANN-01 parameter count: `97`.
- Parameter shapes valid and all values finite: `True`.
- All gradients finite: `True`.
- Numerical gradient relative error: `3.64936552032e-11`.
- Maximum absolute gradient error: `9.49835349262e-12`.
- Group overlaps (train/validation, train/test, validation/test): `[0, 0, 0]`.

## Controlled hyperparameter search

| Experiment | Architecture | LR | Epochs | Batch | Val PR-AUC | Val Bal. Acc. | Val BCE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp-001-ann01-original | 6-8-4-1 | 0.010000 | 120 | 64 | 0.774964 | 0.813380 | 0.378831 |
| exp-002-learning-rate-low | 6-8-4-1 | 0.003000 | 120 | 64 | 0.743488 | 0.775273 | 0.441015 |
| exp-003-learning-rate-high | 6-8-4-1 | 0.030000 | 120 | 64 | 0.819489 | 0.840973 | 0.337696 |
| exp-004-epochs-short | 6-8-4-1 | 0.010000 | 60 | 64 | 0.743727 | 0.787673 | 0.419823 |
| exp-005-epochs-long | 6-8-4-1 | 0.010000 | 200 | 64 | 0.800462 | 0.828962 | 0.357885 |
| exp-006-batch-small | 6-8-4-1 | 0.010000 | 120 | 32 | 0.803765 | 0.840761 | 0.349763 |
| exp-007-batch-large | 6-8-4-1 | 0.010000 | 120 | 128 | 0.747062 | 0.787673 | 0.419185 |
| exp-008-neurons-narrow | 6-6-3-1 | 0.010000 | 120 | 64 | 0.770976 | 0.804434 | 0.404033 |
| exp-009-neurons-wide | 6-12-6-1 | 0.010000 | 120 | 64 | 0.781549 | 0.819132 | 0.369791 |
| exp-010-one-hidden-layer | 6-8-1 | 0.010000 | 120 | 64 | 0.748524 | 0.791645 | 0.410362 |
| exp-011-three-hidden-layers | 6-12-8-4-1 | 0.010000 | 120 | 64 | 0.786772 | 0.834809 | 0.359443 |

The original ANN-01 architecture remains preserved as the reference experiment. Architecture variants are Stage 04 experimental models and do not overwrite the ANN-01 configuration or bundle.

## Seed stability

| Seed | Val PR-AUC | Val ROC-AUC | Val F1 | Val BCE |
| --- | --- | --- | --- | --- |
| 7 | 0.779154 | 0.906673 | 0.788288 | 0.358379 |
| 42 | 0.819489 | 0.922021 | 0.797235 | 0.337696 |
| 97 | 0.815416 | 0.911163 | 0.793220 | 0.356166 |
| 2026 | 0.833706 | 0.919594 | 0.779582 | 0.345315 |
| 31415 | 0.822473 | 0.912792 | 0.789766 | 0.353319 |

Validation PR-AUC mean was `0.814048` with standard deviation `0.018474`.

## Final held-out evaluation and benchmarks

| Model | Accuracy | Balanced Acc. | Precision | Recall | F1 | ROC-AUC | PR-AUC | BCE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ASTRAL Neural Engine | 0.824978 | 0.819532 | 0.737079 | 0.800000 | 0.767251 | 0.906005 | 0.827992 | 0.369904 |
| Majority Class | 0.639402 | 0.500000 | 0.000000 | 0.000000 | 0.000000 | 0.500000 | 0.360598 | 12.997272 |
| Logistic Regression | 0.781003 | 0.772914 | 0.679287 | 0.743902 | 0.710128 | 0.861697 | 0.725675 | 0.434310 |
| Decision Tree | 0.894459 | 0.896731 | 0.820796 | 0.904878 | 0.860789 | 0.951136 | 0.887435 | 0.398735 |
| Random Forest | 0.921724 | 0.916456 | 0.886747 | 0.897561 | 0.892121 | 0.969915 | 0.942934 | 0.241957 |

ASTRAL confusion matrix (`[[TN, FP], [FN, TP]]`): `[[610, 117], [82, 328]]`.

All models used the same training records, frozen test records, feature values, and binary labels. Tree models received the same standardized features for procedural equivalence, although scaling is not necessary for their split rules. Benchmark hyperparameters were fixed before test evaluation and were not tuned on test results.

## Learning behavior

The selected run's train-minus-validation accuracy gap was `-0.002647` and validation-minus-train BCE gap was `-0.023647`. These curves support assessment of overfitting or underfitting but do not prove generalization beyond the Kepler cumulative table.

## Candidate exploration

The selected model scored all original `CANDIDATE` records. Scores are sigmoid model outputs and are explicitly not scientifically calibrated probabilities. Ranking does not change NASA dispositions and does not constitute exoplanet confirmation.

| Rank | KOI | kepid | ANN Score | Period | Radius | Teq |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | K01404.02 | 8874090 | 0.872714 | 18.906092 | 0.870000 | 336.000000 |
| 2 | K02554.02 | 10471621 | 0.871880 | 10.271275 | 0.940000 | 510.000000 |
| 3 | K00430.02 | 10717241 | 0.867407 | 9.340538 | 0.840000 | 530.000000 |
| 4 | K04509.01 | 11350118 | 0.866100 | 12.003678 | 0.970000 | 503.000000 |
| 5 | K00227.01 | 6185476 | 0.864900 | 17.657378 | 2.450000 | 350.000000 |
| 6 | K05831.01 | 10813078 | 0.858602 | 10.552242 | 1.750000 | 485.000000 |
| 7 | K01681.04 | 5531953 | 0.858284 | 21.913955 | 10.390000 | 303.000000 |
| 8 | K00605.02 | 4832837 | 0.856351 | 5.065469 | 0.620000 | 592.000000 |
| 9 | K04957.02 | 2861126 | 0.855910 | 4.987924 | 0.730000 | 485.000000 |
| 10 | K02124.01 | 11462341 | 0.855551 | 42.337580 | 1.450000 | 311.000000 |

## Scientific visualizations

- `figures/01_loss.png`
- `figures/02_accuracy.png`
- `figures/03_confusion_matrix.png`
- `figures/04_roc_curve.png`
- `figures/05_precision_recall_curve.png`
- `figures/06_model_benchmark.png`
- `figures/07_weight_distribution.png`
- `figures/08_activation_distribution.png`
- `figures/09_prediction_score_distribution.png`

## Limitations

1. Stage 03 already exposed a test result from a different split (seed 42). Stage 04 froze a new grouped split before search, but this does not replace external validation on an independent dataset.
2. The NASA cumulative KOI table combines deliveries and analysis pipelines and is not a uniform statistical population.
3. Labels and selected transit features originate from the same broader vetting ecosystem, so dataset and provenance bias may inflate apparent transferability.
4. The controlled search is intentionally modest for local CPU execution; repeated nested grouped cross-validation was not performed.
5. Scores are uncalibrated. No uncertainty intervals, calibration curve, or astrophysical false-discovery control is provided.
6. Candidate ranking is exploratory only and cannot establish planetary status.

## Conclusions

ANN-01's forward and backward mathematics pass numerical validation, its grouped protocol prevents direct host-star leakage across labelled partitions, and its final metrics are reported alongside stronger and weaker classical baselines without result manipulation. The evidence supports that the engine learns reproducible signal within this dataset; it does not yet establish deployment-grade or discovery-grade scientific validity.

## Opportunities for evolution

- External validation on a separately curated Kepler or TESS dataset.
- Nested stratified-group cross-validation and confidence intervals.
- Probability calibration using validation-only procedures.
- Threshold selection based on a declared scientific cost function.
- Feature-ablation and permutation-importance studies.
- Explicit uncertainty and out-of-distribution detection.
