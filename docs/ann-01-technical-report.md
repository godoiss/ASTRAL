# ANN-01 technical validation report

## Scope

This report records the first CPU experiment of ASTRAL's NumPy-only ANN-01 engine. No frontend, REST API, external AI service, synthetic training record, PyTorch, TensorFlow, or Keras component was used.

## Implemented components

- Dense MLP with architecture `6 → 8 → 4 → 1` and exactly 97 trainable parameters.
- He/Xavier initialization, ReLU, stable sigmoid, binary cross-entropy, explicit backpropagation, gradient descent, and mini-batches.
- Grouped train/validation/test partitioning by `kepid`.
- Training-only median imputation and z-score parameters.
- Separate preservation and inference of `CANDIDATE` objects.
- JSON-serializable access to weights, biases, activations, pre-activations, gradients, architecture, history, and contextual inference results.
- NumPy model persistence, preprocessing persistence, JSON hyperparameters, deterministic seed, and structured JSON logging.
- Training and inference command-line interfaces.

## Experimental configuration

| Setting | Value |
| --- | ---: |
| Seed | 42 |
| Learning rate | 0.01 |
| Epochs | 120 |
| Mini-batch size | 64 |
| Train / validation / test | 70% / 15% / 15% |
| Classification threshold | 0.5 |

## Real-data partitions

| Partition | Records | Distinct `kepid` | Confirmed | False positive |
| --- | ---: | ---: | ---: | ---: |
| Train | 5,311 | 4,671 | 1,916 | 3,395 |
| Validation | 1,139 | 974 | 432 | 707 |
| Test | 1,137 | 996 | 400 | 737 |
| Candidate inference | 1,977 | 1,861 | N/A | N/A |

The leakage audit found **zero** `kepid` groups shared by train, validation, and test. Preprocessing parameters were fitted on the 5,311 training records only.

## Preliminary results

| Partition / epoch | Loss | Accuracy |
| --- | ---: | ---: |
| Train, epoch 120 | 0.395572 | 81.0958% |
| Validation, epoch 120 | 0.398606 | 81.2994% |
| Held-out test | 0.406294 | 80.1231% |

Loss decreased from `0.610365` to `0.395572` on training data and from `0.612524` to `0.398606` on validation data. The similar final train and validation trajectories provide no obvious sign of severe overfitting in this initial run, but this is not a substitute for broader scientific validation.

The candidate-only inference file contains all 1,977 original `CANDIDATE` records and valid probabilities in `[0, 1]`. At the experimental threshold, ANN-01 assigned 960 records to the positive side and 1,017 to the negative side. These are uncalibrated experimental classifications, not new astronomical dispositions.

## Numerical and automated validation

- Unit/integration suite: **21 passed**.
- Numerical gradient check: all **97** parameters checked.
- Relative analytical/numerical gradient error: `9.156115638831769e-11`.
- Maximum absolute gradient error: `1.093370533000737e-11`.
- Persisted model reload reproduced model outputs.
- Snapshot JSON contains 97 parameters and real forward/backward state for 16 training records.
- Candidate inference audit found zero probabilities outside `[0, 1]`.

The test run emits one existing optional Lightkurve warning about `tpfmodel`/`oktopus`; it is unrelated to ANN-01 and does not affect the neural engine.

## Primary files

- `src/core/activations.py`: ReLU, derivative, and stable sigmoid.
- `src/core/losses.py`: binary cross-entropy.
- `src/core/mlp.py`: parameters, forward/backward passes, updates, persistence, and observability.
- `src/core/gradient_check.py`: finite-difference validation.
- `src/data/preprocessing.py`: labels, candidates, grouped split, imputation, scaling, and manifests.
- `src/training/config.py`: validated hyperparameters.
- `src/training/trainer.py`: mini-batch loop, evaluation, history, and plot.
- `src/training/persistence.py`: complete model bundles.
- `src/training/logging_utils.py`: structured JSON logging.
- `src/training/cli.py`: training and inference commands.
- `configs/ann01.json`: reproducible experimental configuration.
- `tests/test_core_math.py`, `tests/test_mlp.py`, `tests/test_preprocessing.py`, and `tests/test_training.py`: mathematical, gradient, data, training, and persistence tests.

## Reproduction

From the project root with the virtual environment activated:

```powershell
python -m pytest -q
python -m src.training.cli train
python -m src.training.cli infer
```

Training produces a local bundle under `models/ann-01/`, the split manifest under `data/processed/`, and the real learning curves under `visualization/ann01_training_history.png`. Inference produces `data/processed/ann01_candidate_predictions.csv`.

