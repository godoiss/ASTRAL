# ANN-01 neural engine

ANN-01 is ASTRAL's first proprietary neural engine. Neural operations are implemented with Python and NumPy only: there is no automatic differentiation and no neural-network framework.

## Architecture

| Layer | Units | Activation | Parameters |
| --- | ---: | --- | ---: |
| Input | 6 | Identity | 0 |
| Hidden 1 | 8 | ReLU | `6 × 8 + 8 = 56` |
| Hidden 2 | 4 | ReLU | `8 × 4 + 4 = 36` |
| Output | 1 | Sigmoid | `4 × 1 + 1 = 5` |
| Total | | | **97** |

Weight matrices follow the shape `(input_units, output_units)`. Hidden weights use He initialization, `W ~ N(0, 2 / fan_in)`, while the sigmoid output weights use `W ~ N(0, 1 / fan_in)`. All biases start at zero. The configured random seed makes initialization, splitting, and mini-batch shuffling reproducible.

## Forward propagation

For layer `l`, with mini-batch activation `A[l-1]`, weights `W[l]`, and biases `b[l]`:

```text
Z[l] = A[l-1] W[l] + b[l]
A[l] = ReLU(Z[l])                   for hidden layers
A[L] = sigmoid(Z[L])                for the output layer
```

ReLU is implemented in `src/core/activations.py` as `max(0, z)`. Its derivative is `1` for positive pre-activations and `0` otherwise; zero uses the deterministic subgradient `0`.

The sigmoid is evaluated with separate positive and negative branches to avoid exponential overflow:

```text
sigmoid(z) = 1 / (1 + exp(-z))      when z >= 0
sigmoid(z) = exp(z) / (1 + exp(z))  when z < 0
```

## Loss

`src/core/losses.py` implements mean binary cross-entropy for `m` examples:

```text
L = -(1 / m) Σ [y log(p) + (1 - y) log(1 - p)]
```

Probabilities are clipped only inside the logarithm evaluation to prevent `log(0)`. Input probabilities are not mutated.

## Backpropagation

Because the output combines sigmoid and binary cross-entropy, its derivative simplifies to:

```text
dZ[L] = (A[L] - Y) / m
dW[l] = A[l-1]ᵀ dZ[l]
db[l] = sum(dZ[l], axis=0)
dA[l-1] = dZ[l] W[l]ᵀ
dZ[l-1] = dA[l-1] * ReLU'(Z[l-1])
```

`MLP.backward` evaluates these equations explicitly from the output to the first hidden layer. `MLP.apply_gradients` then performs gradient descent:

```text
W[l] = W[l] - learning_rate × dW[l]
b[l] = b[l] - learning_rate × db[l]
```

`Trainer.fit` shuffles training indices deterministically every epoch and applies these operations to mini-batches.

## Gradient validation

`src/core/gradient_check.py` independently estimates every derivative using central differences:

```text
numeric_gradient[i] = (L(θ[i] + ε) - L(θ[i] - ε)) / (2ε)
```

The test suite compares all 97 analytical gradients against these estimates. This validates the chain-rule implementation without relying on automatic differentiation.

## Leakage-safe data preparation

The six inputs, in fixed order, are:

1. `koi_period`
2. `koi_duration`
3. `koi_depth`
4. `koi_prad`
5. `koi_model_snr`
6. `koi_teq`

Only `CONFIRMED` and `FALSE POSITIVE` objects are labelled for training. `CANDIDATE` objects remain in a separate inference partition.

The labelled split is made over whole `kepid` groups. A star and all of its labelled KOIs therefore belong to exactly one of train, validation, or test. The training partition alone determines:

- each feature's median for missing-value imputation;
- each feature's mean and population standard deviation for z-score scaling.

Those stored values are reused unchanged for validation, test, candidates, and later inference. The split manifest provides a record-level leakage audit.

## Observability

`MLP.observability_snapshot` exposes only actual internal state:

- architecture and layer activations;
- individual weights and biases;
- pre-activations and neuron activations from the latest retained forward pass;
- weight and bias gradients from the latest backward pass;
- per-epoch training and validation history;
- contextual record identifiers, targets, and inference probabilities supplied by the caller.

`MLP.export_snapshot` writes this state as standards-compliant JSON with non-finite values prohibited. The post-training snapshot uses a real training sample for both forward and backward passes, so its activations and gradients share the same context.

## Persistence

A model bundle contains:

- `model.npz`: architecture and NumPy parameter arrays, loaded with pickle disabled;
- `preprocessor.json`: feature order, medians, means, and scales;
- `config.json`: exact hyperparameters and random seed;
- `history.json`: real loss and accuracy for every epoch;
- `metrics.json`: split summaries and held-out test metrics;
- `network_snapshot.json`: post-training neural state;
- `inference_snapshot.json`: most recent inference activations, when inference has run.

## Interpretation limits

ANN-01 is an educational and experimental binary classifier, not a scientific confirmation instrument. A high probability for a `CANDIDATE` is a model output, not evidence that the object has been independently confirmed as an exoplanet. The current experiment does not perform probability calibration, uncertainty estimation, class weighting, or hyperparameter optimization.

