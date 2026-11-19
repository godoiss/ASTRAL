# ASTRAL — Astronomical Neural Network

ASTRAL is a local-first research project for astronomical data analysis, initially focused on identifying potential exoplanets. Its first neural-network implementation will be built from scratch with Python and NumPy so that the learning mathematics remains explicit and testable. PyTorch may be evaluated in a later phase.

The current stage includes real Kepler Objects of Interest (KOI) acquisition, the proprietary NumPy ANN-01 engine, scientific validation, the Deep Space Observatory, and the PyTorch ANN-02 signal-intelligence pilot.

## Principles

- Local execution with no cloud-service dependency.
- No external artificial-intelligence APIs.
- Modular architecture and English naming in source code.
- Reproducible dependencies and automated tests.
- Compatibility with CPU-only development; optional GPU support may be introduced later.

## Requirements

- Python 3.14 or a compatible recent Python 3 release.
- Git.
- Approximately 2 GB of free disk space for the scientific Python environment.

The initial environment was prepared on an Intel Core i5 13th-generation IdeaPad 3i with 12 GB RAM. The architecture remains portable to an Acer Nitro 5 with an NVIDIA GTX 1650. ANN-02 detects CUDA automatically and falls back to CPU; install a CUDA wheel only after checking the target computer's driver against the official PyTorch compatibility matrix.

## Local setup

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the environment check and test suite:

```powershell
python main.py
python -m pytest
```

## Kepler KOI data acquisition

The raw dataset comes from the official [NASA Exoplanet Archive Table Access Protocol (TAP) service](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html). ASTRAL requests the complete `cumulative` table using this ADQL query:

```sql
SELECT * FROM cumulative
```

The archive describes the cumulative delivery as a collection of the current dispositions and transit information for Kepler Objects of Interest gathered from prioritized KOI activity tables. Because the delivery combines multiple activity tables and methods, the archive cautions that it is not a uniform statistical population. See the official [KOI cumulative-table provenance documentation](https://exoplanetarchive.ipac.caltech.edu/docs/PurposeOfKOITable.html) and [KOI column definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html).

Acquire the raw table and regenerate the exploratory artifacts:

```powershell
python -m src.data.acquire_kepler_koi
```

Use `--force` to replace an existing raw snapshot with a fresh archive response. The command produces:

- `data/raw/kepler_koi.csv`: complete, unmodified CSV response from the archive.
- `visualization/kepler_koi_report.md`: record count, original dispositions, missing values, and descriptive statistics.
- `visualization/kepler_koi_disposition_distribution.png`: classification distribution.
- `visualization/kepler_koi_missing_values.png`: missing values in required analysis columns.
- `visualization/kepler_koi_numeric_distributions.png`: distributions of required numeric fields.

The acquisition validates the object identifiers, original disposition columns, and the required fields `koi_period`, `koi_duration`, `koi_depth`, `koi_prad`, `koi_model_snr`, `koi_teq`, and `koi_disposition`. No normalization or other transformation is written to the raw CSV.

## ANN-01 neural engine

ANN-01 is a `6 → 8 → 4 → 1` multilayer perceptron with 97 trainable parameters. Forward propagation, backpropagation, activations, binary cross-entropy, gradient descent, and mini-batch training are implemented directly with NumPy. See the [mathematical and architectural documentation](docs/ann-01.md) and the [first technical validation report](docs/ann-01-technical-report.md).

Run the automated tests and the reproducible CPU experiment:

```powershell
python -m pytest -q
python -m src.training.cli train
python -m src.training.cli infer
```

The training command uses `configs/ann01.json`, splits labelled objects by whole `kepid` groups, fits preprocessing only on training records, and preserves `CANDIDATE` records separately. It writes a complete local model bundle to `models/ann-01/`. The inference command reloads that bundle and writes one result per candidate to `data/processed/ann01_candidate_predictions.csv`.

Use a different structured-log level by placing the global option before the command:

```powershell
python -m src.training.cli --log-level DEBUG train
```

Model bundles expose real weights, biases, activations, gradients, layer structure, training history, and contextual record-level outputs through JSON snapshots. Generated model and processed-data artifacts remain local and are excluded from Git.

## Stage 04 scientific validation

Stage 04 audits ANN-01 numerically, performs a controlled train/validation-only hyperparameter search, measures stability across random seeds, opens a newly frozen test partition once after selection, and compares the neural engine with majority-class, logistic-regression, decision-tree, and random-forest baselines. All labelled partitions are grouped by `kepid`, and preprocessing is fitted only on the training partition.

Run the complete local CPU campaign with a new immutable output directory:

```powershell
python -m src.validation.cli `
  --output experiments/stage04-rerun `
  --report docs/stage04-rerun.md
```

The campaign configuration is stored in `configs/scientific_validation.json`. The completed reference run is documented in the [Stage 04 scientific report](docs/stage04-scientific-validation.md), with machine-readable results in `experiments/stage04/scientific_validation_results.json`. Its artifacts include the experiment registry, per-run histories and parameters, the selected model bundle, a candidate ranking, the frozen split manifest, and nine high-resolution scientific figures.

Candidate sigmoid outputs are uncalibrated model scores. They are intended only for exploratory ranking and do not alter NASA dispositions or constitute exoplanet confirmation.

## ASTRAL Developer Lab

The experimental Developer Lab is a local Streamlit interface connected directly to the existing NumPy engine, persisted preprocessing parameters, Kepler dataset, and Stage 04 scientific artifacts. It provides real network topology and parameters, live training observations, individual KOI inference, scientific benchmarks, and dataset inspection. It does not use external AI services or synthetic interface metrics.

Start the local interface from the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run developer_lab.py
```

Open `http://127.0.0.1:8501` if the browser does not launch automatically. Interactive training writes a separate local bundle to `models/developer-lab/` and deliberately does not evaluate the reserved test partition. Use the model-source controls to return to the immutable Stage 04 selected model at any time.

## Deep Space Observatory

Stage 05 connects ASTRAL to the public MAST archive through Lightkurve and Astropy. The Developer Lab includes a **Deep Space Observatory** tab for bounded Kepler FITS acquisition, local cache inspection, interactive Plotly luminosity views, catalog-centered phase folding, physical KOI metadata, and ANN-01 inference. Original, processed, and neural-representation arrays are persisted separately.

Acquire one object or validate the five-object reference sample:

```powershell
python -m src.observatory.cli fetch K00753.01
python -m src.observatory.cli validate-sample
```

The default protocol downloads at most one long-cadence Quarter 3 product per star. Missing phase bins remain `NaN` with explicit masks; ASTRAL does not synthesize observational gaps. See the [Stage 05 technical report](docs/stage05-deep-space-observatory.md) and the machine-readable result at `experiments/stage05/deep_space_validation.json`.

## ANN-02 signal intelligence

Stage 06 adds a lightweight two-branch PyTorch CNN for real Kepler luminosity signals. Independent global (`2 × 2001`) and local (`2 × 201`) branches receive median-centered flux plus an explicit availability-mask channel, then fuse their learned representations into one binary logit. The architecture has 2,033 trainable parameters and uses `BCEWithLogitsLoss`, Adam, mini-batches, early-stopping support, deterministic seeds, checkpoints, and CPU/CUDA auto-selection.

The default acquisition is explicit and resumable: 8 balanced labelled KOIs plus one candidate, one Q3 FITS per star, no Stage 04 test records, a five-download ceiling, a 250 MB storage ceiling, and a hard timeout per object. It can be expanded deliberately up to 200 classified objects.

```powershell
python -m src.deep_learning.cli hardware
python -m src.deep_learning.cli build
python -m src.deep_learning.cli status
python -m src.deep_learning.cli train
python -m src.deep_learning.cli infer K00753.01
```

The original CPU pilot used 6 training records, 2 validation records, and one reserved candidate. A subsequent versioned experiment used 30 training records, 10 validation records, and the same unlabelled candidate while preserving the pilot model. Both experiments are operational evidence only; the samples are too small for definitive scientific claims. Read the [Stage 06 technical report](docs/stage06-ann02-deep-learning.md) and machine-readable active experiment at `experiments/stage06/latest_experiment.json`.

## Astronomical Intelligence

Stage 07 adds real dataset-quality and bias diagnostics, model-faithful ANN-01 and ANN-02 explanations, like-for-like model comparison on shared KOI identities, versioned CPU experiment campaigns, and an isolated TESS compatibility pilot. The materialized cohort contains 146 training objects, 47 validation objects, and one unlabelled candidate across 194 distinct stars; six requested KICs lacked a bounded Quarter 3 product and one failed the coverage rule. The Developer Lab exposes **Dataset Intelligence**, **Model Comparison**, **Explainable AI**, and **Research Experiments** while preserving every earlier surface. Kepler/TESS data are not mixed, candidates remain unlabelled, and the frozen test set remains outside development decisions. See the [Stage 07 technical report](docs/stage07-astronomical-intelligence.md).

## Backend & Scientific API

Stage 08 provides a local, versioned FastAPI interface over the existing ASTRAL scientific services and persisted artifacts. It exposes bounded read access to datasets, KOIs, cached observations, processed light curves, ANN-01/ANN-02 models, inference, comparison metrics, explanations, experiment histories, and the isolated TESS pilot. The API has no training or download endpoint, excludes frozen test identities, strips internal filesystem paths, and pins ANN-02 inference to the Stage 07 `seed_stability` checkpoint.

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for local interactive documentation. The Streamlit Developer Lab remains independently available on port 8501. See the [Stage 08 technical report](docs/stage08-backend-scientific-api.md) and the [technical and scientific API audit](docs/stage08-api-audit.md).

## Branding & Discovery-first UX — visual evaluation

Stage 09 established **Atlas** as the approved discovery-first identity after comparison with Signal Room and Quiet Orbit. The package includes an original orbit/star mark, machine-readable tokens, five-area UX architecture (`Discover → Observatory → Candidates → Neural Lab → Research`), accessibility and motion standards, and a responsive buildless client. Stage 10 connected that approved client to the scientific API; scientific computation remains in Python services, not in the browser.

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Atlas is now implemented in the publication Next.js application. See the [Stage 09 branding and discovery-first design-system specification](docs/stage09-branding-design-system.md).

## Discovery Engine

Stage 10 adds an independent Astropy Box Least Squares pipeline for real Kepler light curves. It performs catalog-independent quality filtering and detrending, measures periodic box-shaped dips, evaluates event and phase coverage, records aliases and eclipsing-binary diagnostics, and cross-checks the local KOI/TCE-linked catalog only after detection. Campaign targets remain unlabeled and exclude every host in the existing train, validation, and frozen-test partitions.

The completed bounded reference campaign investigated 12/12 stars, reused eight local observations, performed four new Quarter 3 downloads, produced eight screening detections, found three local KOI-period associations, and retained five unmatched screens as explicitly inconclusive. It reported zero confirmed discoveries, did not access the frozen test set, and did not invoke ANN-01 or ANN-02.

```powershell
.\.venv\Scripts\python.exe -m src.discovery.cli plan --sample-size 10
.\.venv\Scripts\python.exe -m src.discovery.cli run --sample-size 10 --max-new-downloads 5 --allow-network
.\.venv\Scripts\python.exe -m src.discovery.cli inspect
```

The approved Atlas interface is now connected directly to the FastAPI scientific services and is the primary Stage 10 product surface. It provides protected campaign planning, optional cache-only execution, real progress, persisted evidence, Observatory light curves, neural inference, model comparison, and unresolved-signal review without using fictional values.

```powershell
$env:ASTRAL_DISCOVERY_EXECUTION_ENABLED = "true"  # local research sessions only
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Streamlit remains available as an internal engineering laboratory, but it is not the primary product interface. Browser campaign execution never downloads observations and is disabled by default for publication safety. See the [Stage 10 technical and scientific report](docs/stage10-discovery-engine.md) and the [historical Atlas product-interface report](docs/stage10-atlas-product-interface.md).

## Immersive scientific visualization

The publication surface now lives in `web/` as a real Next.js application. It replaces the buildless Stage 10 client and provides the five Atlas areas as navigable routes: interactive real-coordinate stellar cartography, Plotly luminosity and BLS instruments, investigative candidate dossiers, checkpoint-faithful ANN visualizations, and a reproducible research timeline. Tables remain available only as secondary audit views.

Run the FastAPI backend and Next.js client in separate terminals:

```powershell
$env:ASTRAL_DISCOVERY_EXECUTION_ENABLED = "true"  # optional, local cache-only campaigns
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000

cd web
npm install
npm run dev
```

Open `http://127.0.0.1:3000`. FastAPI `/` and the legacy `/app/` entry point redirect to this publication client. See the [immersive visualization technical report](docs/immersive-scientific-visualization.md).

## Project structure

```text
ASTRAL/
├── configs/           # Reproducible engine, observatory, and validation configurations
├── data/
│   ├── raw/           # Original, immutable datasets (not versioned)
│   ├── processed/     # Cleaned and transformed datasets (not versioned)
│   ├── observatory/   # Local FITS cache, signal derivatives, provenance, and masks
│   └── ann02/         # Local resumable dataset, manifest, and checkpoints
├── models/            # Serialized model artifacts (not versioned)
├── docs/              # Architecture, mathematics, and technical reports
├── experiments/       # Versioned scientific-validation artifacts
├── web/               # Publication Next.js interface and Atlas instruments
├── src/
│   ├── api/           # FastAPI routers, schemas, services, and local safeguards
│   ├── core/          # Mathematical and neural-network primitives
│   ├── data/          # Data ingestion and preprocessing
│   ├── training/      # Training and evaluation workflows
│   ├── validation/    # Scientific audit, experiments, benchmarks, and reports
│   ├── lab/           # Streamlit integration and neural visualizations
│   ├── observatory/   # MAST acquisition, signal processing, and Plotly
│   ├── deep_learning/ # ANN-02 dataset, CNN, training, persistence, and observability
│   ├── discovery/     # Protected Kepler campaigns, BLS detection, diagnostics, and evidence
│   └── explorer/      # Astronomical-data exploration utilities
├── visualization/     # Visualization modules and outputs
├── tests/             # Automated tests
├── main.py            # Local application entry point
├── developer_lab.py   # Streamlit Developer Lab entry point
├── api.py             # Local FastAPI ASGI entry point
├── requirements.txt   # Direct Python dependencies
├── .gitignore
└── README.md
```

## Current scope

The current platform includes a local read-only REST API and an exploratory Kepler signal detector, but excludes public deployment, external AI services, automatic model combination, authentication infrastructure, and standalone frontend frameworks. TESS support is an isolated technical experiment and is not part of Kepler training. ANN-01 remains an exclusively NumPy neural engine. PyTorch is used only by ANN-02. Scikit-learn provides scientific metrics and classical comparison models. Network access is limited to explicitly invoked, bounded public scientific acquisition workflows; API inference and scientific queries remain local.
