"""Streamlit application for the local ASTRAL Developer Lab."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pandas as pd
import streamlit as st

from src.data.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN
from src.lab.services import (
    DEFAULT_DATA_PATH,
    PROJECT_ROOT,
    InferenceResult,
    LabModelState,
    dataset_overview,
    infer_record,
    load_dataset_frame,
    load_latest_lab_model,
    load_scientific_model,
    scientific_validation_results,
    train_developer_model,
)
from src.lab.ann02 import render_ann02
from src.lab.discovery import render_discovery_lab
from src.lab.intelligence import (
    render_dataset_intelligence,
    render_explainable_ai,
    render_model_comparison,
    render_research_experiments,
)
from src.lab.visualization import (
    CYAN,
    VIOLET,
    activation_sample_count,
    activation_frame,
    close_figure,
    network_figure,
    parameter_frame,
    training_history_figure,
)
from src.observatory.catalog import load_observatory_catalog, metadata_from_row
from src.observatory.config import ObservatoryConfig
from src.observatory.persistence import artifact_is_cached
from src.observatory.pipeline import acquire_observation
from src.observatory.plots import (
    original_flux_figure,
    phase_folded_figure,
    processed_flux_figure,
    transit_zoom_figure,
)
from src.training.config import TrainingConfig


FIGURES_DIRECTORY = PROJECT_ROOT / "experiments" / "stage04" / "figures"
OBSERVATORY_CONFIG_PATH = PROJECT_ROOT / "configs" / "deep_space_observatory.json"
NASA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP"
NASA_DOCUMENTATION_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/docs/PurposeOfKOITable.html"
)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { --cyan:#00E5FF; --blue:#247BFF; --violet:#9D5CFF; }
        .stApp {
            background:
              radial-gradient(circle at 82% 4%, rgba(36,123,255,.12), transparent 30rem),
              radial-gradient(circle at 8% 28%, rgba(157,92,255,.09), transparent 26rem),
              #050816;
        }
        [data-testid="stSidebar"] { background:#070B1D; border-right:1px solid #172545; }
        [data-testid="stHeader"] { background:rgba(5,8,22,.82); }
        .block-container { max-width:1440px; padding-top:2.2rem; padding-bottom:4rem; }
        .astral-hero {
            padding:1.8rem 2rem; border:1px solid rgba(0,229,255,.28); border-radius:18px;
            background:linear-gradient(135deg,rgba(10,16,40,.96),rgba(13,24,57,.78));
            box-shadow:0 0 42px rgba(0,229,255,.08); margin-bottom:1.25rem;
        }
        .eyebrow { color:#00E5FF; letter-spacing:.18em; font-size:.76rem; font-weight:700; }
        .astral-hero h1 { margin:.35rem 0 .2rem; font-size:2.55rem; letter-spacing:-.04em; }
        .astral-hero p { color:#9CB1D2; max-width:850px; margin:0; }
        .status-pill {
            display:inline-flex; align-items:center; gap:.45rem; padding:.35rem .7rem;
            border:1px solid rgba(0,229,255,.28); border-radius:999px; color:#BDF7FF;
            background:rgba(0,229,255,.07); font-size:.76rem; margin-top:1rem;
        }
        .status-dot { width:7px; height:7px; border-radius:50%; background:#00E5FF; box-shadow:0 0 12px #00E5FF; }
        [data-testid="stMetric"] {
            background:rgba(10,16,40,.88); border:1px solid #182847; border-radius:14px;
            padding:.85rem 1rem; box-shadow:inset 0 1px rgba(255,255,255,.02);
        }
        [data-testid="stMetricValue"] { color:#E8F7FF; }
        .lab-card { background:rgba(10,16,40,.76); border:1px solid #182847; border-radius:14px; padding:1rem 1.15rem; }
        .score-card { border-left:3px solid #00E5FF; background:#091027; padding:1.1rem 1.3rem; border-radius:10px; }
        .score-value { color:#00E5FF; font-size:2.25rem; font-weight:750; line-height:1; }
        .score-label { color:#E8F7FF; font-size:1rem; font-weight:650; margin-top:.45rem; }
        .score-note { color:#8EA6C9; font-size:.78rem; margin-top:.45rem; }
        .stButton > button {
            border:1px solid rgba(0,229,255,.55); background:linear-gradient(135deg,#0A2146,#14204A);
            color:#E8F7FF; border-radius:10px; font-weight:650;
        }
        .stButton > button:hover { border-color:#00E5FF; color:#00E5FF; box-shadow:0 0 20px rgba(0,229,255,.12); }
        .atlas-discovery {
            padding:1.6rem 1.8rem; margin-bottom:1rem; border:1px solid #26394D;
            border-radius:14px; background:#0D1724; box-shadow:inset 3px 0 #70DBF7;
        }
        .atlas-discovery span { color:#70DBF7; font:600 .75rem Consolas,monospace; letter-spacing:.12em; }
        .atlas-discovery h2 { margin:.45rem 0; color:#F0F5F8; font:400 2.2rem Georgia,serif; }
        .atlas-discovery p { margin:0; max-width:850px; color:#B5C4D0; }
        div[data-baseweb="tab-list"] { gap:.4rem; border-bottom:1px solid #182847; }
        div[data-baseweb="tab"] { background:#091027; border-radius:9px 9px 0 0; padding:.7rem 1rem; }
        hr { border-color:#182847; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _dataset() -> pd.DataFrame:
    return load_dataset_frame(DEFAULT_DATA_PATH)


@st.cache_data(show_spinner=False)
def _overview() -> dict[str, Any]:
    return dataset_overview(_dataset(), DEFAULT_DATA_PATH)


@st.cache_data(show_spinner=False)
def _scientific_results() -> dict[str, Any] | None:
    return scientific_validation_results()


@st.cache_data(show_spinner=False)
def _observatory_catalog() -> pd.DataFrame:
    return load_observatory_catalog()


def _initial_state() -> LabModelState | None:
    if "model_state" not in st.session_state:
        st.session_state.model_state = load_scientific_model() or load_latest_lab_model()
    return st.session_state.model_state


def _sidebar(state: LabModelState | None, results: dict[str, Any] | None) -> dict[str, Any]:
    with st.sidebar:
        st.markdown("### ASTRAL // LAB")
        st.caption("Local neural systems console")
        if state is None:
            st.error("No trained model is available.")
        else:
            st.success("ANN-01 ONLINE")
            st.caption(state.source)
            st.code(" → ".join(str(size) for size in state.model.layer_sizes), language=None)
            st.caption(f"{state.model.parameter_count} trainable parameters")
        if results is None:
            st.warning("Scientific validation unavailable")
        else:
            st.caption("✓ Stage 04 scientific artifacts detected")

        st.divider()
        st.markdown("#### Model source")
        if st.button("Load validated Stage 04", width="stretch"):
            selected = load_scientific_model()
            if selected is None:
                st.error("Validated model artifacts were not found.")
            else:
                st.session_state.model_state = selected
                st.session_state.pop("inference_result", None)
                st.rerun()
        if st.button("Load latest Lab training", width="stretch"):
            selected = load_latest_lab_model()
            if selected is None:
                st.info("No Developer Lab training has been saved yet.")
            else:
                st.session_state.model_state = selected
                st.session_state.pop("inference_result", None)
                st.rerun()

        st.divider()
        st.markdown("#### Training controls")
        learning_rate = st.select_slider(
            "Learning rate",
            options=[0.003, 0.01, 0.03],
            value=0.03,
        )
        epochs = st.slider("Epochs", min_value=20, max_value=200, value=120, step=10)
        batch_size = st.select_slider("Batch size", options=[32, 64, 128], value=64)
        seed = int(st.number_input("Random seed", min_value=1, max_value=999999, value=2026))
        st.caption("Architecture is locked to ANN-01 · 6–8–4–1")
    return {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
    }


def _hero(state: LabModelState | None) -> None:
    source = state.source if state is not None else "Awaiting a trained model"
    st.markdown(
        f"""
        <section class="astral-hero">
          <div class="eyebrow">ASTRONOMICAL NEURAL SYSTEMS // EXPERIMENTAL CONSOLE</div>
          <h1>ASTRAL Developer Lab</h1>
          <p>Inspect, train and query ANN-01 directly in the browser. Every value below comes from the local Kepler dataset, the NumPy neural engine or persisted scientific artifacts.</p>
          <div class="status-pill"><span class="status-dot"></span>{source}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_observatory(state: LabModelState | None) -> None:
    st.subheader("Neural Observatory")
    if state is None:
        st.warning("Train ANN-01 to activate the neural observatory.")
        return
    columns = st.columns(4)
    columns[0].metric("Architecture", "6 · 8 · 4 · 1")
    columns[1].metric("Parameters", state.model.parameter_count)
    columns[2].metric("Connections", sum(weight.size for weight in state.model.weights))
    columns[3].metric("Seed", state.config.seed)

    active = state.model.activations if state.model.activations else None
    sample_index = 0
    if active is not None:
        sample_count = activation_sample_count(active)
        if sample_count > 1:
            sample_index = st.selectbox(
                "Activation record",
                options=list(range(sample_count)),
                format_func=lambda index: f"Cached record {index + 1} of {sample_count}",
                help="A record is selected explicitly so batch rows are never presented as additional neurons.",
            )
    figure = network_figure(state.model, activations=active, sample_index=sample_index)
    st.pyplot(figure, width="stretch")
    close_figure(figure)
    if active is None:
        st.caption("Nodes are neutral until an individual retained forward pass is executed. Connection colors and widths already reflect the loaded real weights.")
    else:
        st.caption("Node colors reflect the most recent real forward pass in this browser session.")

    layer = st.selectbox(
        "Inspect connection layer",
        options=list(range(len(state.model.weights))),
        format_func=lambda index: f"Layer {index} → {index + 1} · {state.model.weights[index].shape[0]} × {state.model.weights[index].shape[1]}",
    )
    weights = state.model.weights[layer]
    summary = st.columns(4)
    summary[0].metric("Mean weight", f"{weights.mean():.5f}")
    summary[1].metric("Std. deviation", f"{weights.std():.5f}")
    summary[2].metric("Minimum", f"{weights.min():.5f}")
    summary[3].metric("Maximum", f"{weights.max():.5f}")
    weight_column, bias_column = st.columns([3, 1])
    with weight_column:
        st.markdown("**Individual connection weights**")
        st.dataframe(parameter_frame(state.model, layer).style.background_gradient(cmap="coolwarm", axis=None), width="stretch")
    with bias_column:
        st.markdown("**Neuron biases**")
        st.dataframe(
            pd.DataFrame(
                {"bias": state.model.biases[layer].reshape(-1)},
                index=[f"N{index + 1}" for index in range(state.model.biases[layer].size)],
            ),
            width="stretch",
        )
    if active is not None:
        with st.expander("Latest neuron activations", expanded=False):
            st.dataframe(
                activation_frame(state.model, sample_index=sample_index),
                width="stretch",
                hide_index=True,
            )


def _render_training(state: LabModelState | None, controls: dict[str, Any]) -> LabModelState | None:
    st.subheader("Training Console")
    st.write(
        "Start a new ANN-01 training run on the real labelled Kepler records. "
        "The host-star split and preprocessing remain leakage-safe; this interactive run does not open the reserved test partition."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Learning rate", controls["learning_rate"])
    c2.metric("Epochs", controls["epochs"])
    c3.metric("Batch size", controls["batch_size"])
    c4.metric("Seed", controls["seed"])

    if st.button("Initiate ANN-01 training", type="primary", width="stretch"):
        config = TrainingConfig(
            learning_rate=float(controls["learning_rate"]),
            epochs=int(controls["epochs"]),
            batch_size=int(controls["batch_size"]),
            seed=int(controls["seed"]),
        )
        progress = st.progress(0.0, text="Preparing grouped Kepler partitions…")
        live_metrics = st.empty()
        live_chart = st.empty()

        def update_training(
            record: dict[str, int | float],
            history: list[dict[str, int | float]],
        ) -> None:
            epoch = int(record["epoch"])
            progress.progress(
                epoch / config.epochs,
                text=f"Epoch {epoch}/{config.epochs} · real mini-batch optimization",
            )
            refresh_interval = max(1, config.epochs // 60)
            if epoch == 1 or epoch == config.epochs or epoch % refresh_interval == 0:
                live_metrics.markdown(
                    f"**Train loss:** `{float(record['train_loss']):.5f}` &nbsp; "
                    f"**Validation loss:** `{float(record['validation_loss']):.5f}` &nbsp; "
                    f"**Train accuracy:** `{float(record['train_accuracy']):.4f}` &nbsp; "
                    f"**Validation accuracy:** `{float(record['validation_accuracy']):.4f}`"
                )
                live_frame = pd.DataFrame(history).set_index("epoch")
                live_chart.line_chart(
                    live_frame[["train_loss", "validation_loss"]],
                    color=[CYAN, VIOLET],
                )

        try:
            trained = train_developer_model(config, epoch_callback=update_training)
        except Exception as exc:
            progress.empty()
            st.exception(exc)
        else:
            progress.progress(1.0, text="Training completed and persisted locally")
            st.session_state.model_state = trained
            st.session_state.pop("inference_result", None)
            st.session_state.training_completed = True
            st.rerun()

    if state is not None and state.model.training_history:
        st.markdown("#### Loaded model learning history")
        figure = training_history_figure(state.model.training_history)
        st.pyplot(figure, width="stretch")
        close_figure(figure)
        final = state.model.training_history[-1]
        metrics = st.columns(4)
        metrics[0].metric("Final train loss", f"{float(final['train_loss']):.5f}")
        metrics[1].metric("Final validation loss", f"{float(final['validation_loss']):.5f}")
        metrics[2].metric("Train accuracy", f"{float(final['train_accuracy']):.2%}")
        metrics[3].metric("Validation accuracy", f"{float(final['validation_accuracy']):.2%}")
    return state


def _result_card(result: InferenceResult) -> None:
    st.markdown(
        f"""
        <div class="score-card">
          <div class="score-value">{result.score:.6f}</div>
          <div class="score-label">{result.predicted_class}</div>
          <div class="score-note">ANN sigmoid score · threshold {result.threshold:.2f} · not a calibrated scientific probability</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_inference(state: LabModelState | None, frame: pd.DataFrame) -> None:
    st.subheader("Astronomical Object Inference")
    if state is None:
        st.warning("No trained model is available for inference.")
        return
    disposition = st.selectbox(
        "Object population",
        options=["CANDIDATE", "CONFIRMED", "FALSE POSITIVE", "ALL"],
    )
    filtered = frame if disposition == "ALL" else frame[frame[TARGET_COLUMN] == disposition]
    labels = {
        index: f"{row['kepoi_name']} · kepid {row['kepid']} · {row[TARGET_COLUMN]}"
        for index, row in filtered.iterrows()
    }
    selected_index = st.selectbox(
        "Select a Kepler Object of Interest",
        options=list(labels),
        format_func=labels.get,
    )
    if st.button("Run neural inference", type="primary", width="stretch"):
        result = infer_record(state, filtered.loc[selected_index])
        st.session_state.inference_result = result

    result = st.session_state.get("inference_result")
    if result is None:
        st.info("Select an object and execute a retained forward pass.")
        return
    if result.object_id not in set(filtered["kepoi_name"].astype(str)):
        st.caption("Showing the most recent inference from another population filter.")

    identity, decision = st.columns([1, 1])
    with identity:
        st.markdown("#### Object identity")
        st.metric("KOI", result.object_id)
        st.write(f"**Host star (`kepid`):** {result.group_id}")
        st.write(f"**Original NASA disposition:** {result.original_disposition}")
    with decision:
        st.markdown("#### ANN-01 output")
        _result_card(result)
    if result.original_disposition == "CANDIDATE":
        st.warning("Exploratory ranking only: this score does not confirm or reject an exoplanet candidate.")

    raw = pd.DataFrame(
        {
            "feature": list(result.raw_features),
            "original_value": list(result.raw_features.values()),
            "normalized_input": [result.normalized_features[name] for name in result.raw_features],
        }
    )
    st.dataframe(raw, width="stretch", hide_index=True)
    figure = network_figure(state.model, activations=result.activations)
    st.pyplot(figure, width="stretch")
    close_figure(figure)
    with st.expander("Individual neuron activations", expanded=True):
        st.dataframe(
            activation_frame(state.model, result.activations),
            width="stretch",
            hide_index=True,
        )


def _render_validation(results: dict[str, Any] | None) -> None:
    st.subheader("Scientific Validation")
    if results is None:
        st.warning("Stage 04 scientific validation has not been executed or its artifacts are unavailable.")
        return
    metrics = results["final_test"]["astral_metrics"]
    columns = st.columns(4)
    columns[0].metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    columns[1].metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
    columns[2].metric("Balanced accuracy", f"{metrics['balanced_accuracy']:.4f}")
    columns[3].metric("F1-score", f"{metrics['f1']:.4f}")
    st.caption(
        f"Frozen grouped test · {metrics['records']} records · dataset {results['dataset']['version_id']} · evaluated once after model selection"
    )

    benchmark_rows = []
    for model_name, payload in results["benchmark"].items():
        model_metrics = payload["metrics"]
        benchmark_rows.append(
            {
                "model": model_name,
                "accuracy": model_metrics["accuracy"],
                "balanced_accuracy": model_metrics["balanced_accuracy"],
                "f1": model_metrics["f1"],
                "roc_auc": model_metrics["roc_auc"],
                "pr_auc": model_metrics["pr_auc"],
            }
        )
    st.dataframe(pd.DataFrame(benchmark_rows), width="stretch", hide_index=True)
    visualizations = [
        ("Learning curves", "01_loss.png"),
        ("Accuracy curves", "02_accuracy.png"),
        ("Confusion matrix", "03_confusion_matrix.png"),
        ("ROC curve", "04_roc_curve.png"),
        ("Precision–recall curve", "05_precision_recall_curve.png"),
        ("Model benchmark", "06_model_benchmark.png"),
        ("Weight distribution", "07_weight_distribution.png"),
        ("Activation distribution", "08_activation_distribution.png"),
        ("Prediction score distribution", "09_prediction_score_distribution.png"),
    ]
    selected = st.selectbox("Scientific figure", visualizations, format_func=lambda item: item[0])
    image_path = FIGURES_DIRECTORY / selected[1]
    if image_path.is_file():
        st.image(str(image_path), caption=selected[0], width="stretch")
    else:
        st.warning(f"Figure unavailable: {image_path.name}")


def _physical_value(value: float | None, suffix: str = "") -> str:
    return "Unavailable" if value is None else f"{value:,.4g}{suffix}"


def _render_deep_space(
    state: LabModelState | None,
    ann_frame: pd.DataFrame,
    catalog: pd.DataFrame,
) -> None:
    st.subheader("Deep Space Observatory")
    st.write(
        "Retrieve bounded Kepler FITS light curves from MAST, preserve original cadences, "
        "process the signal locally, and inspect the catalog-centered transit in orbital phase."
    )
    query = st.text_input(
        "Search by KOI, Kepler name, or KIC",
        value="K00753.01",
        placeholder="Example: KOI-753.01 or 10811496",
    ).strip()
    if query:
        upper = query.upper()
        matches = catalog[
            catalog["kepoi_name"].astype(str).str.upper().str.contains(upper, regex=False)
            | catalog["display_name"].astype(str).str.upper().str.contains(upper, regex=False)
            | catalog["kepler_name"].fillna("").astype(str).str.upper().str.contains(upper, regex=False)
            | catalog["kepid"].astype(str).str.contains(query, regex=False)
        ]
    else:
        matches = catalog.head(250)
    if matches.empty:
        st.warning("No local KOI metadata matched this search.")
        return
    options = list(matches.index)
    selected_index = st.selectbox(
        "Kepler object",
        options=options,
        format_func=lambda index: (
            f"{matches.loc[index, 'display_name']} · KIC {int(matches.loc[index, 'kepid'])} · "
            f"{matches.loc[index, 'koi_disposition']}"
        ),
    )
    selected_row = matches.loc[selected_index]
    metadata = metadata_from_row(selected_row)

    identity, physical = st.columns([1, 2])
    with identity:
        st.markdown("#### Target identity")
        st.metric("Object", metadata.display_name)
        st.write(f"**Archive KOI:** `{metadata.object_id}`")
        st.write(f"**Host star:** KIC {metadata.kic}")
        st.write(f"**NASA disposition:** {metadata.disposition}")
        st.write(f"**Stage 04 split:** `{metadata.split}`")
    with physical:
        st.markdown("#### Catalog parameters")
        values = pd.DataFrame(
            [
                ("Orbital period", _physical_value(metadata.period_days, " days")),
                ("Transit epoch", _physical_value(metadata.epoch_bkjd, " BKJD")),
                ("Transit duration", _physical_value(metadata.duration_hours, " h")),
                ("Transit depth", _physical_value(metadata.depth_ppm, " ppm")),
                ("Planet radius", _physical_value(metadata.planet_radius_earth, " R⊕")),
                ("Equilibrium temperature", _physical_value(metadata.equilibrium_temperature_k, " K")),
                ("Stellar temperature", _physical_value(metadata.stellar_temperature_k, " K")),
                ("Stellar radius", _physical_value(metadata.stellar_radius_solar, " R☉")),
                ("Stellar mass", _physical_value(metadata.stellar_mass_solar, " M☉")),
                ("Kepler magnitude", _physical_value(metadata.kepler_magnitude)),
            ],
            columns=["parameter", "catalog value"],
        )
        st.dataframe(values, width="stretch", hide_index=True)

    default_config = ObservatoryConfig.load(OBSERVATORY_CONFIG_PATH)
    settings = st.columns([2, 1, 1])
    with settings[0]:
        observation_scope = st.selectbox(
            "Observation scope",
            ["Kepler Quarter 3 · bounded validation", "All available quarters · bounded count"],
        )
    with settings[1]:
        max_products = st.select_slider("Maximum FITS products", options=[1, 2, 3], value=1)
    with settings[2]:
        force_refresh = st.checkbox("Reprocess cache", value=False)
    quarter = 3 if observation_scope.startswith("Kepler Quarter 3") else None
    config = replace(default_config, quarter=quarter, max_products=int(max_products))
    cached = artifact_is_cached(metadata.object_id)
    st.caption(
        ("Local processed cache available" if cached else "No processed cache yet")
        + " · downloads are limited to the selected product count"
    )

    if st.button("Obtain observational data", type="primary", width="stretch"):
        progress = st.progress(0.0, text="Initializing MAST request…")
        status = st.empty()

        def update(stage: str, value: float, message: str) -> None:
            progress.progress(value, text=message)
            status.caption(f"{stage.upper()} · {message}")

        try:
            artifact, example = acquire_observation(
                metadata,
                config,
                force_refresh=force_refresh,
                progress_callback=update,
            )
        except Exception as exc:
            progress.empty()
            status.empty()
            st.error(f"Observational data unavailable: {exc}")
        else:
            st.session_state.observatory_artifact = artifact
            st.session_state.observatory_example = example
            st.session_state.observatory_object_id = metadata.object_id
            progress.progress(1.0, text="Observation ready")
            st.success("Real MAST observation loaded and persisted locally.")

    artifact = st.session_state.get("observatory_artifact")
    example = st.session_state.get("observatory_example")
    current_object = st.session_state.get("observatory_object_id")
    if artifact is None or example is None or current_object != metadata.object_id:
        st.info("Obtain the selected observation to unlock interactive luminosity views.")
        return

    observation_metrics = st.columns(5)
    observation_metrics[0].metric("Mission", artifact.mission)
    observation_metrics[1].metric("Segments", len(artifact.quarters))
    observation_metrics[2].metric("Raw cadences", f"{artifact.raw_records:,}")
    observation_metrics[3].metric("Retained cadences", f"{artifact.processed_records:,}")
    observation_metrics[4].metric("Cadence", f"{artifact.cadence_seconds / 60:.1f} min")
    st.caption(
        f"Observation: BKJD {artifact.observation_start_bkjd:.3f}–{artifact.observation_end_bkjd:.3f} · "
        + ", ".join(artifact.quarters)
    )

    st.plotly_chart(
        original_flux_figure(artifact),
        width="stretch",
        config={"displaylogo": False, "scrollZoom": True},
    )
    st.info(
        "Original luminosity shows the PDCSAP flux read from the cached FITS product. "
        "No missing cadence is synthesized; quality flags remain available in hover data."
    )
    st.plotly_chart(
        processed_flux_figure(artifact),
        width="stretch",
        config={"displaylogo": False, "scrollZoom": True},
    )
    st.info(
        "Processed luminosity removes non-finite and configured quality-flagged cadences, "
        "normalizes each observing segment, and estimates slow trends while masking catalog transit windows."
    )
    phase_column, transit_column = st.columns(2)
    with phase_column:
        st.plotly_chart(
            phase_folded_figure(artifact, example),
            width="stretch",
            config={"displaylogo": False},
        )
        st.caption(
            "Global view folds every retained cadence using the catalog period and `koi_time0bk`; phase zero is the expected transit center."
        )
    with transit_column:
        st.plotly_chart(
            transit_zoom_figure(artifact, example),
            width="stretch",
            config={"displaylogo": False},
        )
        st.caption(
            "Local view magnifies the expected event in hours. A dip can have planetary, stellar, contamination, or instrumental explanations."
        )

    vector_metrics = st.columns(4)
    vector_metrics[0].metric("Global vector", example.global_flux.size)
    vector_metrics[1].metric("Global observed bins", int(example.global_mask.sum()))
    vector_metrics[2].metric("Local vector", example.local_flux.size)
    vector_metrics[3].metric("Local observed bins", int(example.local_mask.sum()))
    st.caption(
        "Fixed views are ANN-02-ready representations. Empty bins remain NaN with mask=False; no signal is interpolated into observational gaps."
    )

    st.markdown("#### ANN-01 catalog inference")
    if state is None:
        st.warning("ANN-01 model unavailable.")
    else:
        ann_matches = ann_frame[ann_frame["kepoi_name"].eq(metadata.object_id)]
        if ann_matches.empty:
            st.warning("This KOI is not available in the ANN-01 feature table.")
        else:
            result = infer_record(state, ann_matches.iloc[0])
            score_columns = st.columns(3)
            score_columns[0].metric("ANN score", f"{result.score:.6f}")
            score_columns[1].metric("ANN decision", result.predicted_class)
            score_columns[2].metric("NASA catalog", result.original_disposition)
            st.warning(
                "ANN-01 uses six catalog features, not the plotted light curve. Its sigmoid output is not a calibrated scientific probability."
            )

    with st.expander("Scientific provenance and local artifacts"):
        st.json(artifact.provenance, expanded=False)
        st.download_button(
            "Download provenance JSON",
            data=json.dumps(artifact.provenance, indent=2, ensure_ascii=False),
            file_name=f"{metadata.object_id}_provenance.json",
            mime="application/json",
        )


def _render_dataset(frame: pd.DataFrame, overview: dict[str, Any]) -> None:
    st.subheader("NASA Kepler Dataset")
    columns = st.columns(4)
    columns[0].metric("KOI records", f"{overview['records']:,}")
    columns[1].metric("Host stars", f"{overview['stars']:,}")
    columns[2].metric("Unique objects", f"{overview['objects']:,}")
    columns[3].metric("Input features", len(FEATURE_COLUMNS))
    st.markdown(
        f"Local snapshot of the NASA Exoplanet Archive `cumulative` Kepler Objects of Interest table. "
        f"Dataset version: `{overview['version_id']}`. [NASA provenance]({NASA_DOCUMENTATION_URL}) · [TAP service]({NASA_TAP_URL})"
    )
    distribution = pd.Series(overview["dispositions"], name="records").sort_values(ascending=False)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Original dispositions")
        st.bar_chart(distribution, color=CYAN)
    with right:
        st.markdown("#### Missing values before training-only imputation")
        missing = pd.DataFrame(
            {
                "feature": list(overview["missing_values"]),
                "missing_records": list(overview["missing_values"].values()),
            }
        )
        st.dataframe(missing, width="stretch", hide_index=True)
    with st.expander("Inspect local KOI records"):
        st.dataframe(
            frame.loc[:, ["kepoi_name", "kepid", TARGET_COLUMN, *FEATURE_COLUMNS]].head(500),
            width="stretch",
            hide_index=True,
        )
        st.caption("Preview limited to 500 records; inference selection uses the complete local table.")


def main() -> None:
    """Render the ASTRAL Developer Lab."""
    st.set_page_config(
        page_title="ASTRAL Developer Lab",
        page_icon="✦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()
    frame = _dataset()
    overview = _overview()
    observatory_catalog = _observatory_catalog()
    results = _scientific_results()
    state = _initial_state()
    controls = _sidebar(state, results)
    state = st.session_state.get("model_state")
    _hero(state)
    if st.session_state.pop("training_completed", False):
        st.toast(
            "ANN-01 training completed. The reserved test partition was not evaluated.",
            icon="✅",
        )

    tabs = st.tabs(
        [
            "Neural Observatory",
            "Training Console",
            "Object Inference",
            "Deep Space Observatory",
            "Discovery Lab",
            "ANN-02 — Signal Intelligence",
            "Scientific Validation",
            "NASA Dataset",
            "Dataset Intelligence",
            "Model Comparison",
            "Explainable AI",
            "Research Experiments",
        ]
    )
    with tabs[0]:
        _render_observatory(state)
    with tabs[1]:
        state = _render_training(state, controls)
    with tabs[2]:
        _render_inference(state, frame)
    with tabs[3]:
        _render_deep_space(state, frame, observatory_catalog)
    with tabs[4]:
        render_discovery_lab()
    with tabs[5]:
        render_ann02(state, frame)
    with tabs[6]:
        _render_validation(results)
    with tabs[7]:
        _render_dataset(frame, overview)
    with tabs[8]:
        render_dataset_intelligence()
    with tabs[9]:
        render_model_comparison(state, frame)
    with tabs[10]:
        render_explainable_ai(state, frame)
    with tabs[11]:
        render_research_experiments()

    st.divider()
    st.caption("ASTRAL Developer Lab · Local execution · NumPy ANN-01 + PyTorch ANN-02 · No external AI services")


if __name__ == "__main__":
    main()
