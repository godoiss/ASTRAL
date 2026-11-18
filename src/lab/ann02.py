"""Streamlit presentation layer for real ANN-02 signal intelligence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.deep_learning.config import ANN02Config, DatasetBuildConfig, TrainingConfig
from src.deep_learning.dataset import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MANIFEST_PATH,
    KeplerSignalDataset,
    build_incremental_dataset,
)
from src.deep_learning.hardware import hardware_report
from src.deep_learning.model import DualBranchCNN
from src.deep_learning.training import (
    infer_ann02,
    load_ann02,
    train_ann02,
    validate_dataset_integrity,
)
from src.lab.services import InferenceResult, LabModelState, infer_record
from src.observatory.catalog import find_koi
from src.observatory.persistence import artifact_is_cached, load_artifact
from src.observatory.plots import original_flux_figure, processed_flux_figure


CYAN = "#00E5FF"
VIOLET = "#9D5CFF"
BLUE = "#247BFF"
PLOT_BACKGROUND = "#070B1D"


def _manifest() -> dict[str, Any] | None:
    if not DEFAULT_MANIFEST_PATH.is_file():
        return None
    try:
        return json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _line_chart(history: list[dict[str, Any]], fields: tuple[str, str], title: str) -> go.Figure:
    figure = go.Figure()
    colors = (CYAN, VIOLET)
    for field, color in zip(fields, colors, strict=True):
        figure.add_trace(
            go.Scatter(
                x=[row["epoch"] for row in history],
                y=[row[field] for row in history],
                mode="lines+markers",
                name=field.replace("_", " ").title(),
                line={"color": color, "width": 2},
            )
        )
    figure.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor=PLOT_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        height=330,
        margin={"l": 45, "r": 20, "t": 55, "b": 40},
        legend={"orientation": "h"},
    )
    return figure


def _view_figure(sample: dict[str, Any], view: str) -> go.Figure:
    values = sample[f"{view}_view"].numpy()
    flux, mask = values[0], values[1].astype(bool)
    coordinates = np.linspace(-0.5, 0.5, flux.size) if view == "global" else np.arange(flux.size)
    figure = go.Figure(
        go.Scattergl(
            x=coordinates[mask],
            y=flux[mask],
            mode="markers",
            marker={"size": 3, "color": CYAN if view == "global" else VIOLET, "opacity": 0.75},
            name="Observed bins",
        )
    )
    figure.update_layout(
        title=f"ANN-02 {view.title()} Tensor · median-centered robust scale",
        xaxis_title="Orbital phase" if view == "global" else "Local bin index",
        yaxis_title="Transformed flux",
        template="plotly_dark",
        paper_bgcolor=PLOT_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        height=300,
        margin={"l": 45, "r": 20, "t": 55, "b": 45},
    )
    return figure


def _heatmap(values: np.ndarray, title: str) -> go.Figure:
    data = values
    if data.ndim == 3:
        data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
    elif data.ndim == 1:
        data = data[np.newaxis, :]
    figure = go.Figure(
        go.Heatmap(z=data, colorscale=[[0.0, "#080A20"], [0.5, VIOLET], [1.0, CYAN]], colorbar={"title": "value"})
    )
    figure.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor=PLOT_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        height=330,
        margin={"l": 45, "r": 30, "t": 55, "b": 40},
    )
    return figure


def _dataset_section() -> None:
    st.markdown("#### Incremental Kepler dataset")
    manifest = _manifest()
    if manifest is None or not DEFAULT_DATASET_PATH.is_file():
        st.info("No ANN-02 dataset has been materialized. Acquisition is explicit and bounded.")
    else:
        counts = manifest["counts"]
        columns = st.columns(5)
        columns[0].metric("Accepted", counts["accepted"])
        columns[1].metric("Train", counts["by_split"]["train"])
        columns[2].metric("Validation", counts["by_split"]["validation"])
        columns[3].metric("Candidates", counts["by_split"]["candidate"])
        columns[4].metric("Failed", counts["failed"])
        with st.expander("Quality, provenance, exclusions, and selection biases"):
            records = pd.json_normalize(manifest["records"])
            if not records.empty:
                selected_columns = [
                    name for name in (
                        "object_id", "kic", "split", "disposition", "status", "exclusion_reason",
                        "quality.global_coverage", "quality.local_coverage", "quality.observed_transits",
                        "quality.segments", "provenance_path",
                    ) if name in records.columns
                ]
                st.dataframe(records[selected_columns], width="stretch", hide_index=True)
            for bias in manifest.get("selection_biases", []):
                st.caption(f"• {bias}")

    acquisition_columns = st.columns(4)
    pilot_size = acquisition_columns[0].select_slider(
        "Classified pilot size", options=[8, 16, 24, 32, 40, 64], value=8, key="ann02_pilot_size"
    )
    max_downloads = acquisition_columns[1].number_input(
        "Maximum new downloads", 0, 64, min(40, pilot_size + 1), key="ann02_max_downloads"
    )
    products = acquisition_columns[2].select_slider(
        "FITS per object", options=[1, 2, 3], value=1, key="ann02_products"
    )
    storage = acquisition_columns[3].number_input(
        "Storage ceiling (MB)", 50, 1000, 250, step=50, key="ann02_storage"
    )
    st.caption(
        "The default pilot is balanced, prefers cached Q3 observations, and excludes periods above 30 days. "
        "These controls reduce storage but introduce documented selection bias. Stage 04 test stars are never acquired here."
    )
    if st.button("Acquire / resume bounded pilot", key="ann02_acquire", type="primary"):
        progress = st.progress(0.0, text="Validating Stage 05 pipeline…")
        status = st.empty()

        def update(stage: str, index: int, total: int, message: str) -> None:
            progress.progress(index / max(total, 1), text=f"{index}/{total} · {message}")
            status.caption(f"{stage.upper()} · checkpoint persisted after each object")

        try:
            result = build_incremental_dataset(
                DatasetBuildConfig(
                    pilot_size=int(pilot_size),
                    max_new_downloads=int(max_downloads),
                    max_products_per_object=int(products),
                    storage_limit_mb=float(storage),
                ),
                progress_callback=update,
            )
        except Exception as exc:
            st.error(f"Bounded acquisition stopped safely: {exc}")
        else:
            progress.progress(1.0, text="Incremental dataset ready")
            st.success(f"Accepted {result['counts']['accepted']} real observations. Reloading status…")
            st.rerun()


def _training_section() -> Any:
    st.markdown("#### Convolutional training console")
    architecture = DualBranchCNN(ANN02Config()).architecture()
    columns = st.columns(4)
    columns[0].metric("Trainable parameters", f"{architecture['trainable_parameters']:,}")
    columns[1].metric("Input channels", "2 / view")
    columns[2].metric("Global bins", "2,001")
    columns[3].metric("Local bins", "201")
    with st.expander("Inspect the real PyTorch module graph"):
        st.dataframe(pd.DataFrame(architecture["modules"]), width="stretch", hide_index=True)
        st.caption("Each branch applies Conv1D → ReLU → pooling → Conv1D → ReLU, then adaptive pooling. Features are concatenated before dropout and one binary logit.")

    bundle = load_ann02()
    if bundle is not None:
        experiment = bundle.experiment
        st.success(f"ANN-02 model available · {experiment['experiment_id']} · {experiment['device_used']}")
        metrics = experiment["validation_metrics"]
        metric_columns = st.columns(5)
        for column, (label, field) in zip(
            metric_columns,
            (("Accuracy", "accuracy"), ("Balanced accuracy", "balanced_accuracy"), ("F1", "f1"), ("ROC-AUC", "roc_auc"), ("BCE", "binary_cross_entropy")),
            strict=True,
        ):
            column.metric(label, f"{metrics[field]:.4f}")
        chart_columns = st.columns(2)
        chart_columns[0].plotly_chart(_line_chart(experiment["history"], ("train_loss", "validation_loss"), "Training and Validation Loss"), width="stretch")
        chart_columns[1].plotly_chart(_line_chart(experiment["history"], ("train_accuracy", "validation_accuracy"), "Training and Validation Accuracy"), width="stretch")
        st.warning("Pilot validation metrics describe this small selected cohort only; they are not definitive scientific validation and the frozen test set remains untouched.")

    controls = st.columns(4)
    learning_rate = controls[0].select_slider("Learning rate", [0.0003, 0.001, 0.003], value=0.001, key="ann02_lr")
    epochs = controls[1].slider("Maximum epochs", 10, 100, 40, 5, key="ann02_epochs")
    batch_size = controls[2].select_slider("Batch size", [4, 8, 16, 32], value=8, key="ann02_batch")
    seed = int(controls[3].number_input("Random seed", 1, 999999, 2026, key="ann02_seed"))
    if st.button("Train ANN-02 on train + validation", key="ann02_train", disabled=not DEFAULT_DATASET_PATH.is_file()):
        chart = st.empty()
        status = st.empty()
        history: list[dict[str, float]] = []

        def epoch_update(row: dict[str, float]) -> None:
            history.append(row)
            status.caption(f"Epoch {int(row['epoch'])} · validation loss {row['validation_loss']:.5f}")
            chart.plotly_chart(_line_chart(history, ("train_loss", "validation_loss"), "Live Loss"), width="stretch")

        try:
            train_ann02(
                ANN02Config(),
                TrainingConfig(
                    learning_rate=float(learning_rate),
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    seed=seed,
                ),
                epoch_callback=epoch_update,
            )
        except Exception as exc:
            st.error(f"ANN-02 training could not start: {exc}")
        else:
            st.success("Best validation checkpoint persisted. The Stage 04 test set was not accessed.")
            st.rerun()
    return bundle


def _object_section(bundle: Any, ann01_state: LabModelState | None, ann_frame: pd.DataFrame) -> None:
    st.markdown("#### Signal inspection and individual inference")
    if not DEFAULT_DATASET_PATH.is_file():
        st.info("Build the bounded dataset to inspect real signals.")
        return
    dataset = KeplerSignalDataset(split=None)
    if not len(dataset):
        return
    options = [dataset[index]["object_id"] for index in range(len(dataset))]
    selected_id = st.selectbox("Astronomical object", options, key="ann02_object")
    sample = next(dataset[index] for index in range(len(dataset)) if dataset[index]["object_id"] == selected_id)
    metadata = find_koi(selected_id)[0]
    identity = st.columns(4)
    identity[0].metric("KOI", selected_id)
    identity[1].metric("KIC", metadata.kic)
    identity[2].metric("NASA disposition", metadata.disposition)
    identity[3].metric("Frozen split", sample["split"])

    if artifact_is_cached(selected_id):
        artifact = load_artifact(metadata)
        source_columns = st.columns(2)
        source_columns[0].plotly_chart(original_flux_figure(artifact), width="stretch", config={"displaylogo": False})
        source_columns[1].plotly_chart(processed_flux_figure(artifact), width="stretch", config={"displaylogo": False})
    tensor_columns = st.columns(2)
    tensor_columns[0].plotly_chart(_view_figure(sample, "global"), width="stretch")
    tensor_columns[1].plotly_chart(_view_figure(sample, "local"), width="stretch")
    st.caption("NaN bins are converted to the neutral centered value zero only in tensors. Their availability remains explicit in the second channel, so missing observations are not treated as measured flux.")

    if bundle is None:
        st.warning("ANN-02 inference is unavailable until a pilot model has been trained.")
        return
    result = infer_ann02(bundle, sample)
    scores = st.columns(5)
    scores[0].metric("ANN-02 score", f"{result['score']:.6f}")
    scores[1].metric("ANN-02 decision", result["predicted_class"])
    scores[2].metric("Device", result["device"])
    scores[3].metric("Latency", f"{result['elapsed_seconds'] * 1000:.2f} ms")
    ann01_result: InferenceResult | None = None
    if ann01_state is not None:
        matches = ann_frame[ann_frame["kepoi_name"].eq(selected_id)]
        if not matches.empty:
            ann01_result = infer_record(ann01_state, matches.iloc[0])
    scores[4].metric("ANN-01 score", "Unavailable" if ann01_result is None else f"{ann01_result.score:.6f}")
    st.warning("ANN-01 and ANN-02 use different evidence and neither sigmoid score is a calibrated confirmation probability. A candidate is never presented as a confirmed discovery.")

    maps = result["feature_maps"]
    if maps:
        selected_map = st.selectbox("Intermediate feature map", sorted(maps), key="ann02_map")
        st.plotly_chart(_heatmap(maps[selected_map], f"Real Feature Map · {selected_map}"), width="stretch")
    filter_name = st.selectbox(
        "Convolutional filter bank",
        [name for name, _ in bundle.model.named_parameters() if "conv_" in name and name.endswith("weight")],
        key="ann02_filter",
    )
    filters = dict(bundle.model.named_parameters())[filter_name].detach().cpu().numpy()
    st.plotly_chart(_heatmap(filters, f"Learned Filters · {filter_name}"), width="stretch")


def render_ann02(ann01_state: LabModelState | None, ann_frame: pd.DataFrame) -> None:
    """Render the ANN-02 Developer Lab without triggering automatic downloads."""
    st.subheader("ANN-02 — Signal Intelligence")
    st.write(
        "A lightweight two-branch 1D convolutional network that learns from real phase-folded Kepler luminosity signals. "
        "The global branch sees the orbit; the local branch focuses on the catalog transit window."
    )
    report = hardware_report()
    device_columns = st.columns(4)
    device_columns[0].metric("PyTorch", report["torch"])
    device_columns[1].metric("Runtime", report["selected_default"].upper())
    device_columns[2].metric("CUDA available", "Yes" if report["cuda_available"] else "No")
    device_columns[3].metric("Python", report["python"])
    if not report["cuda_available"]:
        st.caption("CPU fallback is active. On the GTX 1650 host, install a PyTorch CUDA build only after validating the NVIDIA driver and supported wheel matrix.")
    _dataset_section()
    st.divider()
    bundle = _training_section()
    st.divider()
    _object_section(bundle, ann01_state, ann_frame)
