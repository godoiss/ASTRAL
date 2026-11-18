"""Streamlit surfaces for Stage 07 Astronomical Intelligence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.preprocessing import OBJECT_ID_COLUMN
from src.deep_learning.dataset import KeplerSignalDataset
from src.deep_learning.training import infer_ann02, load_ann02
from src.intelligence.comparison import DEFAULT_REPORT_PATH as COMPARISON_PATH
from src.intelligence.comparison import compare_ann01_ann02
from src.intelligence.dataset import DEFAULT_REPORT_PATH as DATASET_REPORT_PATH
from src.intelligence.dataset import build_dataset_intelligence
from src.intelligence.experiments import DEFAULT_STAGE07_EXPERIMENTS, run_ann02_campaign
from src.intelligence.explainability import (
    explain_ann01_local,
    explain_ann02_signal,
    permutation_importance_ann01,
)
from src.intelligence.tess import DEFAULT_TESS_REPORT, run_tess_pilot, tess_compatibility_report
from src.lab.services import LabModelState


CYAN = "#00E5FF"
VIOLET = "#9D5CFF"
BLUE = "#247BFF"


def _json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _metric_frame(section: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": name.replace("_", " ").title(), **values}
            for name, values in section.items()
        ]
    )


def render_dataset_intelligence() -> None:
    """Render real cohort composition, quality, astronomy, and selection bias."""
    st.subheader("Dataset Intelligence")
    st.write("Scientific diagnostics for the materialized ANN-02 cohort. No test records are opened by this view.")
    try:
        report = build_dataset_intelligence(report_path=DATASET_REPORT_PATH)
    except Exception as exc:
        st.error(f"Dataset intelligence unavailable: {exc}")
        return
    dataset = report["dataset"]
    columns = st.columns(5)
    for column, (label, value) in zip(
        columns,
        (("Accepted", dataset["accepted"]), ("Host stars", dataset["stars"]), ("Excluded", dataset["excluded"]), ("Failed", dataset["failed"]), ("Test accessed", "No")),
        strict=True,
    ):
        column.metric(label, value)

    distribution_rows = [
        {"split": split, "class": disposition, "records": records}
        for split, values in report["class_distribution"].items()
        for disposition, records in values.items()
    ]
    distribution = pd.DataFrame(distribution_rows)
    if not distribution.empty:
        figure = go.Figure()
        for disposition, color in (("CONFIRMED", CYAN), ("FALSE POSITIVE", VIOLET), ("CANDIDATE", BLUE)):
            subset = distribution[distribution["class"].eq(disposition)]
            if not subset.empty:
                figure.add_bar(x=subset["split"], y=subset["records"], name=disposition, marker_color=color)
        figure.update_layout(template="plotly_dark", barmode="group", title="Class distribution by frozen split")
        st.plotly_chart(figure, width="stretch")

    left, right = st.columns(2)
    left.markdown("#### Signal quality")
    left.dataframe(_metric_frame(report["quality"]), width="stretch", hide_index=True)
    right.markdown("#### Astronomical distributions")
    right.dataframe(_metric_frame(report["astronomy"]), width="stretch", hide_index=True)

    bias = report["bias_analysis"]
    st.markdown("#### Selection-bias audit")
    bias_columns = st.columns(4)
    bias_columns[0].metric("Quarter", bias["quarter"])
    bias_columns[1].metric("Period ceiling", f"{bias['period_limit_days']:.0f} d")
    bias_columns[2].metric("Eligible before ceiling", bias["eligible_labelled_records_before_period_limit"])
    bias_columns[3].metric("Removed by ceiling", f"{100*bias['fraction_eligible_removed_by_period_limit']:.1f}%")
    for limitation in bias["documented_effects"]:
        st.warning(limitation)


def render_model_comparison(ann01_state: LabModelState | None, ann_frame: pd.DataFrame) -> None:
    """Render like-for-like scores and metrics on one shared validation cohort."""
    st.subheader("Model Comparison")
    st.write("ANN-01 and ANN-02 are compared only on identical validation object IDs. The frozen test set remains unopened.")
    report = _json(COMPARISON_PATH)
    if st.button("Recompute comparable validation cohort", key="stage07_compare"):
        ann02 = load_ann02(device_preference="cpu")
        if ann01_state is None or ann02 is None:
            st.error("Both persisted models are required.")
        else:
            report = compare_ann01_ann02(ann01_state, ann02, ann_frame)
            st.rerun()
    if report is None:
        st.info("No comparable-cohort artifact is available yet.")
        return
    cohort = report["cohort"]
    summary = st.columns(4)
    summary[0].metric("Shared objects", cohort["records"])
    summary[1].metric("Shared stars", cohort["stars"])
    summary[2].metric("Model agreement", f"{100*report['agreement']['fraction']:.1f}%")
    summary[3].metric("Divergences", report["agreement"]["divergences"])
    metric_names = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "binary_cross_entropy")
    metrics = pd.DataFrame(
        {
            "Metric": [name.replace("_", " ").title() for name in metric_names],
            "ANN-01": [report["ann01_metrics"][name] for name in metric_names],
            "ANN-02": [report["ann02_metrics"][name] for name in metric_names],
        }
    )
    st.dataframe(metrics, width="stretch", hide_index=True)
    rows = pd.DataFrame(report["rows"])
    st.plotly_chart(
        go.Figure(
            [
                go.Scatter(
                    x=rows["ann01_score"], y=rows["ann02_score"], mode="markers+text",
                    text=rows["object_id"], textposition="top center",
                    marker={"color": np.where(rows["models_agree"], CYAN, VIOLET), "size": 10},
                )
            ]
        ).update_layout(template="plotly_dark", title="Individual score comparison", xaxis_title="ANN-01 score", yaxis_title="ANN-02 score"),
        width="stretch",
    )
    st.dataframe(rows, width="stretch", hide_index=True)
    for limitation in report["limitations"]:
        st.caption(f"• {limitation}")


def render_explainable_ai(ann01_state: LabModelState | None, ann_frame: pd.DataFrame) -> None:
    """Render model-faithful explanations for one real KOI."""
    st.subheader("Explainable AI")
    st.write("These views explain model sensitivity. They do not establish astrophysical causality or confirm planets.")
    ann02 = load_ann02(device_preference="cpu")
    if ann01_state is None or ann02 is None:
        st.info("Both persisted models are required for this area.")
        return
    dataset = KeplerSignalDataset(split=None)
    validation_dataset = KeplerSignalDataset(split="validation")
    validation_ids = {validation_dataset[index]["object_id"] for index in range(len(validation_dataset))}
    validation_frame = ann_frame[ann_frame[OBJECT_ID_COLUMN].isin(validation_ids)]
    global_importance = permutation_importance_ann01(
        ann01_state, validation_frame, repeats=16, seed=2027
    )
    importance_frame = pd.DataFrame(global_importance["features"])
    st.plotly_chart(
        go.Figure(
            [
                go.Bar(
                    x=importance_frame["feature"],
                    y=importance_frame["mean_bce_increase"],
                    error_y={"type": "data", "array": importance_frame["std_bce_increase"]},
                    marker_color=CYAN,
                )
            ]
        ).update_layout(
            template="plotly_dark",
            title=f"ANN-01 permutation importance · shared validation cohort ({global_importance['records']} objects)",
            yaxis_title="Validation BCE increase",
        ),
        width="stretch",
    )
    st.caption("A positive BCE increase means the feature carried useful information for ANN-01 on this validation cohort. Correlation and small sample size limit interpretation.")
    options = [dataset[index]["object_id"] for index in range(len(dataset))]
    selected = st.selectbox("KOI for explanation", options, key="stage07_xai_object")
    sample = next(dataset[index] for index in range(len(dataset)) if dataset[index]["object_id"] == selected)
    matches = ann_frame[ann_frame[OBJECT_ID_COLUMN].eq(selected)]
    left, right = st.columns(2)
    if len(matches) == 1:
        ann01 = explain_ann01_local(ann01_state, matches.iloc[0])
        left.metric("ANN-01 score", f"{ann01['score']:.6f}")
        left.dataframe(pd.DataFrame(ann01["features"]), width="stretch", hide_index=True)
        left.caption("Finite-difference gradients and first-order contributions around the real normalized input.")
    if st.button("Compute ANN-02 signal sensitivity", key="stage07_xai_compute"):
        st.session_state["stage07_ann02_xai"] = explain_ann02_signal(ann02, sample)
    explanation = st.session_state.get("stage07_ann02_xai")
    if explanation and explanation.get("object_id") == selected:
        right.metric("ANN-02 score", f"{explanation['score']:.6f}")
        regions = pd.DataFrame(explanation["regions"])
        top = regions.head(20)
        right.plotly_chart(
            go.Figure([go.Bar(x=[f"{v}:{s}-{e}" for v, s, e in zip(top["view"], top["start_bin"], top["end_bin"], strict=True)], y=top["score_delta"], marker_color=VIOLET)]).update_layout(template="plotly_dark", title="Most influential occluded regions", xaxis_title="View and bins", yaxis_title="Base score - occluded score"),
            width="stretch",
        )
        right.dataframe(top, width="stretch", hide_index=True)
        inference = infer_ann02(ann02, sample)
        maps = inference["feature_maps"]
        if maps:
            map_name = right.selectbox("Convolutional activation map", sorted(maps), key="stage07_xai_map")
            values = np.asarray(maps[map_name])
            right.plotly_chart(
                go.Figure([go.Heatmap(z=values, colorscale=[[0.0, "#070B1D"], [0.5, BLUE], [1.0, CYAN]])]).update_layout(template="plotly_dark", title=f"Real activations · {map_name}", xaxis_title="Signal position", yaxis_title="Filter"),
                width="stretch",
            )
        for limitation in explanation["limitations"]:
            right.caption(f"• {limitation}")
    else:
        right.info("Run the bounded sensitivity calculation to inspect ANN-02 regions.")


def render_research_experiments() -> None:
    """Render controlled ANN-02 campaigns and the isolated TESS technical pilot."""
    st.subheader("Research Experiments")
    st.write("Local, versioned experiments. Training uses train/validation only; TESS data remain isolated from Kepler models.")
    campaign_path = DEFAULT_STAGE07_EXPERIMENTS / "latest_campaign.json"
    campaign = _json(campaign_path)
    if st.button("Run controlled ANN-02 campaign", key="stage07_campaign"):
        with st.spinner("Training versioned CPU trials on train/validation only…"):
            run_ann02_campaign()
        st.rerun()
    if campaign is None:
        st.info("No Stage 07 ANN-02 campaign has been executed yet.")
    else:
        st.success(f"Selected trial: {campaign['selected_trial']} · {campaign['selected_experiment_id']}")
        rows = []
        for run in campaign["runs"]:
            metrics = run["validation_metrics"]
            rows.append({"Trial": run["name"], "Changed factor": run["changed_factor"], "Parameters": run["architecture"]["trainable_parameters"], "Best epoch": run["best_epoch"], "Accuracy": metrics["accuracy"], "ROC-AUC": metrics["roc_auc"], "PR-AUC": metrics["pr_auc"], "BCE": metrics["binary_cross_entropy"]})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        loss_figure = go.Figure()
        accuracy_figure = go.Figure()
        for run in campaign["runs"]:
            artifact = _json(
                DEFAULT_STAGE07_EXPERIMENTS / "runs" / f"{run['experiment_id']}.json"
            )
            if artifact is None:
                continue
            history = pd.DataFrame(artifact.get("history", []))
            if history.empty:
                continue
            loss_figure.add_scatter(
                x=history["epoch"],
                y=history["validation_loss"],
                mode="lines",
                name=run["name"],
            )
            accuracy_figure.add_scatter(
                x=history["epoch"],
                y=history["validation_accuracy"],
                mode="lines",
                name=run["name"],
            )
        charts = st.columns(2)
        charts[0].plotly_chart(
            loss_figure.update_layout(
                template="plotly_dark",
                title="Validation loss by controlled trial",
                xaxis_title="Epoch",
                yaxis_title="Binary cross-entropy",
            ),
            width="stretch",
        )
        charts[1].plotly_chart(
            accuracy_figure.update_layout(
                template="plotly_dark",
                title="Validation accuracy by controlled trial",
                xaxis_title="Epoch",
                yaxis_title="Accuracy",
            ),
            width="stretch",
        )
        for limitation in campaign["limitations"]:
            st.caption(f"• {limitation}")

    st.markdown("#### TESS integration study")
    compatibility = tess_compatibility_report()
    st.dataframe(pd.DataFrame(compatibility["differences"]), width="stretch", hide_index=True)
    if st.button("Acquire one bounded TESS SPOC product", key="stage07_tess"):
        with st.spinner("Querying MAST and downloading at most one TESS light curve…"):
            run_tess_pilot(download=True)
        st.rerun()
    tess = _json(DEFAULT_TESS_REPORT)
    if tess is not None:
        observation = tess.get("observation", {})
        columns = st.columns(4)
        columns[0].metric("TESS target", tess["target"])
        columns[1].metric("Sector", observation.get("sector", tess["sector_requested"]))
        columns[2].metric("Cadence", f"{observation.get('cadence_seconds_median', tess['exptime_requested_seconds']):.1f} s")
        columns[3].metric("Records", observation.get("records", "search only"))
        st.warning("This isolated TESS product is a compatibility study and is not part of ANN-02 training.")
