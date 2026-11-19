"""Streamlit presentation for real, bounded Stage 10 discovery campaigns."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.discovery.acquisition import load_or_acquire_raw
from src.discovery.campaign import CAMPAIGN_ROOT, PROJECT_ROOT, load_campaign, run_campaign
from src.discovery.catalog import select_exploration_targets
from src.discovery.config import DiscoveryConfig
from src.observatory.config import ObservatoryConfig


ATLAS_CYAN = "#70DBF7"
ATLAS_AMBER = "#EFB681"
ATLAS_VIOLET = "#BFA7EF"
ATLAS_CANVAS = "#070D17"


@st.cache_data(show_spinner=False)
def _planned_targets(config_payload: str) -> list[dict[str, Any]]:
    config = DiscoveryConfig.from_dict(json.loads(config_payload))
    return [item.to_dict() for item in select_exploration_targets(config.campaign)]


def _campaign_paths() -> list[Path]:
    if not CAMPAIGN_ROOT.is_dir():
        return []
    return sorted(CAMPAIGN_ROOT.glob("*/campaign.json"), reverse=True)


def _safe_artifact(relative: str) -> Path:
    root = PROJECT_ROOT.resolve()
    path = (PROJECT_ROOT / relative).resolve()
    if root not in path.parents:
        raise ValueError("campaign artifact points outside the project")
    return path


def _figure(title: str, x_title: str, y_title: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor=ATLAS_CANVAS,
        plot_bgcolor=ATLAS_CANVAS,
        height=360,
        margin={"l": 55, "r": 25, "t": 60, "b": 50},
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend={"orientation": "h"},
    )
    return figure


def _raw_figure(kic: int) -> go.Figure:
    raw = load_or_acquire_raw(kic, ObservatoryConfig(), allow_network=False)
    finite = np.isfinite(raw.time_bkjd) & np.isfinite(raw.flux)
    indices = np.flatnonzero(finite)
    if indices.size > 20_000:
        indices = indices[np.linspace(0, indices.size - 1, 20_000).astype(int)]
    figure = _figure("Observed Kepler luminosity", "Time [BKJD]", "PDCSAP flux")
    figure.add_trace(
        go.Scattergl(
            x=raw.time_bkjd[indices],
            y=raw.flux[indices],
            mode="markers",
            marker={"size": 3, "color": "#B5C4D0", "opacity": 0.55},
            name="Observed cadence",
        )
    )
    return figure


def _processed_figure(arrays: dict[str, np.ndarray]) -> go.Figure:
    figure = _figure("Catalog-independent detrended luminosity", "Time [BKJD]", "Relative flux")
    figure.add_trace(
        go.Scattergl(
            x=arrays["time_bkjd"],
            y=arrays["detrended_flux"],
            mode="markers",
            marker={"size": 3, "color": ATLAS_CYAN, "opacity": 0.6},
            name="Detrended observed cadence",
        )
    )
    return figure


def _periodogram_figure(arrays: dict[str, np.ndarray], period: float) -> go.Figure:
    figure = _figure("Box Least Squares periodogram", "Trial period [days]", "Depth SNR diagnostic")
    figure.add_trace(
        go.Scatter(
            x=arrays["periods"],
            y=arrays["powers"],
            mode="lines",
            line={"color": ATLAS_CYAN, "width": 1.5},
            name="BLS grid",
        )
    )
    figure.add_vline(x=period, line_color=ATLAS_AMBER, line_dash="dash")
    return figure


def _phase_figure(arrays: dict[str, np.ndarray]) -> go.Figure:
    figure = _figure("Phase-folded screening signal", "Orbital phase", "Relative flux")
    figure.add_trace(
        go.Scattergl(
            x=arrays["phase"],
            y=arrays["phase_flux"],
            mode="markers",
            marker={"size": 3, "color": "#8295A5", "opacity": 0.35},
            name="Observed cadences",
        )
    )
    observed = np.isfinite(arrays["phase_bin_flux"])
    figure.add_trace(
        go.Scatter(
            x=arrays["phase_bin_centers"][observed],
            y=arrays["phase_bin_flux"][observed],
            mode="lines+markers",
            line={"color": ATLAS_VIOLET, "width": 2},
            marker={"size": 5},
            name="Median phase bins",
        )
    )
    return figure


def _render_configuration(default: DiscoveryConfig) -> DiscoveryConfig:
    with st.expander("Expedition configuration", expanded=False):
        row = st.columns(4)
        sample_size = row[0].select_slider(
            "Kepler stars", options=[1, 5, 10, 15, 20, 25], value=default.campaign.sample_size
        )
        minimum_period = row[1].number_input(
            "Minimum period [days]", 0.5, 10.0, default.detection.minimum_period_days, 0.5
        )
        maximum_floor = float(minimum_period) + 0.5
        maximum_period = row[2].number_input(
            "Maximum period [days]",
            maximum_floor,
            60.0,
            max(default.detection.maximum_period_days, maximum_floor),
            0.5,
        )
        minimum_snr = row[3].number_input(
            "Screening depth SNR", 3.0, 30.0, default.detection.minimum_depth_snr, 0.5
        )
        resource = st.columns(3)
        allow_network = resource[0].checkbox(
            "Allow bounded MAST acquisition",
            value=False,
            help="Explicit opt-in. Existing cache is always preferred.",
        )
        max_downloads = resource[1].number_input(
            "Maximum new downloads",
            min_value=0,
            max_value=int(sample_size),
            value=min(default.campaign.max_new_downloads, int(sample_size)) if allow_network else 0,
            disabled=not allow_network,
        )
        period_samples = resource[2].select_slider(
            "BLS period samples", options=[1000, 2000, 4000, 8000], value=4000
        )
        campaign = replace(
            default.campaign,
            sample_size=int(sample_size),
            allow_network=allow_network,
            max_new_downloads=int(max_downloads),
        )
        detection = replace(
            default.detection,
            minimum_period_days=float(minimum_period),
            maximum_period_days=float(maximum_period),
            minimum_depth_snr=float(minimum_snr),
            period_samples=int(period_samples),
        )
        return replace(default, campaign=campaign, detection=detection)


def _render_campaign(report: dict[str, Any]) -> None:
    summary = report.get("summary", {})
    metrics = st.columns(6)
    metrics[0].metric("Investigated", summary.get("investigated_stars", 0))
    metrics[1].metric("BLS screens", summary.get("screening_detections", 0))
    metrics[2].metric("Catalog associated", summary.get("catalog_associated_detections", 0))
    metrics[3].metric("Inconclusive", summary.get("unassociated_inconclusive_detections", 0))
    metrics[4].metric("Failures", summary.get("failed_or_excluded_stars", 0))
    metrics[5].metric("Confirmed discoveries", 0)
    st.caption(
        f"Campaign `{report['campaign_id']}` · {report['status']} · "
        f"{report.get('elapsed_seconds', 0):.1f} s · frozen test accessed: no"
    )
    results = report.get("results", [])
    if not results:
        st.info("This campaign contains no successfully processed observations.")
        if report.get("failures"):
            st.dataframe(pd.DataFrame(report["failures"]), width="stretch", hide_index=True)
        return
    table = pd.DataFrame(
        [
            {
                "KIC": item["target"]["kic"],
                "status": item["detection"]["status"],
                "period_days": item["detection"]["period_days"],
                "duration_hours": item["detection"]["duration_hours"],
                "depth_ppm": item["detection"]["depth_ppm"],
                "depth_snr": item["detection"]["depth_snr"],
                "events": item["detection"]["observed_events"],
                "catalog_relation": item["crossmatch"]["period_association_found"],
                "interpretation": item["interpretation"],
            }
            for item in results
        ]
    ).sort_values("depth_snr", ascending=False)
    st.dataframe(table, width="stretch", hide_index=True)
    selected_kic = st.selectbox("Inspect investigated star", table["KIC"].astype(int).tolist())
    selected = next(item for item in results if int(item["target"]["kic"]) == int(selected_kic))
    detection = selected["detection"]
    details = st.columns(5)
    details[0].metric("Period", f"{detection['period_days']:.6f} d")
    details[1].metric("Duration", f"{detection['duration_hours']:.3f} h")
    details[2].metric("Depth", f"{detection['depth_ppm']:.1f} ppm")
    details[3].metric("Depth SNR", f"{detection['depth_snr']:.2f}")
    details[4].metric("Event coverage", f"{100*detection['event_coverage']:.1f}%")

    processed_path = _safe_artifact(selected["artifacts"]["processed_arrays"])
    detection_path = _safe_artifact(selected["artifacts"]["detection_arrays"])
    with np.load(processed_path, allow_pickle=False) as payload:
        processed = {name: payload[name] for name in payload.files}
    with np.load(detection_path, allow_pickle=False) as payload:
        detection_arrays = {name: payload[name] for name in payload.files}
    try:
        st.plotly_chart(_raw_figure(int(selected_kic)), width="stretch", config={"displaylogo": False})
    except Exception as exc:
        st.warning(f"Raw cached luminosity is unavailable: {exc}")
    st.plotly_chart(_processed_figure(processed), width="stretch", config={"displaylogo": False})
    charts = st.columns(2)
    with charts[0]:
        st.plotly_chart(
            _periodogram_figure(detection_arrays, detection["period_days"]),
            width="stretch",
            config={"displaylogo": False},
        )
    with charts[1]:
        st.plotly_chart(_phase_figure(detection_arrays), width="stretch", config={"displaylogo": False})

    st.warning(
        "Depth SNR and ranking score are detector diagnostics, not calibrated significance or confirmation probability."
    )
    associations = selected.get("catalog_associations", [])
    st.markdown("#### Catalog cross-check")
    if associations:
        st.dataframe(pd.DataFrame(associations), width="stretch", hide_index=True)
    else:
        st.info(
            "No KOI association exists in the local snapshot. This does not establish novelty; the standalone TCE catalog was not queried."
        )
    st.caption(selected["crossmatch"]["limitation"])
    with st.expander("Diagnostics, aliases, evidence, and campaign files"):
        st.json(selected, expanded=False)
        st.code(selected["result_path"])


def render_discovery_lab() -> None:
    """Render planning, execution, evidence, and persisted campaign inspection."""
    st.markdown(
        """
        <section class="atlas-discovery">
          <span>STAGE 10 · REAL ASTRONOMICAL SIGNAL DETECTION</span>
          <h2>Discovery Lab</h2>
          <p>Search real Kepler cadences by host star, measure periodic box-shaped dips, and preserve the evidence before interpretation.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Exploratory instrument only. A screening detection is not a discovery, an unmatched signal is not necessarily new, and no neural model confirms these results."
    )
    default = DiscoveryConfig.load(PROJECT_ROOT / "configs" / "discovery_engine.json")
    config = _render_configuration(default)
    try:
        targets = _planned_targets(json.dumps(config.to_dict(), sort_keys=True))
    except Exception as exc:
        st.error(f"Protected target planning failed: {exc}")
        return
    with st.expander("Review protected expedition plan", expanded=True):
        st.dataframe(pd.DataFrame(targets), width="stretch", hide_index=True)
        cached = sum(bool(item["cache_available"]) for item in targets)
        st.caption(
            f"{len(targets)} unlabeled stars · {cached} already cached · all labeled train/validation/test host stars excluded"
        )
    if st.button("Start bounded Kepler campaign", type="primary", width="stretch"):
        progress = st.progress(0.0, text="Persisting campaign plan…")
        status = st.empty()

        def update(stage: str, index: int, total: int, message: str) -> None:
            progress.progress(index / max(total, 1), text=f"{index}/{total} · {message}")
            status.caption(f"{stage.upper()} · intermediate report persisted")

        try:
            report = run_campaign(config, progress_callback=update)
        except Exception as exc:
            st.error(f"Campaign stopped safely: {exc}")
        else:
            progress.progress(1.0, text="Campaign persisted")
            st.session_state.discovery_campaign_id = report["campaign_id"]
            st.success("Campaign completed. Review detections and failures below.")
            st.rerun()

    paths = _campaign_paths()
    if not paths:
        st.info("No Stage 10 campaign has been persisted yet.")
        return
    labels = {path.parent.name: path for path in paths}
    preferred = st.session_state.get("discovery_campaign_id")
    names = list(labels)
    index = names.index(preferred) if preferred in names else 0
    selected_name = st.selectbox("Persisted campaign", names, index=index)
    report = load_campaign(labels[selected_name])
    if report is not None:
        _render_campaign(report)

    st.markdown("#### Neural compatibility boundary")
    st.caption(
        "ANN-01 requires six KOI catalog parameters that a new BLS signal does not automatically provide. "
        "ANN-02 was trained on a bounded, catalog-derived Q3 distribution. Applying either model to a new detection requires an explicit compatibility and out-of-distribution study; no automatic inference is performed here."
    )
