"""Interactive Plotly views for real stellar luminosity observations."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from src.observatory.models import LightCurveArtifact, NeuralLightCurveExample


BACKGROUND = "#050816"
PANEL = "#0A1028"
GRID = "#1B2A4A"
TEXT = "#E8F7FF"
MUTED = "#8EA6C9"
CYAN = "#00E5FF"
BLUE = "#247BFF"
VIOLET = "#9D5CFF"


def _layout(figure: go.Figure, title: str, x_title: str, y_title: str) -> go.Figure:
    figure.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=PANEL,
        font={"color": TEXT, "family": "Arial, sans-serif"},
        hovermode="closest",
        margin={"l": 65, "r": 25, "t": 65, "b": 55},
        legend={"bgcolor": "rgba(5,8,22,0.7)", "bordercolor": GRID, "borderwidth": 1},
    )
    figure.update_xaxes(title=x_title, gridcolor=GRID, zerolinecolor=GRID, color=MUTED)
    figure.update_yaxes(title=y_title, gridcolor=GRID, zerolinecolor=GRID, color=MUTED)
    return figure


def original_flux_figure(artifact: LightCurveArtifact) -> go.Figure:
    """Plot unmodified FITS flux values by observation segment."""
    figure = go.Figure()
    for segment_index, name in enumerate(artifact.quarters):
        mask = (
            (artifact.raw_segment == segment_index)
            & np.isfinite(artifact.raw_time_bkjd)
            & np.isfinite(artifact.raw_flux)
        )
        figure.add_trace(
            go.Scattergl(
                x=artifact.raw_time_bkjd[mask],
                y=artifact.raw_flux[mask],
                mode="markers",
                marker={"size": 3, "color": CYAN, "opacity": 0.58},
                name=name,
                customdata=artifact.raw_quality[mask],
                hovertemplate="BKJD %{x:.5f}<br>Flux %{y:.3f}<br>Quality %{customdata}<extra></extra>",
            )
        )
    return _layout(figure, "Original MAST Luminosity", "Time (BKJD)", "PDCSAP flux")


def processed_flux_figure(artifact: LightCurveArtifact) -> go.Figure:
    """Compare normalized observations with transit-preserving detrending."""
    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=artifact.clean_time_bkjd,
            y=artifact.normalized_flux,
            mode="markers",
            marker={"size": 3, "color": VIOLET, "opacity": 0.34},
            name="Normalized",
        )
    )
    figure.add_trace(
        go.Scattergl(
            x=artifact.clean_time_bkjd,
            y=artifact.detrended_flux,
            mode="markers",
            marker={"size": 3, "color": CYAN, "opacity": 0.65},
            name="Detrended",
        )
    )
    return _layout(figure, "Processed Luminosity", "Time (BKJD)", "Relative flux")


def phase_folded_figure(
    artifact: LightCurveArtifact,
    example: NeuralLightCurveExample,
) -> go.Figure:
    """Plot the full orbital phase and observed median bins."""
    order = np.argsort(artifact.phase)
    observed = example.global_mask
    figure = go.Figure(
        go.Scattergl(
            x=artifact.phase[order],
            y=artifact.detrended_flux[order],
            mode="markers",
            marker={"size": 3, "color": BLUE, "opacity": 0.34},
            name="Observed cadences",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=example.global_coordinates[observed],
            y=example.global_flux[observed],
            mode="lines",
            line={"color": CYAN, "width": 2},
            name="Observed-bin median",
        )
    )
    figure.add_vline(x=0.0, line_color=VIOLET, line_dash="dash", opacity=0.8)
    return _layout(figure, "Global Orbital View", "Orbital phase", "Detrended relative flux")


def transit_zoom_figure(
    artifact: LightCurveArtifact,
    example: NeuralLightCurveExample,
) -> go.Figure:
    """Plot only the catalog-centered local transit window."""
    order = np.argsort(artifact.local_phase_hours)
    observed = example.local_mask
    figure = go.Figure(
        go.Scattergl(
            x=artifact.local_phase_hours[order],
            y=artifact.local_flux[order],
            mode="markers",
            marker={"size": 4, "color": VIOLET, "opacity": 0.42},
            name="Observed cadences",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=example.local_coordinates_hours[observed],
            y=example.local_flux[observed],
            mode="lines+markers",
            marker={"size": 4},
            line={"color": CYAN, "width": 2},
            name="Observed-bin median",
        )
    )
    figure.add_vline(x=0.0, line_color=VIOLET, line_dash="dash", opacity=0.8)
    return _layout(figure, "Transit-Centered Local View", "Hours from catalog epoch", "Detrended relative flux")
