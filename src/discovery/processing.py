"""Catalog-independent quality filtering and detrending for discovery searches."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.discovery.config import DetectionConfig
from src.discovery.models import ProcessedLightCurve, RawLightCurve


class DiscoveryProcessingError(RuntimeError):
    """Raised when observed cadences cannot produce a searchable series."""


def robust_noise(values: np.ndarray) -> float:
    """Estimate Gaussian-equivalent scatter from the median absolute deviation."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 3:
        return float("nan")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return max(1.4826 * mad, np.finfo(np.float64).eps)


def _odd_window(window: int, records: int) -> int:
    value = min(max(window, 11), records if records % 2 else records - 1)
    return max(value if value % 2 else value - 1, 3)


def process_discovery_light_curve(
    raw: RawLightCurve,
    config: DetectionConfig,
) -> ProcessedLightCurve:
    """Filter observed cadences and remove slow trends without catalog ephemerides.

    A centered rolling median is evaluated only at observed cadences. It estimates
    the slow baseline but does not create samples inside temporal gaps.
    """
    finite = (
        np.isfinite(raw.time_bkjd)
        & np.isfinite(raw.flux)
        & np.isfinite(raw.flux_error)
        & (raw.flux > 0)
    )
    quality_ok = (raw.quality.astype(np.int64) & config.quality_bitmask) == 0
    keep = finite & quality_ok
    if int(np.sum(keep)) < 200:
        raise DiscoveryProcessingError("fewer than 200 cadences survive quality filtering")

    time = raw.time_bkjd[keep].astype(np.float64)
    flux = raw.flux[keep].astype(np.float64)
    error = raw.flux_error[keep].astype(np.float64)
    segment = raw.segment[keep].astype(np.int64)
    order = np.argsort(time)
    time, flux, error, segment = time[order], flux[order], error[order], segment[order]

    normalized = np.empty_like(flux)
    detrended = np.empty_like(flux)
    normalized_error = np.empty_like(error)
    segment_diagnostics: list[dict[str, float | int | str]] = []
    for segment_id in np.unique(segment):
        indices = np.flatnonzero(segment == segment_id)
        segment_flux = flux[indices]
        segment_error = error[indices]
        center = float(np.median(segment_flux))
        if not np.isfinite(center) or center <= 0:
            raise DiscoveryProcessingError(f"segment {segment_id} has an invalid flux median")
        values = segment_flux / center
        errors = segment_error / center
        cadence_days = float(np.median(np.diff(time[indices]))) if indices.size > 1 else 0.0204
        requested_window = int(round(config.detrend_window_days / max(cadence_days, 1e-6)))
        window = _odd_window(requested_window, indices.size)
        if indices.size < 11 or window < 5:
            trend = np.full_like(values, float(np.median(values)))
            method = "segment_median"
        else:
            trend = (
                pd.Series(values)
                .rolling(window=window, center=True, min_periods=max(5, window // 4))
                .median()
                .to_numpy(dtype=np.float64)
            )
            missing_trend = ~np.isfinite(trend) | (trend <= 0)
            trend[missing_trend] = float(np.median(values))
            method = "centered_rolling_median"
        normalized[indices] = values
        detrended[indices] = values / trend
        normalized_error[indices] = errors / trend
        segment_diagnostics.append(
            {
                "segment": int(segment_id),
                "records": int(indices.size),
                "cadence_days": cadence_days,
                "detrend_window_cadences": int(window),
                "detrending": method,
            }
        )

    searchable = (
        np.isfinite(time)
        & np.isfinite(detrended)
        & np.isfinite(normalized_error)
        & (normalized_error > 0)
    )
    time = time[searchable]
    normalized = normalized[searchable]
    detrended = detrended[searchable]
    normalized_error = normalized_error[searchable]
    segment = segment[searchable]
    if time.size < 200:
        raise DiscoveryProcessingError("fewer than 200 finite detrended cadences remain")

    residual_noise = robust_noise(detrended - np.median(detrended))
    if not np.isfinite(residual_noise):
        raise DiscoveryProcessingError("robust residual noise could not be estimated")
    conservative_error = np.maximum(normalized_error, residual_noise)
    diagnostics = {
        "raw_records": int(raw.time_bkjd.size),
        "finite_records": int(np.sum(finite)),
        "quality_rejected": int(np.sum(finite & ~quality_ok)),
        "retained_records": int(time.size),
        "retention_fraction": float(time.size / max(raw.time_bkjd.size, 1)),
        "observation_start_bkjd": float(time.min()),
        "observation_end_bkjd": float(time.max()),
        "observation_span_days": float(time.max() - time.min()),
        "robust_flux_noise": residual_noise,
        "catalog_ephemeris_used": False,
        "gap_policy": "no synthetic cadences; rolling baseline evaluated at observed times only",
        "segments": segment_diagnostics,
    }
    return ProcessedLightCurve(
        kic=raw.kic,
        time_bkjd=time,
        normalized_flux=normalized,
        detrended_flux=detrended,
        normalized_error=conservative_error,
        segment=segment,
        diagnostics=diagnostics,
    )
