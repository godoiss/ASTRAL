"""Transit-preserving processing for real Kepler light curves."""

from __future__ import annotations

import warnings
from typing import Any

import astropy.units as u
import numpy as np
from astropy.time import Time
from lightkurve.utils import LightkurveWarning

from src.observatory.config import ObservatoryConfig
from src.observatory.models import FloatArray, IntArray, KoiMetadata, LightCurveArtifact


BKJD_OFFSET = 2454833.0


class SignalProcessingError(RuntimeError):
    """Raised when real observations cannot yield a finite processed series."""


def bkjd_to_bjd(values: float | FloatArray) -> float | FloatArray:
    """Convert Kepler BKJD to full Barycentric Julian Date."""
    converted = np.asarray(values, dtype=np.float64) + BKJD_OFFSET
    return float(converted) if converted.ndim == 0 else converted


def bjd_to_bkjd(values: float | FloatArray) -> float | FloatArray:
    """Convert Barycentric Julian Date to Kepler BKJD."""
    converted = np.asarray(values, dtype=np.float64) - BKJD_OFFSET
    return float(converted) if converted.ndim == 0 else converted


def _numeric(values: Any, dtype: Any = np.float64) -> np.ndarray:
    data = getattr(values, "value", values)
    return np.asarray(data, dtype=dtype)


def _time_bkjd(light_curve: Any) -> FloatArray:
    time = light_curve.time
    if getattr(time, "format", None) == "bkjd":
        return np.asarray(time.value, dtype=np.float64)
    return np.asarray(time.tdb.jd - BKJD_OFFSET, dtype=np.float64)


def _segment_name(light_curve: Any, index: int) -> str:
    quarter = light_curve.meta.get("QUARTER")
    sector = light_curve.meta.get("SECTOR")
    if quarter is not None:
        return f"Kepler Q{int(quarter):02d}"
    if sector is not None:
        return f"TESS Sector {int(sector)}"
    return f"Segment {index + 1}"


def _safe_window(config: ObservatoryConfig, records: int) -> int | None:
    maximum = records if records % 2 == 1 else records - 1
    window = min(config.flatten_window_length, maximum)
    return None if window <= config.flatten_polyorder + 1 else window


def process_light_curves(
    collection: Any,
    metadata: KoiMetadata,
    config: ObservatoryConfig,
    provenance: dict[str, Any],
) -> LightCurveArtifact:
    """Clean quality flags, normalize, detrend, unite segments, and phase-fold."""
    raw_time_parts: list[FloatArray] = []
    raw_flux_parts: list[FloatArray] = []
    raw_error_parts: list[FloatArray] = []
    raw_quality_parts: list[IntArray] = []
    raw_segment_parts: list[IntArray] = []
    clean_time_parts: list[FloatArray] = []
    normalized_parts: list[FloatArray] = []
    detrended_parts: list[FloatArray] = []
    segment_records: list[dict[str, Any]] = []
    segment_names: list[str] = []

    transit_epoch = Time(
        float(bkjd_to_bjd(metadata.epoch_bkjd)),
        format="jd",
        scale="tdb",
    )
    transit_duration = (
        metadata.duration_hours
        * config.transit_mask_duration_multiplier
        / 24.0
        * u.day
    )

    for segment_index, light_curve in enumerate(collection):
        time_bkjd = _time_bkjd(light_curve)
        flux = _numeric(light_curve.flux)
        flux_error = _numeric(light_curve.flux_err)
        quality = _numeric(light_curve.quality, np.int64)
        segment_name = _segment_name(light_curve, segment_index)
        segment_names.append(segment_name)

        raw_time_parts.append(time_bkjd)
        raw_flux_parts.append(flux)
        raw_error_parts.append(flux_error)
        raw_quality_parts.append(quality)
        raw_segment_parts.append(np.full(time_bkjd.size, segment_index, dtype=np.int64))

        finite = np.isfinite(time_bkjd) & np.isfinite(flux)
        quality_ok = (quality & config.quality_bitmask) == 0
        retained = finite & quality_ok
        if int(np.sum(retained)) < config.flatten_polyorder + 3:
            segment_records.append(
                {
                    "segment": segment_name,
                    "raw_records": int(time_bkjd.size),
                    "retained_records": int(np.sum(retained)),
                    "status": "insufficient_after_quality_filter",
                }
            )
            continue

        clean = light_curve[retained].remove_nans()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LightkurveWarning)
            normalized = clean.normalize(unit="unscaled")
        transit_mask = normalized.create_transit_mask(
            period=metadata.period_days * u.day,
            transit_time=transit_epoch,
            duration=transit_duration,
        )
        window = _safe_window(config, len(normalized))
        if window is None:
            flattened = normalized.copy()
            detrending = "skipped_short_segment"
        else:
            flattened = normalized.flatten(
                window_length=window,
                polyorder=config.flatten_polyorder,
                break_tolerance=5,
                niters=3,
                sigma=3,
                mask=transit_mask,
            )
            detrending = "savitzky_golay_transit_masked"

        clean_time = _time_bkjd(normalized)
        normalized_flux = _numeric(normalized.flux)
        detrended_flux = _numeric(flattened.flux)
        valid_processed = (
            np.isfinite(clean_time)
            & np.isfinite(normalized_flux)
            & np.isfinite(detrended_flux)
        )
        clean_time_parts.append(clean_time[valid_processed])
        normalized_parts.append(normalized_flux[valid_processed])
        detrended_parts.append(detrended_flux[valid_processed])
        segment_records.append(
            {
                "segment": segment_name,
                "raw_records": int(time_bkjd.size),
                "retained_records": int(np.sum(valid_processed)),
                "quality_rejected": int(np.sum(finite & ~quality_ok)),
                "non_finite_rejected": int(np.sum(~finite)),
                "detrending": detrending,
                "flatten_window_length": window,
            }
        )

    if not clean_time_parts:
        raise SignalProcessingError("No segment survived finite and quality filtering")

    raw_time = np.concatenate(raw_time_parts)
    raw_flux = np.concatenate(raw_flux_parts)
    raw_error = np.concatenate(raw_error_parts)
    raw_quality = np.concatenate(raw_quality_parts)
    raw_segment = np.concatenate(raw_segment_parts)
    clean_time = np.concatenate(clean_time_parts)
    normalized_flux = np.concatenate(normalized_parts)
    detrended_flux = np.concatenate(detrended_parts)
    order = np.argsort(clean_time)
    clean_time = clean_time[order]
    normalized_flux = normalized_flux[order]
    detrended_flux = detrended_flux[order]

    period = metadata.period_days
    phase = ((clean_time - metadata.epoch_bkjd + 0.5 * period) % period) / period - 0.5
    local_half_width_hours = max(
        metadata.duration_hours * config.local_view_duration_multiplier,
        3.0,
    )
    phase_hours = phase * period * 24.0
    local_mask = np.abs(phase_hours) <= local_half_width_hours
    local_phase_hours = phase_hours[local_mask]
    local_flux = detrended_flux[local_mask]
    if not local_phase_hours.size:
        raise SignalProcessingError("No observed cadence falls inside the transit window")

    raw_finite_time = raw_time[np.isfinite(raw_time)]
    cadence_seconds = float(np.nanmedian(np.diff(clean_time)) * 86400.0)
    provenance["processing"] = {
        "quality_bitmask": config.quality_bitmask,
        "missing_values": "removed; no gap filling",
        "normalization": "per-segment Lightkurve median normalization",
        "detrending": "Lightkurve Savitzky-Golay with catalog transit mask",
        "phase_definition": "((BKJD - epoch + period/2) mod period) / period - 0.5",
        "segments": segment_records,
        "raw_records": int(raw_time.size),
        "processed_records": int(clean_time.size),
        "local_records": int(local_phase_hours.size),
    }
    return LightCurveArtifact(
        metadata=metadata,
        mission=config.mission,
        author=config.author,
        cadence_seconds=cadence_seconds,
        observation_start_bkjd=float(np.min(raw_finite_time)),
        observation_end_bkjd=float(np.max(raw_finite_time)),
        quarters=tuple(segment_names),
        raw_time_bkjd=raw_time,
        raw_flux=raw_flux,
        raw_flux_error=raw_error,
        raw_quality=raw_quality,
        raw_segment=raw_segment,
        clean_time_bkjd=clean_time,
        normalized_flux=normalized_flux,
        detrended_flux=detrended_flux,
        phase=phase,
        local_phase_hours=local_phase_hours,
        local_flux=local_flux,
        provenance=provenance,
    )
