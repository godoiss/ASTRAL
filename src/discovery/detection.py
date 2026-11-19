"""Astropy Box Least Squares detector and transparent signal diagnostics."""

from __future__ import annotations

import math

import numpy as np
from astropy.timeseries import BoxLeastSquares

from src.discovery.config import DetectionConfig
from src.discovery.models import AliasDiagnostic, DetectionResult, ProcessedLightCurve
from src.discovery.processing import robust_noise


class DetectionError(RuntimeError):
    """Raised when a light curve cannot support the configured BLS search."""


_HARMONICS = {
    0.5: "half-period",
    1.0 / 3.0: "one-third-period",
    2.0: "double-period",
    3.0: "triple-period",
}


def _harmonic_relation(ratio: float, tolerance: float) -> str | None:
    for expected, label in _HARMONICS.items():
        if abs(ratio - expected) <= tolerance * expected:
            return label
    return None


def _alias_peaks(
    periods: np.ndarray,
    powers: np.ndarray,
    primary_period: float,
    config: DetectionConfig,
) -> list[AliasDiagnostic]:
    maxima = np.flatnonzero(
        (powers[1:-1] > powers[:-2]) & (powers[1:-1] >= powers[2:])
    ) + 1
    ordered = maxima[np.argsort(powers[maxima])[::-1]] if maxima.size else np.array([], dtype=int)
    aliases: list[AliasDiagnostic] = []
    for index in ordered:
        period = float(periods[index])
        if abs(period - primary_period) / primary_period < config.alias_ratio_tolerance:
            continue
        ratio = period / primary_period
        aliases.append(
            AliasDiagnostic(
                period_days=period,
                depth_snr=float(powers[index]),
                ratio_to_primary=ratio,
                relation=_harmonic_relation(ratio, config.alias_ratio_tolerance),
            )
        )
        if len(aliases) >= config.max_reported_aliases:
            break
    return aliases


def _expected_epochs(start: float, end: float, epoch: float, period: float) -> np.ndarray:
    first = math.ceil((start - epoch) / period)
    last = math.floor((end - epoch) / period)
    return epoch + np.arange(first, last + 1, dtype=np.float64) * period


def _event_diagnostics(
    series: ProcessedLightCurve,
    period: float,
    epoch: float,
    duration_days: float,
    config: DetectionConfig,
) -> tuple[list[float], list[float], int, int, float | None, float | None, float | None]:
    epochs = _expected_epochs(series.time_bkjd.min(), series.time_bkjd.max(), epoch, period)
    depths: list[float] = []
    errors: list[float] = []
    event_indices: list[int] = []
    noise = robust_noise(series.detrended_flux - np.median(series.detrended_flux))
    for event_index, event_time in enumerate(epochs):
        distance = np.abs(series.time_bkjd - event_time)
        in_event = distance <= duration_days / 2.0
        local_baseline = (distance > duration_days) & (distance <= 2.5 * duration_days)
        points = int(np.sum(in_event))
        if points < config.minimum_points_per_event or int(np.sum(local_baseline)) < 3:
            continue
        baseline = float(np.median(series.detrended_flux[local_baseline]))
        event_flux = float(np.median(series.detrended_flux[in_event]))
        depths.append(baseline - event_flux)
        errors.append(float(noise / math.sqrt(max(points, 1))))
        event_indices.append(event_index)
    odd = [depth for depth, index in zip(depths, event_indices, strict=True) if index % 2 == 0]
    even = [depth for depth, index in zip(depths, event_indices, strict=True) if index % 2 == 1]
    odd_depth = float(np.median(odd)) if odd else None
    even_depth = float(np.median(even)) if even else None
    odd_even_sigma: float | None = None
    if odd and even and np.isfinite(noise):
        uncertainty = noise * math.sqrt(1.0 / len(odd) + 1.0 / len(even))
        if uncertainty > 0:
            odd_even_sigma = abs(odd_depth - even_depth) / uncertainty
    return (
        depths,
        errors,
        len(epochs),
        len(depths),
        odd_depth,
        even_depth,
        odd_even_sigma,
    )


def _binned_phase(
    phase: np.ndarray,
    flux: np.ndarray,
    bins: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    edges = np.linspace(-0.5, 0.5, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    output = np.full(bins, np.nan, dtype=np.float64)
    indices = np.searchsorted(edges, phase, side="right") - 1
    valid = (indices >= 0) & (indices < bins)
    for index in np.unique(indices[valid]):
        output[index] = float(np.median(flux[valid & (indices == index)]))
    return centers, output, float(np.mean(np.isfinite(output)))


def detect_periodic_signal(
    series: ProcessedLightCurve,
    config: DetectionConfig,
) -> DetectionResult:
    """Search one real or controlled light curve for a periodic box-shaped dip.

    The returned depth SNR and ranking score are screening diagnostics. They are
    not calibrated false-alarm probabilities or validated statistical significance.
    """
    baseline = float(series.time_bkjd.max() - series.time_bkjd.min())
    maximum_period = min(config.maximum_period_days, baseline / 2.0)
    if maximum_period <= config.minimum_period_days:
        raise DetectionError("observation baseline is too short for the configured period range")
    frequencies = np.linspace(
        1.0 / maximum_period,
        1.0 / config.minimum_period_days,
        config.period_samples,
        dtype=np.float64,
    )
    periods = np.sort(1.0 / frequencies)
    durations = np.asarray(config.durations_hours, dtype=np.float64) / 24.0
    model = BoxLeastSquares(
        series.time_bkjd,
        series.detrended_flux,
        dy=series.normalized_error,
    )
    periodogram = model.power(
        periods,
        durations,
        objective="snr",
        method="fast",
        oversample=config.bls_oversample,
    )
    powers = np.asarray(periodogram.power, dtype=np.float64)
    if not np.any(np.isfinite(powers)):
        raise DetectionError("BLS returned no finite periodogram values")
    best = int(np.nanargmax(powers))
    period = float(periodogram.period[best])
    duration_days = float(periodogram.duration[best])
    epoch = float(periodogram.transit_time[best])
    depth = float(periodogram.depth[best])
    depth_error = float(periodogram.depth_err[best])
    depth_snr = float(periodogram.depth_snr[best])

    depths, _event_errors, expected, observed, odd, even, odd_even_sigma = _event_diagnostics(
        series, period, epoch, duration_days, config
    )
    secondary_depths, secondary_errors, _, _, _, _, _ = _event_diagnostics(
        series, period, epoch + 0.5 * period, duration_days, config
    )
    secondary_depth = (
        float(np.median(secondary_depths)) if len(secondary_depths) >= 2 else None
    )
    secondary_sigma: float | None = None
    if secondary_depth is not None and secondary_errors:
        secondary_uncertainty = float(np.median(secondary_errors)) / math.sqrt(
            len(secondary_errors)
        )
        if secondary_uncertainty > 0:
            secondary_sigma = secondary_depth / secondary_uncertainty
    event_coverage = observed / max(expected, 1)
    phase = ((series.time_bkjd - epoch + 0.5 * period) % period) / period - 0.5
    phase_order = np.argsort(phase)
    phase = phase[phase_order]
    phase_flux = series.detrended_flux[phase_order]
    centers, binned_flux, phase_coverage = _binned_phase(
        phase, phase_flux, config.phase_bins
    )
    event_mad = (
        float(np.median(np.abs(np.asarray(depths) - np.median(depths))))
        if len(depths) >= 2
        else None
    )
    possible_binary = bool(
        (odd_even_sigma is not None and odd_even_sigma >= config.odd_even_sigma_threshold)
        or (
            secondary_sigma is not None
            and secondary_sigma >= config.odd_even_sigma_threshold
            and secondary_depth is not None
            and secondary_depth > 0.1 * max(depth, np.finfo(float).eps)
        )
        or depth >= 0.05
    )
    aliases = _alias_peaks(periods, powers, period, config)
    alias_penalty = 0.8 if any(item.relation is not None for item in aliases[:3]) else 1.0
    binary_penalty = 0.6 if possible_binary else 1.0
    ranking_score = depth_snr * event_coverage * phase_coverage * alias_penalty * binary_penalty
    detected = (
        depth > 0
        and depth_snr >= config.minimum_depth_snr
        and observed >= config.minimum_observed_events
        and event_coverage >= config.minimum_event_coverage
    )
    left_spacing = period - float(periods[max(best - 1, 0)])
    right_spacing = float(periods[min(best + 1, periods.size - 1)]) - period
    grid_uncertainty = 0.5 * max(left_spacing, right_spacing)
    event_uncertainty = duration_days / max(depth_snr * math.sqrt(max(observed, 1)), 1.0)
    period_uncertainty = max(grid_uncertainty, event_uncertainty)
    duration_steps = np.diff(np.sort(durations))
    duration_uncertainty = (
        float(np.min(duration_steps) * 12.0) if duration_steps.size else duration_days * 12.0
    )
    transit_uncertainty = duration_days / max(2.0 * depth_snr, 1.0)
    limitations = [
        "BLS assumes a periodic box-shaped event and can favor stellar or instrumental signals.",
        "Depth SNR is a screening diagnostic, not a calibrated false-alarm probability.",
        "Only the configured period and duration grid was searched.",
        "Independent quarters and pixel-level contamination were not validated here.",
        "Reported uncertainties are grid/noise approximations, not posterior credible intervals.",
    ]
    search_edge_tolerance = max(left_spacing, right_spacing) * 1.5
    if (
        abs(period - float(periods[0])) <= search_edge_tolerance
        or abs(period - float(periods[-1])) <= search_edge_tolerance
    ):
        limitations.append(
            "The strongest peak lies on a configured period-search boundary and is especially inconclusive."
        )
    if observed < config.minimum_observed_events:
        limitations.append("Too few individual events were covered for the configured screen.")
    if possible_binary:
        limitations.append("Depth or odd/even behavior is compatible with an eclipsing binary hypothesis.")
    return DetectionResult(
        status="screening_detection" if detected else "below_screening_threshold",
        period_days=period,
        period_uncertainty_days=period_uncertainty,
        transit_time_bkjd=epoch,
        transit_time_uncertainty_days=transit_uncertainty,
        duration_hours=duration_days * 24.0,
        duration_uncertainty_hours=duration_uncertainty,
        depth_fraction=depth,
        depth_ppm=depth * 1_000_000.0,
        depth_error_fraction=depth_error,
        depth_snr=depth_snr,
        ranking_score=float(ranking_score),
        expected_events=expected,
        observed_events=observed,
        event_coverage=float(event_coverage),
        phase_coverage=phase_coverage,
        event_depths=[float(value) for value in depths],
        event_depth_mad=event_mad,
        odd_depth=odd,
        even_depth=even,
        odd_even_sigma=odd_even_sigma,
        secondary_depth=secondary_depth,
        secondary_sigma=secondary_sigma,
        possible_eclipsing_binary=possible_binary,
        aliases=aliases,
        limitations=limitations,
        periods=periods,
        powers=powers,
        phase=phase,
        phase_flux=phase_flux,
        phase_bin_centers=centers,
        phase_bin_flux=binned_flux,
    )
