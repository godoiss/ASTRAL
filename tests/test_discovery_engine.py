"""Scientific and safety tests for the Stage 10 Discovery Engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.discovery.acquisition import load_or_acquire_raw
from src.discovery.catalog import crossmatch_detection, select_exploration_targets
from src.discovery.config import CampaignConfig, DetectionConfig, DiscoveryConfig
from src.discovery.detection import detect_periodic_signal
from src.discovery.models import RawLightCurve
from src.discovery.processing import process_discovery_light_curve
from src.observatory.config import ObservatoryConfig


def _controlled_curve(
    *,
    period: float | None,
    depth: float = 0.01,
    duration_days: float = 0.16,
    seed: int = 17,
) -> RawLightCurve:
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, 80.0, 0.0204, dtype=np.float64)
    normalized = 1.0 + rng.normal(0.0, 0.0008, size=time.size)
    if period is not None:
        distance = np.abs(((time - 0.7 + 0.5 * period) % period) - 0.5 * period)
        normalized[distance <= duration_days / 2.0] -= depth
    return RawLightCurve(
        kic=999999,
        time_bkjd=time,
        flux=normalized * 100_000.0,
        flux_error=np.full(time.size, 80.0),
        quality=np.zeros(time.size, dtype=np.int64),
        segment=np.zeros(time.size, dtype=np.int64),
        provenance={"kind": "controlled_synthetic_test_only"},
    )


def _fast_detection_config(**changes: object) -> DetectionConfig:
    values: dict[str, object] = {
        "minimum_period_days": 1.0,
        "maximum_period_days": 10.0,
        "durations_hours": (2.5, 4.0, 6.0),
        "period_samples": 1600,
        "bls_oversample": 7,
        "minimum_depth_snr": 6.0,
        "minimum_observed_events": 3,
        "minimum_event_coverage": 0.7,
        "detrend_window_days": 2.0,
    }
    values.update(changes)
    return DetectionConfig(**values)


def test_bls_recovers_controlled_periodic_transit() -> None:
    config = _fast_detection_config()
    processed = process_discovery_light_curve(_controlled_curve(period=5.25), config)
    result = detect_periodic_signal(processed, config)

    assert result.status == "screening_detection"
    assert abs(result.period_days - 5.25) / 5.25 < 0.01
    assert result.depth_snr > config.minimum_depth_snr
    assert result.observed_events >= 10
    assert result.depth_ppm > 5_000
    assert processed.diagnostics["catalog_ephemeris_used"] is False


def test_high_threshold_control_does_not_claim_a_detection() -> None:
    config = _fast_detection_config(minimum_depth_snr=40.0)
    processed = process_discovery_light_curve(_controlled_curve(period=None), config)
    result = detect_periodic_signal(processed, config)

    assert result.status == "below_screening_threshold"
    assert "not a calibrated false-alarm probability" in " ".join(result.limitations)


def test_alternating_eclipses_trigger_secondary_binary_diagnostic() -> None:
    rng = np.random.default_rng(4)
    time = np.arange(0.0, 80.0, 0.0204)
    flux = 1.0 + rng.normal(0.0, 0.0008, size=time.size)
    for index, epoch in enumerate(np.arange(0.7, 80.0, 5.0)):
        depth = 0.02 if index % 2 == 0 else 0.004
        flux[np.abs(time - epoch) <= 0.08] -= depth
    raw = RawLightCurve(
        kic=999998,
        time_bkjd=time,
        flux=flux * 100_000.0,
        flux_error=np.full(time.size, 80.0),
        quality=np.zeros(time.size, dtype=np.int64),
        segment=np.zeros(time.size, dtype=np.int64),
        provenance={"kind": "controlled_alternating_test_only"},
    )
    config = _fast_detection_config()
    result = detect_periodic_signal(process_discovery_light_curve(raw, config), config)

    assert result.possible_eclipsing_binary is True
    assert result.secondary_depth is not None
    assert result.secondary_sigma is not None and result.secondary_sigma > 3.0


def test_selection_excludes_all_labeled_host_partitions(tmp_path: Path) -> None:
    catalog = pd.DataFrame(
        {
            "kepid": [101, 202, 303],
            "kepoi_name": ["K00001.01", "K00002.01", "K00003.01"],
            "koi_disposition": ["CANDIDATE", "CANDIDATE", "CANDIDATE"],
            "koi_period": [3.0, 4.0, 5.0],
            "koi_tce_plnt_num": [1, 1, 1],
            "koi_tce_delivname": ["test", "test", "test"],
        }
    )
    manifest = tmp_path / "split.csv"
    pd.DataFrame(
        {"kepoi_name": ["K99999.01"], "kepid": [101], "split": ["test"]}
    ).to_csv(manifest, index=False)
    targets = select_exploration_targets(
        CampaignConfig(sample_size=2, max_new_downloads=0),
        catalog=catalog,
        manifest_path=manifest,
    )

    assert {target.kic for target in targets} == {202, 303}
    assert all(target.label == -1 for target in targets)


def test_crossmatch_reports_period_relation_without_claiming_novelty() -> None:
    catalog = pd.DataFrame(
        {
            "kepid": [42],
            "kepoi_name": ["K00042.01"],
            "koi_disposition": ["CANDIDATE"],
            "koi_period": [7.0],
            "koi_tce_plnt_num": [1],
            "koi_tce_delivname": ["q1_q17_dr25_tce"],
        }
    )
    matches = crossmatch_detection(42, 14.0, catalog=catalog)

    assert len(matches) == 1
    assert matches[0].relation == "double_catalog_period"
    assert matches[0].tce_planet_number == 1


def test_cached_real_access_uses_no_network_and_no_processed_koi_series() -> None:
    raw = load_or_acquire_raw(
        10811496,
        ObservatoryConfig(),
        allow_network=False,
    )

    assert raw.kic == 10811496
    assert raw.time_bkjd.size > 1_000
    assert raw.provenance["catalog_ephemeris_used_for_discovery_processing"] is False


def test_default_config_keeps_network_opt_in_and_campaign_bounded() -> None:
    config = DiscoveryConfig()

    assert config.campaign.allow_network is False
    assert 1 <= config.campaign.sample_size <= 25
    assert config.campaign.max_products_per_star == 1
    assert config.detection.maximum_period_days == 30.0
