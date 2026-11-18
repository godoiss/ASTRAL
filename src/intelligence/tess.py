"""Strictly isolated experimental TESS compatibility and acquisition study."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TESS_ROOT = PROJECT_ROOT / "data" / "tess_experimental"
DEFAULT_TESS_REPORT = PROJECT_ROOT / "experiments" / "stage07" / "tess_pilot.json"


def tess_compatibility_report() -> dict[str, Any]:
    """Document mission compatibility without asserting cross-mission equivalence."""
    return {
        "schema_version": 1,
        "status": "technical_compatibility_only",
        "common_contract": {
            "container": "FITS light-curve products readable as Lightkurve LightCurve/Astropy Table",
            "core_columns": ["time", "flux", "flux_err", "quality"],
            "search_interface": "lightkurve.search_lightcurve(..., mission='TESS')",
        },
        "differences": [
            {"topic": "identity", "kepler": "KIC/KOI", "tess": "TIC/TOI"},
            {"topic": "segment", "kepler": "quarter", "tess": "sector"},
            {"topic": "time zero-point", "kepler": "BKJD = BJD - 2454833", "tess": "BTJD = BJD - 2457000"},
            {"topic": "cadence", "kepler": "typically ~30-minute long cadence", "tess": "20 s, 120 s, 200 s, 10 min, or 30 min depending on product/epoch"},
            {"topic": "coverage", "kepler": "repeated quarters over years", "tess": "typically sector-scale windows near 27 days, with overlap-dependent revisits"},
            {"topic": "pixels", "kepler": "~4 arcsec/pixel", "tess": "~21 arcsec/pixel; blending risk differs"},
        ],
        "requirements_before_cross_mission_validation": [
            "Mission-specific provenance, quality masks, time conversion, cadence handling, and detrending.",
            "TIC/TOI labels with host-star grouped partitions and explicit label provenance.",
            "Separate Kepler and TESS calibration studies before any mixed training.",
            "Evaluation of dilution, crowding, sector coverage, domain shift, and score calibration.",
        ],
        "sources": [
            "https://lightkurve.github.io/lightkurve/reference/api/lightkurve.search_lightcurve.html",
            "https://lightkurve.github.io/lightkurve/tutorials/1-getting-started/using-light-curve-file-products.html",
            "https://archive.stsci.edu/missions-and-data/tess",
            "https://outerspace.stsci.edu/spaces/TESS/pages/14563420/2.0%2B-%2BData%2BProduct%2BOverview",
        ],
        "mixed_with_kepler_training": False,
    }


def run_tess_pilot(
    target: str = "TIC 441420236",
    sector: int = 27,
    exptime_seconds: int = 120,
    download: bool = True,
    report_path: Path = DEFAULT_TESS_REPORT,
) -> dict[str, Any]:
    """Search and optionally download exactly one official SPOC TESS light curve."""
    import lightkurve as lk
    from astropy.utils.data import conf as astropy_data_conf
    from astroquery.mast import conf as mast_conf

    started = time.perf_counter()
    mast_conf.timeout = 60
    astropy_data_conf.remote_timeout = 60
    result = lk.search_lightcurve(
        target,
        mission="TESS",
        author="SPOC",
        sector=sector,
        exptime=exptime_seconds,
        limit=5,
    )
    if len(result) == 0:
        raise RuntimeError(f"No bounded official TESS product found for {target}, sector {sector}")
    selected = result[:1]
    table = selected.table
    columns = [name for name in ("target_name", "mission", "author", "exptime", "obsid", "productFilename", "dataURI") if name in table.colnames]
    product = {name: str(table[0][name]) for name in columns}
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "target": target,
        "sector_requested": sector,
        "exptime_requested_seconds": exptime_seconds,
        "search_records": int(len(result)),
        "selected_records": 1,
        "product": product,
        "downloaded": False,
        "compatibility": tess_compatibility_report(),
        "guardrails": {"files_requested": 1, "mixed_with_kepler_training": False},
    }
    if download:
        cache = DEFAULT_TESS_ROOT / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        light_curve = selected.download(
            quality_bitmask="none",
            download_dir=str(cache),
            flux_column="pdcsap_flux",
        )
        if light_curve is None:
            raise RuntimeError("MAST returned no downloadable TESS light curve")
        time_values = np.asarray(light_curve.time.value, dtype=float)
        flux_values = np.asarray(light_curve.flux.value, dtype=float)
        quality = np.asarray(light_curve["quality"], dtype=np.int64) if "quality" in light_curve.colnames else np.zeros(len(light_curve), dtype=np.int64)
        finite = np.isfinite(time_values) & np.isfinite(flux_values)
        cadence = float(np.nanmedian(np.diff(time_values)) * 86400.0) if len(time_values) > 1 else None
        payload["downloaded"] = True
        payload["observation"] = {
            "mission": str(light_curve.meta.get("MISSION", "TESS")),
            "author": str(light_curve.meta.get("AUTHOR", "SPOC")),
            "sector": int(light_curve.meta.get("SECTOR", sector)),
            "target_id": str(light_curve.meta.get("TARGETID", target)),
            "time_format": str(getattr(light_curve.time, "format", "unknown")),
            "time_scale": str(getattr(light_curve.time, "scale", "unknown")),
            "cadence_seconds_median": cadence,
            "records": int(len(light_curve)),
            "finite_records": int(finite.sum()),
            "quality_flagged_records": int(np.sum(quality != 0)),
            "time_start": float(np.nanmin(time_values)),
            "time_end": float(np.nanmax(time_values)),
            "time_span_days": float(np.nanmax(time_values) - np.nanmin(time_values)),
        }
    payload["elapsed_seconds"] = float(time.perf_counter() - started)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return payload
