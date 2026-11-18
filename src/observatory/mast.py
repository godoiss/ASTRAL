"""Bounded Lightkurve/MAST acquisition with retries and persistent cache."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import lightkurve as lk
import numpy as np
from astropy.utils.data import conf as astropy_data_conf
from astroquery.mast import conf as mast_conf

from src.observatory.config import ObservatoryConfig
from src.observatory.models import KoiMetadata


LOGGER = logging.getLogger(__name__)
DEFAULT_FITS_CACHE = PROJECT_ROOT / "data" / "observatory" / "cache" / "lightkurve"
ProgressCallback = Callable[[str, float, str], None]


class MastAcquisitionError(RuntimeError):
    """Raised when MAST search or bounded download cannot be completed."""


@dataclass(frozen=True)
class DownloadedLightCurves:
    """Downloaded Lightkurve collection and serializable product metadata."""

    collection: Any
    products: list[dict[str, Any]]
    search_records: int
    selected_records: int
    cache_directory: Path
    elapsed_seconds: float


def _emit(
    callback: ProgressCallback | None,
    stage: str,
    progress: float,
    message: str,
) -> None:
    if callback is not None:
        callback(stage, float(progress), message)


def _json_value(value: Any) -> Any:
    if np.ma.is_masked(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _product_records(search_result: Any) -> list[dict[str, Any]]:
    table = search_result.table
    preferred = (
        "target_name",
        "mission",
        "author",
        "exptime",
        "distance",
        "year",
        "obsid",
        "productFilename",
        "dataURI",
    )
    columns = [column for column in preferred if column in table.colnames]
    return [
        {column: _json_value(row[column]) for column in columns}
        for row in table
    ]


def _with_retries(operation: Callable[[], Any], config: ObservatoryConfig, label: str) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, config.retries + 1):
        try:
            return operation()
        except Exception as exc:  # third-party network errors vary by backend
            last_error = exc
            LOGGER.warning(
                "%s failed on attempt %d/%d: %s",
                label,
                attempt,
                config.retries,
                exc,
            )
            if attempt < config.retries:
                time.sleep(min(2.0**(attempt - 1), 4.0))
    assert last_error is not None
    raise MastAcquisitionError(
        f"{label} failed after {config.retries} attempt(s): {last_error}"
    ) from last_error


def search_and_download(
    metadata: KoiMetadata,
    config: ObservatoryConfig,
    cache_directory: Path = DEFAULT_FITS_CACHE,
    progress_callback: ProgressCallback | None = None,
) -> DownloadedLightCurves:
    """Search by KIC and download only the configured number of FITS products."""
    started = time.perf_counter()
    cache_directory = Path(cache_directory)
    cache_directory.mkdir(parents=True, exist_ok=True)
    mast_conf.timeout = config.timeout_seconds
    astropy_data_conf.remote_timeout = config.timeout_seconds

    _emit(progress_callback, "search", 0.08, f"Searching MAST for KIC {metadata.kic}")
    search_kwargs: dict[str, Any] = {
        "target": f"KIC {metadata.kic}",
        "mission": config.mission,
        "author": config.author,
        "cadence": config.cadence,
        "limit": 50,
    }
    if config.mission == "Kepler" and config.quarter is not None:
        search_kwargs["quarter"] = config.quarter

    result = _with_retries(
        lambda: lk.search_lightcurve(**search_kwargs),
        config,
        "MAST light-curve search",
    )
    if len(result) == 0:
        raise MastAcquisitionError(
            f"No {config.mission} light curves found for KIC {metadata.kic}"
        )
    selected = result[: min(config.max_products, len(result))]
    products = _product_records(selected)
    _emit(
        progress_callback,
        "download",
        0.25,
        f"Downloading {len(selected)} bounded FITS product(s)",
    )
    collection = _with_retries(
        lambda: selected.download_all(
            quality_bitmask="none",
            download_dir=str(cache_directory),
            flux_column="pdcsap_flux",
        ),
        config,
        "MAST FITS download",
    )
    if collection is None or len(collection) == 0:
        raise MastAcquisitionError(
            f"MAST returned no downloadable light curve for KIC {metadata.kic}"
        )
    _emit(progress_callback, "download", 0.58, "FITS products available in local cache")
    return DownloadedLightCurves(
        collection=collection,
        products=products,
        search_records=len(result),
        selected_records=len(selected),
        cache_directory=cache_directory.resolve(),
        elapsed_seconds=float(time.perf_counter() - started),
    )
