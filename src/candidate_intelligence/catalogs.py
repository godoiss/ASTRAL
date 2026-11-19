"""Bounded clients for official Kepler catalog verification.

The verification workflow queries only the KIC identifiers already present in a
persisted discovery campaign. Responses are persisted by the Stage 11 builder;
the public API never performs a network request.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


TAP_SYNC_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
KOI_TABLE = "cumulative"
TCE_TABLE = "q1_q17_dr25_tce"

KOI_COLUMNS = (
    "kepid", "kepoi_name", "kepler_name", "koi_disposition", "koi_pdisposition",
    "koi_period", "koi_duration", "koi_depth", "koi_model_snr", "koi_time0bk",
    "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
    "koi_comment", "koi_tce_plnt_num", "koi_tce_delivname",
)
TCE_COLUMNS = (
    "kepid", "tce_plnt_num", "tce_period", "tce_duration", "tce_depth",
    "tce_model_snr", "tce_num_transits", "tce_bin_oedp_stat", "tce_rogue_flag",
    "av_training_set", "av_pred_class", "boot_fap", "tce_max_mult_ev", "tce_delivname",
)


class CatalogQueryError(RuntimeError):
    """Raised when an official bounded catalog query cannot be completed."""


def _default_fetch(url: str, timeout: float) -> bytes:
    request = Request(url, headers={"User-Agent": "ASTRAL/11 candidate-verification"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS endpoint
        return response.read()


class ExoplanetArchiveClient:
    """Query small KOI and DR25 TCE projections from the official NASA TAP service."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        fetch: Callable[[str, float], bytes] = _default_fetch,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.fetch = fetch

    @staticmethod
    def _identifiers(kics: Iterable[int]) -> list[int]:
        identifiers = sorted({int(kic) for kic in kics})
        if not identifiers or len(identifiers) > 25 or any(kic <= 0 or kic > 999_999_999 for kic in identifiers):
            raise ValueError("catalog verification requires between 1 and 25 valid KIC identifiers")
        return identifiers

    def _query(self, table: str, columns: tuple[str, ...], kics: Iterable[int]) -> tuple[list[dict[str, Any]], str]:
        identifiers = self._identifiers(kics)
        if table not in {KOI_TABLE, TCE_TABLE}:
            raise ValueError("catalog table is not allowed")
        query = f"select {','.join(columns)} from {table} where kepid in ({','.join(map(str, identifiers))})"
        url = f"{TAP_SYNC_URL}?{urlencode({'query': query, 'format': 'json'})}"
        try:
            payload = json.loads(self.fetch(url, self.timeout_seconds).decode("utf-8"))
        except (OSError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CatalogQueryError(f"official {table} query failed") from exc
        if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
            raise CatalogQueryError(f"official {table} response has an invalid structure")
        allowed = set(identifiers)
        rows = [row for row in payload if int(row.get("kepid", -1)) in allowed]
        return rows, query

    def verify(self, kics: Iterable[int]) -> dict[str, Any]:
        """Return bounded current KOI and DR25 TCE records for the supplied hosts."""
        identifiers = self._identifiers(kics)
        koi_rows, koi_query = self._query(KOI_TABLE, KOI_COLUMNS, identifiers)
        tce_rows, tce_query = self._query(TCE_TABLE, TCE_COLUMNS, identifiers)
        return {
            "kics": identifiers,
            "sources": [
                {
                    "name": "NASA Exoplanet Archive KOI Cumulative",
                    "table": KOI_TABLE,
                    "url": TAP_SYNC_URL,
                    "query": koi_query,
                    "records": len(koi_rows),
                },
                {
                    "name": "NASA Exoplanet Archive Q1-Q17 DR25 TCE",
                    "table": TCE_TABLE,
                    "url": TAP_SYNC_URL,
                    "query": tce_query,
                    "records": len(tce_rows),
                },
            ],
            "koi_records": koi_rows,
            "tce_records": tce_rows,
            "eclipsing_binary_catalog": {
                "source": "Kepler Eclipsing Binary Catalog",
                "status": "not_machine_queried",
                "reason": (
                    "The NASA Exoplanet Archive documents the catalog but does not expose its table "
                    "through the current TAP schema used by this reproducible verifier."
                ),
                "documentation": "https://exoplanetarchive.ipac.caltech.edu/docs/eclbin.html",
            },
        }
