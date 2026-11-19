"""Safe target selection and post-detection local catalog cross-checks."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.discovery.acquisition import cached_kics
from src.discovery.config import CampaignConfig
from src.discovery.models import CatalogAssociation, DiscoveryTarget


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KOI_PATH = PROJECT_ROOT / "data" / "raw" / "kepler_koi.csv"
SPLIT_MANIFEST = PROJECT_ROOT / "experiments" / "stage04" / "split_manifest.csv"

CATALOG_COLUMNS = (
    "kepid",
    "kepoi_name",
    "koi_disposition",
    "koi_period",
    "koi_tce_plnt_num",
    "koi_tce_delivname",
)


def load_crossmatch_catalog(path: Path = KOI_PATH) -> pd.DataFrame:
    """Load only fields needed after a detection has been measured."""
    return pd.read_csv(path, usecols=list(CATALOG_COLUMNS))


def protected_host_stars(manifest_path: Path = SPLIT_MANIFEST) -> set[int]:
    """Return every host assigned to a labeled train/validation/test partition."""
    manifest = pd.read_csv(manifest_path, usecols=["kepid", "split"])
    protected = manifest[manifest["split"].isin(["train", "validation", "test"])]
    return set(protected["kepid"].astype(int))


def select_exploration_targets(
    config: CampaignConfig,
    *,
    catalog: pd.DataFrame | None = None,
    manifest_path: Path = SPLIT_MANIFEST,
) -> list[DiscoveryTarget]:
    """Select unique unlabeled candidate hosts, excluding all labeled partitions.

    Catalog periods, epochs, durations, and scores are not returned to the detector.
    The KOI identifier is retained only as an auditable selection reference.
    """
    catalog = load_crossmatch_catalog() if catalog is None else catalog.copy()
    frozen = protected_host_stars(manifest_path)
    eligible = catalog[
        catalog["koi_disposition"].eq("CANDIDATE")
        & catalog["kepid"].notna()
        & ~catalog["kepid"].astype(int).isin(frozen)
    ].copy()
    eligible = eligible.sort_values("kepoi_name").drop_duplicates("kepid", keep="first")
    local = cached_kics()
    cached_rows = [row for _, row in eligible[eligible["kepid"].astype(int).isin(local)].iterrows()]
    uncached_rows = [row for _, row in eligible[~eligible["kepid"].astype(int).isin(local)].iterrows()]
    rng = random.Random(config.random_seed)
    rng.shuffle(cached_rows)
    rng.shuffle(uncached_rows)
    rows = cached_rows + uncached_rows
    if len(rows) < config.sample_size:
        raise RuntimeError("insufficient unlabeled, non-protected Kepler hosts for campaign")
    targets = [
        DiscoveryTarget(
            kic=int(row["kepid"]),
            mission=config.mission,
            selection_strategy=config.strategy,
            cache_available=int(row["kepid"]) in local,
            selection_reference=str(row["kepoi_name"]),
            label=-1,
        )
        for row in rows[: config.sample_size]
    ]
    overlap = {target.kic for target in targets} & frozen
    if overlap:
        raise RuntimeError(f"protected host-star overlap detected: {sorted(overlap)}")
    if any(target.label != -1 for target in targets):
        raise RuntimeError("exploration targets must remain unlabeled")
    return targets


def _period_relation(detected: float, catalog_period: float, tolerance: float) -> str | None:
    ratio = detected / catalog_period
    for expected, label in (
        (1.0, "period_match"),
        (0.5, "half_catalog_period"),
        (2.0, "double_catalog_period"),
        (1.0 / 3.0, "one_third_catalog_period"),
        (3.0, "triple_catalog_period"),
    ):
        if abs(ratio - expected) <= tolerance * expected:
            return label
    return None


def crossmatch_detection(
    kic: int,
    detected_period_days: float,
    *,
    tolerance: float = 0.02,
    catalog: pd.DataFrame | None = None,
) -> list[CatalogAssociation]:
    """Compare a measured period with KOIs and KOI-linked TCE fields by host star."""
    catalog = load_crossmatch_catalog() if catalog is None else catalog
    matches = catalog[catalog["kepid"].eq(int(kic))]
    associations: list[CatalogAssociation] = []
    for _, row in matches.iterrows():
        period = float(row["koi_period"]) if pd.notna(row["koi_period"]) else None
        ratio = detected_period_days / period if period and period > 0 else None
        relation = _period_relation(detected_period_days, period, tolerance) if period else None
        tce_number = int(row["koi_tce_plnt_num"]) if pd.notna(row["koi_tce_plnt_num"]) else None
        tce_delivery = str(row["koi_tce_delivname"]) if pd.notna(row["koi_tce_delivname"]) else None
        associations.append(
            CatalogAssociation(
                object_id=str(row["kepoi_name"]),
                disposition=str(row["koi_disposition"]),
                catalog_period_days=period,
                period_ratio=float(ratio) if ratio is not None and np.isfinite(ratio) else None,
                relation=relation,
                tce_planet_number=tce_number,
                tce_delivery=tce_delivery,
            )
        )
    return associations


def crossmatch_summary(associations: list[CatalogAssociation]) -> dict[str, Any]:
    """Summarize catalog evidence without claiming novelty from a missing match."""
    period_matches = [item for item in associations if item.relation is not None]
    tce_links = [item for item in associations if item.tce_planet_number is not None]
    return {
        "host_has_koi_records": bool(associations),
        "period_association_found": bool(period_matches),
        "period_associations": len(period_matches),
        "koi_linked_tce_records": len(tce_links),
        "standalone_tce_catalog_checked": False,
        "novelty_established": False,
        "limitation": (
            "TCE information is limited to fields linked from the local KOI snapshot; "
            "absence of a match does not establish a new discovery."
        ),
    }
