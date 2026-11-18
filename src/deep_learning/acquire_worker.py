"""Isolated one-object MAST worker used to enforce hard acquisition timeouts."""

from __future__ import annotations

import argparse
import json

from src.observatory.catalog import find_koi
from src.observatory.config import ObservatoryConfig
from src.observatory.pipeline import acquire_observation


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated ASTRAL MAST acquisition worker")
    parser.add_argument("object_id")
    parser.add_argument("--quarter", type=int)
    parser.add_argument("--max-products", type=int, default=1)
    arguments = parser.parse_args()
    matches = find_koi(arguments.object_id)
    if not matches:
        raise ValueError(f"Unknown KOI: {arguments.object_id}")
    config = ObservatoryConfig(
        quarter=arguments.quarter,
        max_products=arguments.max_products,
    )
    artifact, example = acquire_observation(matches[0], config)
    print(
        json.dumps(
            {
                "object_id": matches[0].object_id,
                "raw_records": artifact.raw_records,
                "processed_records": artifact.processed_records,
                "global_observed_bins": int(example.global_mask.sum()),
                "local_observed_bins": int(example.local_mask.sum()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
