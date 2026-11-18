"""Command-line acquisition and bounded validation for Deep Space Observatory."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from src.observatory.catalog import export_derived_catalog, find_koi
from src.observatory.config import ObservatoryConfig
from src.observatory.dataset import save_dataset_collection
from src.observatory.models import NeuralLightCurveExample
from src.observatory.pipeline import acquire_observation
from src.training.logging_utils import configure_structured_logging


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "deep_space_observatory.json"
DEFAULT_SAMPLE = PROJECT_ROOT / "configs" / "deep_space_validation_sample.json"
DEFAULT_SAMPLE_DATASET = (
    PROJECT_ROOT / "data" / "observatory" / "neural" / "stage05_validation_sample.npz"
)
DEFAULT_REPORT = PROJECT_ROOT / "experiments" / "stage05" / "deep_space_validation.json"
LOGGER = logging.getLogger(__name__)


def _progress(stage: str, progress: float, message: str) -> None:
    LOGGER.info(
        message,
        extra={"event": f"observatory_{stage}", "details": {"progress": progress}},
    )


def _fetch(
    identifier: str,
    config: ObservatoryConfig,
    force: bool,
) -> tuple[dict[str, object], NeuralLightCurveExample]:
    matches = find_koi(identifier)
    if not matches:
        raise ValueError(f"No KOI found for {identifier}")
    metadata = matches[0]
    artifact, example = acquire_observation(
        metadata,
        config,
        force_refresh=force,
        progress_callback=_progress,
    )
    return {
        "object_id": metadata.object_id,
        "display_name": metadata.display_name,
        "kic": metadata.kic,
        "disposition": metadata.disposition,
        "split": metadata.split,
        "raw_records": artifact.raw_records,
        "processed_records": artifact.processed_records,
        "segments": list(artifact.quarters),
        "global_observed_bins": int(example.global_mask.sum()),
        "local_observed_bins": int(example.local_mask.sum()),
        "provenance": str(artifact.provenance_path),
    }, example


def main() -> int:
    parser = argparse.ArgumentParser(description="ASTRAL Deep Space Observatory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch", help="Acquire one KOI light curve")
    fetch.add_argument("identifier")
    fetch.add_argument("--force", action="store_true")
    sample = subparsers.add_parser("validate-sample", help="Acquire bounded five-object sample")
    sample.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    sample.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    configure_structured_logging(arguments.log_level)
    config = ObservatoryConfig.load(arguments.config)
    export_derived_catalog()
    try:
        if arguments.command == "fetch":
            summary, _ = _fetch(arguments.identifier, config, arguments.force)
            output: object = [summary]
        else:
            payload = json.loads(arguments.sample.read_text(encoding="utf-8"))
            acquired = [
                _fetch(identifier, config, arguments.force)
                for identifier in payload["objects"]
            ]
            results = [item[0] for item in acquired]
            examples = [item[1] for item in acquired]
            dataset_path = save_dataset_collection(examples, DEFAULT_SAMPLE_DATASET)
            report = {
                "schema_version": 1,
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "configuration": config.to_dict(),
                "sample_description": payload["description"],
                "objects": results,
                "neural_dataset": str(dataset_path.resolve()),
                "test_partition_used_for_tuning": False,
            }
            DEFAULT_REPORT.parent.mkdir(parents=True, exist_ok=True)
            DEFAULT_REPORT.write_text(
                json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
            output = report
    except Exception:
        LOGGER.exception("Deep Space Observatory command failed", extra={"event": "observatory_failed"})
        return 1
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
