"""Command-line entry point for Stage 04 scientific validation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.training.logging_utils import configure_structured_logging
from src.validation.campaign import run_campaign


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    """Build the reproducible validation campaign CLI."""
    parser = argparse.ArgumentParser(
        description="Run ASTRAL Stage 04 scientific validation"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "scientific_validation.json",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "kepler_koi.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "stage04",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "docs" / "stage04-scientific-validation.md",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main() -> int:
    """Execute the complete local campaign with structured logging."""
    arguments = build_parser().parse_args()
    configure_structured_logging(arguments.log_level)
    try:
        run_campaign(
            config_path=arguments.config,
            data_path=arguments.data,
            output_directory=arguments.output,
            report_path=arguments.report,
        )
    except Exception:
        LOGGER.exception(
            "Stage 04 validation failed",
            extra={"event": "stage04_failed"},
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

