"""Acquire and inspect the NASA Exoplanet Archive cumulative KOI table.

The downloaded CSV is kept as delivered by the official TAP service. Any type
conversion performed by this module is restricted to in-memory exploratory
analysis and is never written back to the raw dataset.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(PROJECT_ROOT / ".cache" / "matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import EmptyDataError, ParserError


LOGGER = logging.getLogger(__name__)

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "kepler_koi.csv"
REPORT_PATH = PROJECT_ROOT / "visualization" / "kepler_koi_report.md"
VISUALIZATION_DIR = PROJECT_ROOT / "visualization"

TAP_ENDPOINT = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
TAP_QUERY = "select * from cumulative"

IDENTIFIER_COLUMNS = ("kepid", "kepoi_name", "kepler_name")
CLASSIFICATION_COLUMNS = ("koi_disposition", "koi_pdisposition")
NUMERIC_ANALYSIS_COLUMNS = (
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_model_snr",
    "koi_teq",
)
REQUIRED_COLUMNS = (
    *IDENTIFIER_COLUMNS,
    *CLASSIFICATION_COLUMNS,
    *NUMERIC_ANALYSIS_COLUMNS,
)


class DataAcquisitionError(RuntimeError):
    """Raised when the archive response cannot be downloaded."""


class DataValidationError(RuntimeError):
    """Raised when the downloaded table does not meet the expected schema."""


def build_tap_url() -> str:
    """Build the official synchronous TAP request URL."""
    return f"{TAP_ENDPOINT}?{urlencode({'query': TAP_QUERY, 'format': 'csv'})}"


def find_missing_required_columns(columns: pd.Index | list[str]) -> list[str]:
    """Return required schema fields that are absent from ``columns``."""
    available = set(columns)
    return [column for column in REQUIRED_COLUMNS if column not in available]


def load_and_validate_dataset(path: Path) -> pd.DataFrame:
    """Load a KOI CSV and validate its minimum schema without modifying it."""
    try:
        data = pd.read_csv(path, low_memory=False)
    except (OSError, EmptyDataError, ParserError) as exc:
        raise DataValidationError(f"Could not parse CSV file {path}: {exc}") from exc

    if data.empty:
        raise DataValidationError(f"The downloaded table is empty: {path}")

    missing_columns = find_missing_required_columns(data.columns)
    if missing_columns:
        joined_columns = ", ".join(missing_columns)
        raise DataValidationError(
            f"The downloaded table is missing required columns: {joined_columns}"
        )

    if data["kepoi_name"].isna().any():
        raise DataValidationError("The object identifier column kepoi_name has missing values")

    LOGGER.info("Validated %s records and %s columns", len(data), len(data.columns))
    return data


def download_cumulative_table(
    destination: Path = RAW_DATA_PATH,
    timeout_seconds: float = 180.0,
) -> pd.DataFrame:
    """Download the full cumulative table and atomically store the raw CSV."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_url = build_tap_url()
    request = Request(
        source_url,
        headers={
            "Accept": "text/csv",
            "User-Agent": "ASTRAL/0.1 (astronomical research project)",
        },
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix="kepler_koi_",
        suffix=".csv.tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    LOGGER.info("Downloading the NASA Exoplanet Archive cumulative KOI table")
    LOGGER.info("TAP source: %s", source_url)

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise DataAcquisitionError(f"Archive returned HTTP status {status}")

            with temporary_path.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)

        if temporary_path.stat().st_size == 0:
            raise DataAcquisitionError("Archive returned an empty response")

        data = load_and_validate_dataset(temporary_path)
        os.replace(temporary_path, destination)
        LOGGER.info(
            "Raw table stored at %s (%s bytes)",
            destination,
            destination.stat().st_size,
        )
        return data
    except DataValidationError:
        raise
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise DataAcquisitionError(
            f"Could not download the cumulative KOI table: {exc}"
        ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def _format_report_value(value: object) -> str:
    """Format a scalar for a compact Markdown report."""
    if pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    """Render a small Markdown table without optional dependencies."""
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(_format_report_value(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def generate_exploratory_report(
    data: pd.DataFrame,
    report_path: Path = REPORT_PATH,
    generated_at: datetime | None = None,
) -> Path:
    """Generate a Markdown report from the unmodified raw table."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = generated_at or datetime.now(UTC)

    disposition_counts = data["koi_disposition"].fillna("<MISSING>").value_counts()
    disposition_rows = [
        [classification, count, count / len(data) * 100]
        for classification, count in disposition_counts.items()
    ]

    missing_counts = data.isna().sum().sort_values(ascending=False)
    missing_rows = [
        [column, count, count / len(data) * 100]
        for column, count in missing_counts.items()
        if count > 0
    ]
    if not missing_rows:
        missing_rows = [["None", 0, 0.0]]

    numeric_data = data.loc[:, NUMERIC_ANALYSIS_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    statistics = numeric_data.describe().transpose()
    statistics_rows = [
        [column, *statistics.loc[column].tolist()]
        for column in NUMERIC_ANALYSIS_COLUMNS
    ]

    report = "\n".join(
        [
            "# Kepler KOI exploratory report",
            "",
            f"- Generated at (UTC): {generated_at.isoformat()}",
            f"- Source: {build_tap_url()}",
            f"- Records: {len(data)}",
            f"- Columns: {len(data.columns)}",
            "- Raw-data transformations: none",
            "",
            "## Original KOI disposition distribution",
            "",
            _markdown_table(
                ["koi_disposition", "count", "percentage"],
                disposition_rows,
            ),
            "",
            "## Missing values",
            "",
            "Only columns with at least one missing value are listed.",
            "",
            _markdown_table(
                ["column", "missing_count", "missing_percentage"],
                missing_rows,
            ),
            "",
            "## Descriptive statistics for required numeric columns",
            "",
            _markdown_table(
                ["column", *statistics.columns.tolist()],
                statistics_rows,
            ),
            "",
            "Numeric conversion was performed in memory only for this report. "
            "The raw CSV remains unchanged.",
            "",
        ]
    )
    report_path.write_text(report, encoding="utf-8")
    LOGGER.info("Exploratory report stored at %s", report_path)
    return report_path


def generate_exploratory_plots(
    data: pd.DataFrame,
    output_directory: Path = VISUALIZATION_DIR,
) -> list[Path]:
    """Generate classification, missing-value, and numeric-distribution plots."""
    output_directory.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    disposition_counts = data["koi_disposition"].fillna("<MISSING>").value_counts()
    figure, axis = plt.subplots(figsize=(9, 5))
    disposition_counts.plot(kind="bar", ax=axis, color="#3b82f6")
    axis.set_title("Kepler KOI disposition distribution")
    axis.set_xlabel("Original koi_disposition")
    axis.set_ylabel("Number of objects")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    disposition_path = output_directory / "kepler_koi_disposition_distribution.png"
    figure.savefig(disposition_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    output_paths.append(disposition_path)

    required_analysis_columns = [*NUMERIC_ANALYSIS_COLUMNS, "koi_disposition"]
    missing_counts = data.loc[:, required_analysis_columns].isna().sum()
    figure, axis = plt.subplots(figsize=(10, 5))
    missing_counts.plot(kind="bar", ax=axis, color="#f59e0b")
    axis.set_title("Missing values in required analysis columns")
    axis.set_xlabel("Column")
    axis.set_ylabel("Missing values")
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    missing_path = output_directory / "kepler_koi_missing_values.png"
    figure.savefig(missing_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    output_paths.append(missing_path)

    numeric_data = data.loc[:, NUMERIC_ANALYSIS_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, column in zip(axes.flat, NUMERIC_ANALYSIS_COLUMNS, strict=True):
        axis.hist(numeric_data[column].dropna(), bins=40, color="#10b981")
        axis.set_title(column)
        axis.set_xlabel("Original value")
        axis.set_ylabel("Frequency")
    figure.suptitle("Required Kepler KOI numeric distributions", fontsize=14)
    figure.tight_layout()
    distributions_path = output_directory / "kepler_koi_numeric_distributions.png"
    figure.savefig(distributions_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    output_paths.append(distributions_path)

    for output_path in output_paths:
        LOGGER.info("Exploratory plot stored at %s", output_path)
    return output_paths


def run_pipeline(force_download: bool = False, timeout_seconds: float = 180.0) -> None:
    """Acquire or reuse the raw table, then generate exploratory artifacts."""
    if RAW_DATA_PATH.exists() and not force_download:
        LOGGER.info("Using existing raw table at %s", RAW_DATA_PATH)
        data = load_and_validate_dataset(RAW_DATA_PATH)
    else:
        data = download_cumulative_table(timeout_seconds=timeout_seconds)

    generate_exploratory_report(data)
    generate_exploratory_plots(data)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Acquire and inspect the NASA cumulative Kepler KOI table."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download a fresh copy even when the raw CSV already exists.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP timeout in seconds (default: 180).",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Console logging level (default: INFO).",
    )
    return parser.parse_args()


def main() -> int:
    """Run the acquisition CLI and return a process status code."""
    arguments = parse_arguments()
    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    try:
        run_pipeline(
            force_download=arguments.force,
            timeout_seconds=arguments.timeout,
        )
    except (DataAcquisitionError, DataValidationError) as exc:
        LOGGER.error("%s", exc)
        return 1
    except Exception:
        LOGGER.exception("Unexpected failure during KOI data acquisition")
        return 1

    LOGGER.info("Kepler KOI acquisition and exploration completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
