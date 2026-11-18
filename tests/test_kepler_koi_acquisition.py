"""Unit tests for the Kepler KOI acquisition module."""

from urllib.parse import parse_qs, urlparse

from src.data.acquire_kepler_koi import (
    REQUIRED_COLUMNS,
    TAP_ENDPOINT,
    TAP_QUERY,
    build_tap_url,
    find_missing_required_columns,
)


def test_tap_url_targets_full_official_cumulative_table() -> None:
    """Ensure the request uses the official TAP endpoint and full table query."""
    parsed_url = urlparse(build_tap_url())
    parsed_endpoint = urlparse(TAP_ENDPOINT)
    query_parameters = parse_qs(parsed_url.query)

    assert parsed_url.scheme == "https"
    assert parsed_url.netloc == parsed_endpoint.netloc
    assert parsed_url.path == parsed_endpoint.path
    assert query_parameters == {"query": [TAP_QUERY], "format": ["csv"]}


def test_required_schema_covers_identifiers_classifications_and_features() -> None:
    """Ensure all contract columns remain explicit in the acquisition schema."""
    assert set(REQUIRED_COLUMNS) == {
        "kepid",
        "kepoi_name",
        "kepler_name",
        "koi_disposition",
        "koi_pdisposition",
        "koi_period",
        "koi_duration",
        "koi_depth",
        "koi_prad",
        "koi_model_snr",
        "koi_teq",
    }


def test_missing_required_columns_are_reported_in_contract_order() -> None:
    """Ensure schema validation reports absent fields deterministically."""
    available_columns = list(REQUIRED_COLUMNS[:-2])

    assert find_missing_required_columns(available_columns) == [
        "koi_model_snr",
        "koi_teq",
    ]

