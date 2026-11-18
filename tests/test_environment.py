"""Initial smoke tests for the ASTRAL project environment."""

from importlib import import_module

import pytest

from main import main


@pytest.mark.parametrize(
    "package_name",
    ["numpy", "pandas", "matplotlib", "lightkurve", "astropy", "plotly", "streamlit", "torch"],
)
def test_scientific_dependency_is_importable(package_name: str) -> None:
    """Verify that each initial scientific dependency can be imported."""
    assert import_module(package_name) is not None


def test_main_reports_ready(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify the minimal application entry point."""
    assert main() == 0
    assert capsys.readouterr().out.strip() == "ASTRAL environment is ready."
