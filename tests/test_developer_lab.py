"""Integration tests for the local ASTRAL Developer Lab."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.lab.services import (
    DEFAULT_DATA_PATH,
    dataset_overview,
    infer_record,
    load_dataset_frame,
    load_scientific_model,
    scientific_validation_results,
    train_developer_model,
)
from src.training.config import TrainingConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_scientific_model_drives_real_candidate_inference() -> None:
    """Ensure the Lab reproduces the persisted Stage 04 candidate score."""
    state = load_scientific_model()
    assert state is not None
    assert state.scientific_validation is True
    assert state.model.parameter_count == 97
    assert len(state.model.training_history) == 120

    ranking = pd.read_csv(PROJECT_ROOT / "experiments/stage04/candidate_ranking.csv")
    expected = ranking.iloc[0]
    frame = load_dataset_frame()
    record = frame[frame["kepoi_name"] == expected["kepoi_name"]].iloc[0]
    result = infer_record(state, record)

    assert result.object_id == expected["kepoi_name"]
    assert result.original_disposition == "CANDIDATE"
    assert result.score == pytest.approx(expected["ann_score_confirmed"], abs=1e-10)
    assert len(result.activations) == 4
    assert [values.shape[1] for values in result.activations] == [6, 8, 4, 1]


def test_dataset_overview_uses_local_nasa_snapshot() -> None:
    """Report identities and labels from the real local cumulative table."""
    frame = load_dataset_frame()
    overview = dataset_overview(frame, DEFAULT_DATA_PATH)

    assert overview["records"] == len(frame) == 9564
    assert overview["objects"] == frame["kepoi_name"].nunique()
    assert overview["dispositions"]["CANDIDATE"] == 1977
    assert overview["version_id"].startswith("sha256:")


def test_developer_training_persists_without_opening_test(tmp_path) -> None:
    """Exercise the browser training service with genuine Kepler partitions."""
    observed_epochs: list[int] = []
    config = TrainingConfig(epochs=1, batch_size=512, seed=2718, log_every=1)
    state = train_developer_model(
        config,
        output_directory=tmp_path / "model",
        manifest_path=tmp_path / "split_manifest.csv",
        epoch_callback=lambda record, _: observed_epochs.append(int(record["epoch"])),
    )

    assert observed_epochs == [1]
    assert state.metrics["test_accessed"] is False
    assert state.metrics["splits"]["test_reserved"]["records"] > 0
    assert (tmp_path / "model/model.npz").is_file()
    assert (tmp_path / "model/network_snapshot.json").is_file()
    assert (tmp_path / "split_manifest.csv").is_file()


def test_missing_validation_artifact_is_reported(tmp_path) -> None:
    """Return unavailable rather than inventing scientific results."""
    assert scientific_validation_results(tmp_path / "missing.json") is None


def test_streamlit_exposes_ann02_without_breaking_existing_tabs() -> None:
    """Smoke-test all original tabs plus both signal-intelligence surfaces."""
    app = AppTest.from_file(PROJECT_ROOT / "developer_lab.py")
    app.run(timeout=45)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Neural Observatory",
        "Training Console",
        "Object Inference",
        "Deep Space Observatory",
        "ANN-02 — Signal Intelligence",
        "Scientific Validation",
        "NASA Dataset",
    ]
