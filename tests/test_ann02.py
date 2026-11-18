"""Architecture, data-integrity, training, and observability tests for ANN-02."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.deep_learning.config import ANN02Config, DatasetBuildConfig, TrainingConfig
from src.deep_learning.dataset import KeplerSignalDataset, select_balanced_cohort
from src.deep_learning.model import DualBranchCNN
from src.deep_learning.training import load_ann02, train_ann02, validate_dataset_integrity
from src.observatory.catalog import load_observatory_catalog


def _write_small_dataset(path) -> None:
    """Create deterministic numerical fixtures solely for isolated unit testing."""
    rng = np.random.default_rng(17)
    records = 13
    labels = np.asarray([0, 1] * 4 + [0, 1] * 2 + [-1], dtype=np.int64)
    splits = np.asarray(["train"] * 8 + ["validation"] * 4 + ["candidate"])
    global_flux = 1.0 + rng.normal(0.0, 0.001, size=(records, 2001))
    local_flux = 1.0 + rng.normal(0.0, 0.001, size=(records, 201))
    for index, label in enumerate(labels):
        if label == 1:
            global_flux[index, 990:1011] -= 0.006
            local_flux[index, 95:106] -= 0.006
    global_mask = np.ones_like(global_flux, dtype=bool)
    local_mask = np.ones_like(local_flux, dtype=bool)
    global_mask[:, 0] = False
    local_mask[:, -1] = False
    global_flux[~global_mask] = np.nan
    local_flux[~local_mask] = np.nan
    np.savez_compressed(
        path,
        object_ids=np.asarray([f"fixture-{index}" for index in range(records)]),
        kic=np.arange(1000, 1000 + records),
        splits=splits,
        class_labels=labels,
        global_flux=global_flux,
        global_mask=global_mask,
        local_flux=local_flux,
        local_mask=local_mask,
    )


def test_ann02_architecture_has_two_mask_aware_branches() -> None:
    model = DualBranchCNN(ANN02Config())
    global_view = torch.zeros((3, 2, 2001), dtype=torch.float32)
    local_view = torch.zeros((3, 2, 201), dtype=torch.float32)
    logits = model(global_view, local_view, retain_observability=True)

    assert logits.shape == (3,)
    assert model.trainable_parameter_count == 2033
    assert "global.relu_2" in model.last_activations
    assert "local.relu_2" in model.last_activations
    assert model.architecture()["channels"] == ["median-centered flux", "availability mask"]


def test_balanced_selection_never_uses_frozen_test_stars() -> None:
    cohort = select_balanced_cohort(
        load_observatory_catalog(),
        DatasetBuildConfig(pilot_size=16, candidate_size=1),
    )
    labelled = [item for item in cohort if item["class_label"] >= 0]

    assert len(labelled) == 16
    assert {item["split"] for item in labelled} == {"train", "validation"}
    assert sum(item["class_label"] == 0 for item in labelled) == 8
    assert sum(item["class_label"] == 1 for item in labelled) == 8
    assert len({item["kic"] for item in cohort}) == len(cohort)
    assert any(item["object_id"] == "K00753.01" for item in cohort)


def test_tensor_preparation_keeps_mask_and_neutral_missing_bins(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.npz"
    _write_small_dataset(dataset_path)
    integrity = validate_dataset_integrity(dataset_path)
    sample = KeplerSignalDataset(dataset_path, "train")[0]

    assert integrity["test_records"] == 0
    assert sample["global_view"].shape == (2, 2001)
    assert sample["local_view"].shape == (2, 201)
    assert sample["global_view"][0, 0].item() == pytest.approx(0.0)
    assert sample["global_view"][1, 0].item() == pytest.approx(0.0)
    assert torch.isfinite(sample["global_view"]).all()


def test_integrity_rejects_labelled_candidate(tmp_path) -> None:
    """Candidates must never enter ANN-02 as labelled supervision."""
    dataset_path = tmp_path / "dataset.npz"
    _write_small_dataset(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as payload:
        values = {key: payload[key] for key in payload.files}
    values["class_labels"][-1] = 1
    np.savez_compressed(dataset_path, **values)

    with pytest.raises(ValueError, match="Candidate records must remain unlabelled"):
        validate_dataset_integrity(dataset_path)


def test_integrity_rejects_host_star_crossing_splits(tmp_path) -> None:
    """No host KIC may cross training, validation, test, or candidate partitions."""
    dataset_path = tmp_path / "dataset.npz"
    _write_small_dataset(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as payload:
        values = {key: payload[key] for key in payload.files}
    values["kic"][8] = values["kic"][0]
    np.savez_compressed(dataset_path, **values)

    with pytest.raises(ValueError, match="KIC leakage detected"):
        validate_dataset_integrity(dataset_path)


def test_training_persists_best_checkpoint_and_real_snapshot(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.npz"
    _write_small_dataset(dataset_path)
    model_directory = tmp_path / "model"
    bundle = train_ann02(
        ANN02Config(),
        TrainingConfig(epochs=2, batch_size=3, patience=2, seed=31),
        dataset_path=dataset_path,
        model_directory=model_directory,
        experiment_directory=tmp_path / "experiments",
    )

    assert bundle.experiment["scientific_guardrails"]["test_accessed"] is False
    assert bundle.experiment["validation_metrics"]["records"] == 4
    best_history = bundle.experiment["history"][bundle.experiment["best_epoch"] - 1]
    assert best_history["validation_loss"] == pytest.approx(
        bundle.experiment["validation_metrics"]["binary_cross_entropy"], abs=1e-6
    )
    assert (model_directory / "model.pt").is_file()
    snapshot = json.loads((model_directory / "observability_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["architecture"]["trainable_parameters"] == 2033
    assert snapshot["convolutional_filters"]
    assert snapshot["activations"]["global.relu_2"]["sample_index"] == 0
    loaded = load_ann02(model_directory, device_preference="cpu")
    assert loaded is not None
    assert loaded.model.trainable_parameter_count == 2033
