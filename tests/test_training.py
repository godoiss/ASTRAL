"""Tests for ANN-01 mini-batch training and bundle persistence."""

import numpy as np

from src.core.mlp import MLP
from src.data.preprocessing import DataPartition, FeaturePreprocessor
from src.training.config import TrainingConfig
from src.training.persistence import load_model_bundle, save_model_bundle
from src.training.trainer import Trainer


def _partition(features: np.ndarray, targets: np.ndarray) -> DataPartition:
    record_count = len(features)
    return DataPartition(
        features=features,
        targets=targets,
        group_ids=np.asarray([f"star-{index}" for index in range(record_count)]),
        object_ids=np.asarray([f"object-{index}" for index in range(record_count)]),
        dispositions=np.where(targets.reshape(-1) == 1.0, "CONFIRMED", "FALSE POSITIVE"),
    )


def test_mini_batch_training_records_real_history() -> None:
    """Verify mini-batch updates create finite per-epoch observations."""
    generator = np.random.default_rng(41)
    features = generator.normal(size=(80, 6))
    targets = (features[:, :1] + features[:, 1:2] > 0.0).astype(np.float64)
    train = _partition(features[:64], targets[:64])
    validation = _partition(features[64:], targets[64:])
    config = TrainingConfig(epochs=8, batch_size=16, seed=43, log_every=8)
    model = MLP(seed=config.seed)
    callback_epochs: list[int] = []

    history = Trainer(config).fit(
        model,
        train,
        validation,
        epoch_callback=lambda record, _: callback_epochs.append(int(record["epoch"])),
    )

    assert len(history) == 8
    assert all(np.isfinite(record["train_loss"]) for record in history)
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert model.training_history == history
    assert callback_epochs == list(range(1, 9))


def test_complete_model_bundle_round_trip(tmp_path) -> None:
    """Restore model, preprocessing, config, history, and metrics together."""
    config = TrainingConfig(epochs=2)
    model = MLP(seed=config.seed)
    model.training_history = [
        {
            "epoch": 1,
            "train_loss": 0.5,
            "train_accuracy": 0.75,
            "validation_loss": 0.6,
            "validation_accuracy": 0.7,
        }
    ]
    training_frame = {
        feature: np.array([1.0, 2.0, 3.0])
        for feature in FeaturePreprocessor().feature_names
    }
    import pandas as pd

    preprocessor = FeaturePreprocessor().fit(pd.DataFrame(training_frame))
    metrics = {"test": {"loss": 0.4, "accuracy": 0.8, "records": 10}}
    inputs = np.ones((2, 6))

    save_model_bundle(tmp_path, model, preprocessor, config, metrics)
    restored = load_model_bundle(tmp_path)

    np.testing.assert_allclose(
        restored.model.predict_proba(inputs),
        model.predict_proba(inputs),
    )
    assert restored.config == config
    assert restored.model.training_history == model.training_history
    assert restored.metrics == metrics
