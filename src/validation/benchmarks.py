"""Fair Scikit-learn benchmarks evaluated on the same frozen partitions."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.data.preprocessing import DataPartition
from src.validation.metrics import compute_classification_metrics


def _positive_scores(model: Any, features: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(features)
    positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
    return probabilities[:, positive_index].astype(np.float64)


def evaluate_benchmarks(
    train: DataPartition,
    test: DataPartition,
    astral_scores: np.ndarray,
    threshold: float,
    seed: int,
    settings: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Fit fixed baselines on train and evaluate all models on one test split."""
    if train.targets is None or test.targets is None:
        raise ValueError("Benchmark evaluation requires labelled partitions")
    train_targets = train.targets.reshape(-1).astype(np.int64)
    test_targets = test.targets.reshape(-1).astype(np.int64)

    models: dict[str, Any] = {
        "Majority Class": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=int(settings["logistic_regression"]["max_iter"]),
            random_state=seed,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=int(settings["decision_tree"]["max_depth"]),
            min_samples_leaf=int(
                settings["decision_tree"]["min_samples_leaf"]
            ),
            random_state=seed,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=int(settings["random_forest"]["n_estimators"]),
            min_samples_leaf=int(
                settings["random_forest"]["min_samples_leaf"]
            ),
            random_state=seed,
            n_jobs=-1,
        ),
    }
    results: dict[str, Any] = {
        "ASTRAL Neural Engine": {
            "training_seconds": None,
            "metrics": compute_classification_metrics(
                test_targets,
                astral_scores,
                threshold=threshold,
            ),
            "implementation": "NumPy MLP",
        }
    }
    score_map: dict[str, np.ndarray] = {
        "ASTRAL Neural Engine": np.asarray(astral_scores, dtype=np.float64)
    }
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(train.features, train_targets)
        training_seconds = time.perf_counter() - start
        scores = _positive_scores(model, test.features)
        score_map[name] = scores
        results[name] = {
            "training_seconds": training_seconds,
            "metrics": compute_classification_metrics(
                test_targets,
                scores,
                threshold=threshold,
            ),
            "implementation": type(model).__name__,
        }
    return results, score_map

