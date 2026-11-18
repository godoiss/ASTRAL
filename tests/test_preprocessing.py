"""Tests for leakage-safe ANN-01 data preparation."""

import numpy as np
import pandas as pd

from src.data.preprocessing import (
    FEATURE_COLUMNS,
    FeaturePreprocessor,
    grouped_split_indices,
)


def _feature_frame(first_value: float, second_value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: [first_value + index, second_value + index]
            for index, column in enumerate(FEATURE_COLUMNS)
        }
    )


def test_preprocessor_uses_training_statistics_only() -> None:
    """Ensure validation values cannot influence imputation or scaling."""
    training = _feature_frame(1.0, 3.0)
    training.loc[0, "koi_depth"] = np.nan
    validation = _feature_frame(101.0, 103.0)
    preprocessor = FeaturePreprocessor().fit(training)

    transformed_training = preprocessor.transform(training)
    transformed_validation = preprocessor.transform(validation)

    np.testing.assert_allclose(transformed_training.mean(axis=0), np.zeros(6))
    assert np.all(transformed_validation.mean(axis=0) > 10.0)
    assert preprocessor.fitted_on_rows == 2


def test_preprocessor_persistence_round_trip(tmp_path) -> None:
    """Ensure persisted medians, means, and scales reproduce transformations."""
    training = _feature_frame(2.0, 6.0)
    preprocessor = FeaturePreprocessor().fit(training)
    path = tmp_path / "preprocessor.json"

    preprocessor.save(path)
    restored = FeaturePreprocessor.load(path)

    np.testing.assert_allclose(
        restored.transform(training),
        preprocessor.transform(training),
    )
    assert restored.to_dict()["strategy"]["missing_values"] == "training_median"


def test_grouped_split_has_no_host_star_overlap_and_is_reproducible() -> None:
    """Keep every KOI from one kepid in exactly one labelled partition."""
    groups = np.array(
        ["star-1", "star-1", "star-2", "star-3", "star-4", "star-5"]
        + [f"star-{index}" for index in range(6, 31)]
    )

    first = grouped_split_indices(groups, seed=37)
    second = grouped_split_indices(groups, seed=37)
    for first_indices, second_indices in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_indices, second_indices)

    split_groups = [set(groups[indices]) for indices in first]
    assert split_groups[0].isdisjoint(split_groups[1])
    assert split_groups[0].isdisjoint(split_groups[2])
    assert split_groups[1].isdisjoint(split_groups[2])
    assert sum(len(indices) for indices in first) == len(groups)

