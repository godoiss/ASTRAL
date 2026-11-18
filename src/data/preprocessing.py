"""Leakage-safe preparation of Kepler KOI data for ANN-01."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
StringArray = NDArray[np.str_]

FEATURE_COLUMNS = (
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_model_snr",
    "koi_teq",
)
GROUP_COLUMN = "kepid"
OBJECT_ID_COLUMN = "kepoi_name"
TARGET_COLUMN = "koi_disposition"
LABEL_MAPPING = {"FALSE POSITIVE": 0.0, "CONFIRMED": 1.0}
CANDIDATE_LABEL = "CANDIDATE"
REQUIRED_TRAINING_COLUMNS = (
    GROUP_COLUMN,
    OBJECT_ID_COLUMN,
    TARGET_COLUMN,
    *FEATURE_COLUMNS,
)


class PreprocessingError(RuntimeError):
    """Raised when source data cannot be prepared without ambiguity."""


@dataclass(frozen=True)
class DataPartition:
    """A transformed partition plus its original identity metadata."""

    features: FloatArray
    targets: FloatArray | None
    group_ids: StringArray
    object_ids: StringArray
    dispositions: StringArray

    @property
    def size(self) -> int:
        """Return the number of records in the partition."""
        return int(self.features.shape[0])

    @property
    def group_count(self) -> int:
        """Return the number of distinct host-star identifiers."""
        return int(np.unique(self.group_ids).size)


@dataclass(frozen=True)
class PreparedDataset:
    """All leakage-safe ANN-01 partitions and their fitted transformer."""

    train: DataPartition
    validation: DataPartition
    test: DataPartition
    candidates: DataPartition
    preprocessor: FeaturePreprocessor


class FeaturePreprocessor:
    """Median imputation followed by z-score standardization.

    The object must be fitted using the training partition only. Stored medians,
    means, and standard deviations are then reused unchanged for validation,
    test, candidate, and future inference records.
    """

    def __init__(self, feature_names: tuple[str, ...] = FEATURE_COLUMNS) -> None:
        self.feature_names = tuple(feature_names)
        self.medians: FloatArray | None = None
        self.means: FloatArray | None = None
        self.scales: FloatArray | None = None
        self.fitted_on_rows = 0

    @property
    def is_fitted(self) -> bool:
        """Return whether all transformation parameters are available."""
        return all(
            value is not None for value in (self.medians, self.means, self.scales)
        )

    def _numeric_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        missing_columns = [
            column for column in self.feature_names if column not in data.columns
        ]
        if missing_columns:
            raise PreprocessingError(
                "Missing feature columns: " + ", ".join(missing_columns)
            )
        numeric = data.loc[:, self.feature_names].apply(pd.to_numeric, errors="coerce")
        return numeric.replace([np.inf, -np.inf], np.nan)

    def fit(self, training_data: pd.DataFrame) -> FeaturePreprocessor:
        """Fit imputation and scaling parameters on training records only."""
        if training_data.empty:
            raise PreprocessingError("Cannot fit preprocessing on an empty dataset")
        numeric = self._numeric_frame(training_data)
        medians = numeric.median(axis=0, skipna=True).to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(medians)):
            invalid = [
                name
                for name, value in zip(self.feature_names, medians, strict=True)
                if not np.isfinite(value)
            ]
            raise PreprocessingError(
                "Training data has no finite values for: " + ", ".join(invalid)
            )

        imputed = numeric.fillna(
            pd.Series(medians, index=self.feature_names, dtype=np.float64)
        )
        means = imputed.mean(axis=0).to_numpy(dtype=np.float64)
        scales = imputed.std(axis=0, ddof=0).to_numpy(dtype=np.float64)
        scales = np.where(scales < 1e-12, 1.0, scales)

        self.medians = medians
        self.means = means
        self.scales = scales
        self.fitted_on_rows = len(training_data)
        return self

    def transform(self, data: pd.DataFrame) -> FloatArray:
        """Impute and standardize records using previously fitted parameters."""
        if not self.is_fitted:
            raise PreprocessingError("Preprocessor must be fitted before transform")
        assert self.medians is not None
        assert self.means is not None
        assert self.scales is not None

        numeric = self._numeric_frame(data)
        imputed = numeric.fillna(
            pd.Series(self.medians, index=self.feature_names, dtype=np.float64)
        ).to_numpy(dtype=np.float64)
        transformed = (imputed - self.means) / self.scales
        if not np.all(np.isfinite(transformed)):
            raise PreprocessingError("Transformation produced non-finite values")
        return transformed

    def fit_transform(self, training_data: pd.DataFrame) -> FloatArray:
        """Fit on and transform the training partition."""
        return self.fit(training_data).transform(training_data)

    def to_dict(self) -> dict[str, Any]:
        """Return fitted parameters as a JSON-ready dictionary."""
        if not self.is_fitted:
            raise PreprocessingError("Cannot serialize an unfitted preprocessor")
        assert self.medians is not None
        assert self.means is not None
        assert self.scales is not None
        return {
            "schema_version": 1,
            "feature_names": list(self.feature_names),
            "strategy": {
                "missing_values": "training_median",
                "normalization": "training_z_score",
            },
            "medians": self.medians.tolist(),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "fitted_on_rows": self.fitted_on_rows,
        }

    def save(self, path: Path) -> Path:
        """Atomically persist preprocessing parameters as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f"{path.stem}_",
            suffix=".json.tmp",
            dir=path.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            temporary_path.write_text(
                json.dumps(
                    self.to_dict(),
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return path

    @classmethod
    def load(cls, path: Path) -> FeaturePreprocessor:
        """Load fitted preprocessing parameters from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        preprocessor = cls(tuple(payload["feature_names"]))
        expected_size = len(preprocessor.feature_names)
        for key in ("medians", "means", "scales"):
            if len(payload[key]) != expected_size:
                raise PreprocessingError(f"Invalid {key} length in {path}")
        preprocessor.medians = np.asarray(payload["medians"], dtype=np.float64)
        preprocessor.means = np.asarray(payload["means"], dtype=np.float64)
        preprocessor.scales = np.asarray(payload["scales"], dtype=np.float64)
        preprocessor.fitted_on_rows = int(payload["fitted_on_rows"])
        if not np.all(preprocessor.scales > 0.0):
            raise PreprocessingError("Persisted preprocessing scales must be positive")
        return preprocessor


def _validate_ratios(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> None:
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio <= 0.0 for ratio in ratios):
        raise ValueError("Every split ratio must be positive")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Train, validation, and test ratios must sum to one")


def grouped_split_indices(
    group_ids: StringArray,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Split row indices by whole host-star groups with deterministic shuffling."""
    _validate_ratios(train_ratio, validation_ratio, test_ratio)
    group_ids = np.asarray(group_ids, dtype=str)
    unique_groups, counts = np.unique(group_ids, return_counts=True)
    if unique_groups.size < 3:
        raise PreprocessingError("At least three host-star groups are required")

    generator = np.random.default_rng(seed)
    order = generator.permutation(unique_groups.size)
    shuffled_groups = unique_groups[order]
    shuffled_counts = counts[order]
    row_count = group_ids.size
    train_target = row_count * train_ratio
    validation_target = row_count * validation_ratio

    train_groups: list[str] = []
    validation_groups: list[str] = []
    test_groups: list[str] = []
    train_rows = 0
    validation_rows = 0
    for group, count in zip(shuffled_groups, shuffled_counts, strict=True):
        if train_rows < train_target:
            train_groups.append(group)
            train_rows += int(count)
        elif validation_rows < validation_target:
            validation_groups.append(group)
            validation_rows += int(count)
        else:
            test_groups.append(group)

    if not train_groups or not validation_groups or not test_groups:
        raise PreprocessingError("Grouped split produced an empty partition")

    train_indices = np.flatnonzero(np.isin(group_ids, train_groups)).astype(np.int64)
    validation_indices = np.flatnonzero(
        np.isin(group_ids, validation_groups)
    ).astype(np.int64)
    test_indices = np.flatnonzero(np.isin(group_ids, test_groups)).astype(np.int64)
    return train_indices, validation_indices, test_indices


def _partition_from_frame(
    frame: pd.DataFrame,
    preprocessor: FeaturePreprocessor,
    targets: FloatArray | None,
) -> DataPartition:
    return DataPartition(
        features=preprocessor.transform(frame),
        targets=targets,
        group_ids=frame[GROUP_COLUMN].astype(str).to_numpy(dtype=str),
        object_ids=frame[OBJECT_ID_COLUMN].astype(str).to_numpy(dtype=str),
        dispositions=frame[TARGET_COLUMN].astype(str).to_numpy(dtype=str),
    )


def load_training_frame(path: Path) -> pd.DataFrame:
    """Load only ANN-01 fields and validate identifiers and dispositions."""
    try:
        frame = pd.read_csv(path, usecols=list(REQUIRED_TRAINING_COLUMNS))
    except (OSError, ValueError) as exc:
        raise PreprocessingError(f"Could not load ANN-01 data from {path}: {exc}") from exc
    if frame.empty:
        raise PreprocessingError(f"Training dataset is empty: {path}")
    if frame[GROUP_COLUMN].isna().any() or frame[OBJECT_ID_COLUMN].isna().any():
        raise PreprocessingError("kepid and kepoi_name must be present for every record")
    return frame


def prepare_kepler_dataset(
    path: Path,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> PreparedDataset:
    """Create grouped ANN-01 partitions and fit preprocessing on training only."""
    frame = load_training_frame(path)
    labelled = frame[frame[TARGET_COLUMN].isin(LABEL_MAPPING)].copy()
    candidates = frame[frame[TARGET_COLUMN] == CANDIDATE_LABEL].copy()
    if labelled.empty or candidates.empty:
        raise PreprocessingError(
            "Dataset must contain labelled and CANDIDATE KOI records"
        )

    group_ids = labelled[GROUP_COLUMN].astype(str).to_numpy(dtype=str)
    train_indices, validation_indices, test_indices = grouped_split_indices(
        group_ids,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    targets = (
        labelled[TARGET_COLUMN]
        .map(LABEL_MAPPING)
        .to_numpy(dtype=np.float64)
        .reshape(-1, 1)
    )

    train_frame = labelled.iloc[train_indices]
    validation_frame = labelled.iloc[validation_indices]
    test_frame = labelled.iloc[test_indices]
    preprocessor = FeaturePreprocessor()
    preprocessor.fit(train_frame)

    train = _partition_from_frame(
        train_frame,
        preprocessor,
        targets[train_indices],
    )
    validation = _partition_from_frame(
        validation_frame,
        preprocessor,
        targets[validation_indices],
    )
    test = _partition_from_frame(
        test_frame,
        preprocessor,
        targets[test_indices],
    )
    candidate_partition = _partition_from_frame(candidates, preprocessor, targets=None)
    return PreparedDataset(
        train=train,
        validation=validation,
        test=test,
        candidates=candidate_partition,
        preprocessor=preprocessor,
    )


def load_inference_partition(
    path: Path,
    preprocessor: FeaturePreprocessor,
    candidates_only: bool = True,
) -> DataPartition:
    """Load and transform inference records with persisted training parameters."""
    frame = load_training_frame(path)
    if candidates_only:
        frame = frame[frame[TARGET_COLUMN] == CANDIDATE_LABEL].copy()
    if frame.empty:
        raise PreprocessingError("No records matched the inference selection")
    mapped_targets = frame[TARGET_COLUMN].map(LABEL_MAPPING)
    targets: FloatArray | None
    if mapped_targets.notna().all():
        targets = mapped_targets.to_numpy(dtype=np.float64).reshape(-1, 1)
    else:
        targets = None
    return _partition_from_frame(frame, preprocessor, targets=targets)


def export_split_manifest(dataset: PreparedDataset, path: Path) -> Path:
    """Persist identities and split assignments for leakage auditing."""
    records: list[pd.DataFrame] = []
    for split_name, partition in (
        ("train", dataset.train),
        ("validation", dataset.validation),
        ("test", dataset.test),
        ("candidate", dataset.candidates),
    ):
        records.append(
            pd.DataFrame(
                {
                    OBJECT_ID_COLUMN: partition.object_ids,
                    GROUP_COLUMN: partition.group_ids,
                    TARGET_COLUMN: partition.dispositions,
                    "split": split_name,
                }
            )
        )
    manifest = pd.concat(records, ignore_index=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    return path

