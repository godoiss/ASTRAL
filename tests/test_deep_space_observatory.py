"""Scientific and structural tests for Stage 05 observational infrastructure."""

import numpy as np
import pytest

from src.core.mlp import MLP
from src.lab.visualization import activation_frame, activation_sample_count
from src.observatory.catalog import find_koi, load_observatory_catalog
from src.observatory.config import ObservatoryConfig
from src.observatory.dataset import median_bin_view, validate_group_separation
from src.observatory.models import NeuralLightCurveExample
from src.observatory.processing import bjd_to_bkjd, bkjd_to_bjd


def test_koi_753_metadata_and_time_system_are_preserved() -> None:
    """Read the requested candidate and verify the documented BKJD offset."""
    catalog = load_observatory_catalog()
    matches = find_koi("KOI-753.01", catalog)

    assert len(matches) == 1
    metadata = matches[0]
    assert metadata.object_id == "K00753.01"
    assert metadata.kic == 10811496
    assert metadata.disposition == "CANDIDATE"
    assert metadata.epoch_bkjd == pytest.approx(175.850252)
    assert bjd_to_bkjd(bkjd_to_bjd(metadata.epoch_bkjd)) == pytest.approx(
        metadata.epoch_bkjd
    )


def test_fixed_views_keep_unobserved_bins_masked() -> None:
    """Never interpolate a fictitious signal into empty phase bins."""
    coordinates = np.asarray([-0.45, -0.44, 0.42], dtype=np.float64)
    flux = np.asarray([0.99, 1.01, 0.98], dtype=np.float64)
    centers, values, mask = median_bin_view(coordinates, flux, -0.5, 0.5, 10)

    assert centers.shape == values.shape == mask.shape == (10,)
    assert int(mask.sum()) == 2
    assert np.all(np.isnan(values[~mask]))
    assert np.all(np.isfinite(values[mask]))


def test_group_separation_rejects_conflicting_labelled_star() -> None:
    """Prevent a KIC from crossing labelled ANN-02 partitions."""
    def example(split: str) -> NeuralLightCurveExample:
        return NeuralLightCurveExample(
            object_id=f"object-{split}",
            kic=123,
            split=split,
            class_label=1,
            global_coordinates=np.arange(3, dtype=float),
            global_flux=np.ones(3),
            global_mask=np.ones(3, dtype=bool),
            local_coordinates_hours=np.arange(2, dtype=float),
            local_flux=np.ones(2),
            local_mask=np.ones(2, dtype=bool),
        )

    with pytest.raises(ValueError, match="Host-star split conflicts"):
        validate_group_separation([example("train"), example("test")])


def test_activation_table_selects_one_batch_record_and_real_neurons() -> None:
    """Do not flatten multiple records into fictitious neuron identities."""
    model = MLP(seed=19)
    inputs = np.vstack([np.arange(6), np.arange(6) + 10]).astype(np.float64)
    model.forward(inputs, retain_cache=True)

    assert activation_sample_count(model.activations) == 2
    first = activation_frame(model, sample_index=0)
    second = activation_frame(model, sample_index=1)
    assert len(first) == len(second) == sum(model.layer_sizes) == 19
    assert first["sample_index"].unique().tolist() == [0]
    assert second["sample_index"].unique().tolist() == [1]
    assert first.groupby("layer").size().to_dict() == {
        "input": 6,
        "hidden_1": 8,
        "hidden_2": 4,
        "output": 1,
    }
    assert not np.allclose(first["activation"], second["activation"])


def test_observatory_config_enforces_bounded_downloads() -> None:
    with pytest.raises(ValueError, match="max_products"):
        ObservatoryConfig(max_products=20)
