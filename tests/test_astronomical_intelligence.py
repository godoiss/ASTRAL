"""Scientific guardrails and real calculations introduced in Stage 07."""

from __future__ import annotations

import numpy as np
import torch

from src.deep_learning.config import ANN02Config, TrainingConfig
from src.deep_learning.dataset import KeplerSignalDataset
from src.deep_learning.model import DualBranchCNN
from src.deep_learning.training import ANN02Bundle, load_ann02
from src.intelligence.comparison import compare_ann01_ann02
from src.intelligence.dataset import build_dataset_intelligence
from src.intelligence.experiments import default_trials
from src.intelligence.explainability import explain_ann01_local, explain_ann02_signal
from src.intelligence.tess import tess_compatibility_report
from src.lab.services import load_dataset_frame, load_scientific_model


def test_dataset_intelligence_quantifies_real_selection_bias() -> None:
    """The report must expose the period and SNR selection effects, not hide them."""
    report = build_dataset_intelligence(report_path=None, records_path=None)

    assert report["dataset"]["accepted"] >= 41
    assert report["dataset"]["stars"] == report["dataset"]["accepted"]
    assert report["bias_analysis"]["fraction_eligible_removed_by_period_limit"] > 0
    assert report["bias_analysis"]["selected_model_snr"]["median"] > report["bias_analysis"]["eligible_model_snr"]["median"]
    assert report["guardrails"]["frozen_test_accessed"] is False


def test_ann01_local_explanation_uses_real_model_sensitivity() -> None:
    """ANN-01 explanations must produce finite, model-derived feature effects."""
    state = load_scientific_model()
    assert state is not None
    frame = load_dataset_frame()
    result = explain_ann01_local(state, frame.iloc[0])

    assert 0.0 <= result["score"] <= 1.0
    assert len(result["features"]) == 6
    assert all(np.isfinite(item["normalized_gradient"]) for item in result["features"])


def test_ann02_explanation_uses_gradients_and_real_occlusion() -> None:
    """Signal regions must be recomputed from the actual CNN, not simulated."""
    config = ANN02Config()
    model = DualBranchCNN(config)
    bundle = ANN02Bundle(model, config, TrainingConfig(device="cpu"), {}, torch.device("cpu"))
    sample = {
        "object_id": "fixture",
        "kic": torch.tensor(123, dtype=torch.int64),
        "global_view": torch.vstack((torch.linspace(-1, 1, 2001), torch.ones(2001))),
        "local_view": torch.vstack((torch.linspace(-1, 1, 201), torch.ones(201))),
    }
    result = explain_ann02_signal(bundle, sample, global_regions=4, local_regions=2)

    assert 0.0 <= result["score"] <= 1.0
    assert len(result["regions"]) == 6
    assert len(result["saliency"]["global"]) == 2001
    assert all(np.isfinite(item["score_delta"]) for item in result["regions"])


def test_model_comparison_uses_one_shared_validation_cohort() -> None:
    """ANN metrics must be recomputed on identical KOI identities."""
    ann01 = load_scientific_model()
    ann02 = load_ann02(device_preference="cpu")
    assert ann01 is not None and ann02 is not None
    report = compare_ann01_ann02(
        ann01,
        ann02,
        load_dataset_frame(),
        report_path=None,
        rows_path=None,
    )

    assert report["cohort"]["records"] == len(KeplerSignalDataset(split="validation"))
    assert report["cohort"]["test_accessed"] is False
    assert len(report["rows"]) == report["cohort"]["records"]


def test_campaign_and_tess_stay_within_stage07_guardrails() -> None:
    """Campaign definitions stay compact and the TESS study remains isolated."""
    trials = default_trials()
    tess = tess_compatibility_report()

    assert 2 <= len(trials) <= 5
    assert len({trial.name for trial in trials}) == len(trials)
    assert tess["mixed_with_kepler_training"] is False
    assert any(item["topic"] == "time zero-point" for item in tess["differences"])
