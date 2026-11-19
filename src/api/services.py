"""Read-only application services over existing ASTRAL scientific modules."""

from __future__ import annotations

import hashlib
import json
import re
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.api.errors import ApiError
from src.data.preprocessing import FEATURE_COLUMNS, OBJECT_ID_COLUMN
from src.deep_learning.dataset import DEFAULT_DATASET_PATH, KeplerSignalDataset
from src.deep_learning.training import ANN02Bundle, infer_ann02, load_ann02
from src.lab.services import (
    DEFAULT_DATA_PATH,
    LabModelState,
    dataset_overview,
    infer_record,
    load_dataset_frame,
    load_scientific_model,
)
from src.observatory.catalog import find_koi, load_observatory_catalog
from src.observatory.persistence import (
    PROCESSED_DIRECTORY,
    PROVENANCE_DIRECTORY,
    RAW_DIRECTORY,
    artifact_is_cached,
    load_artifact,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE07_ROOT = PROJECT_ROOT / "experiments" / "stage07"
CAMPAIGN_PATH = STAGE07_ROOT / "latest_campaign.json"
DATASET_REPORT_PATH = STAGE07_ROOT / "dataset_intelligence.json"
COMPARISON_PATH = STAGE07_ROOT / "model_comparison.json"
EXPLAINABILITY_PATH = STAGE07_ROOT / "explainability.json"
TESS_REPORT_PATH = STAGE07_ROOT / "tess_pilot.json"
SPLIT_MANIFEST_PATH = PROJECT_ROOT / "experiments" / "stage04" / "split_manifest.csv"

_PRIVATE_KEYS = {
    "path",
    "dataset_path",
    "model_directory",
    "source_path",
    "raw_path",
    "processed_path",
    "provenance_path",
    "python_executable",
    "raw_npz",
    "processed_npz",
}
_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9_.-]{3,96}$")


def load_json(path: Path) -> dict[str, Any]:
    """Load a required JSON artifact and map failures to a public error."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ApiError(404, "artifact_not_found", "The requested scientific artifact is unavailable.") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ApiError(503, "artifact_unreadable", "A persisted scientific artifact could not be read.") from exc
    if not isinstance(payload, dict):
        raise ApiError(503, "artifact_invalid", "A persisted scientific artifact has an invalid format.")
    return payload


def public_payload(value: Any) -> Any:
    """Recursively remove filesystem paths and convert scientific scalar types."""
    if isinstance(value, dict):
        return {
            str(key): public_payload(item)
            for key, item in value.items()
            if str(key) not in _PRIVATE_KEYS
            and not str(key).endswith("_path")
            and not str(key).endswith("_directory")
        }
    if isinstance(value, (list, tuple)):
        return [public_payload(item) for item in value]
    if isinstance(value, Path):
        return None
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        project = str(PROJECT_ROOT).replace("\\", "/")
        if normalized.lower().startswith(project.lower()) or re.match(r"^[A-Za-z]:/", normalized):
            return None
        return value
    if isinstance(value, np.generic):
        return public_payload(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_indices(records: int, limit: int) -> np.ndarray:
    if records <= limit:
        return np.arange(records, dtype=np.int64)
    return np.unique(np.linspace(0, records - 1, limit, dtype=np.int64))


def _finite_list(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


def _tensor_state(name: str, role: str, values: Any) -> dict[str, Any]:
    """Return a JSON-safe real tensor with summary statistics."""
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return {
        "name": name,
        "role": role,
        "shape": list(array.shape),
        "values": public_payload(array.tolist()),
        "minimum": float(finite.min()) if finite.size else None,
        "maximum": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "standard_deviation": float(finite.std()) if finite.size else None,
    }


class ScientificService:
    """Facade that guarantees read-only access and blocks frozen test identities."""

    @cached_property
    def catalog(self) -> pd.DataFrame:
        frame = load_observatory_catalog()
        return frame.loc[~frame["split"].eq("test")].copy()

    @cached_property
    def ann01_frame(self) -> pd.DataFrame:
        frame = load_dataset_frame()
        safe_ids = set(self.catalog["kepoi_name"].astype(str))
        return frame.loc[frame[OBJECT_ID_COLUMN].astype(str).isin(safe_ids)].copy()

    @cached_property
    def ann01_overview(self) -> dict[str, Any]:
        """Compute the catalog checksum once per process, not once per request."""
        return dataset_overview(self.ann01_frame, DEFAULT_DATA_PATH)

    def cached_observation_ids(self) -> frozenset[str]:
        """Index complete artifacts and refresh when a cache directory changes."""
        stamp = tuple(
            directory.stat().st_mtime_ns if directory.exists() else 0
            for directory in (RAW_DIRECTORY, PROCESSED_DIRECTORY, PROVENANCE_DIRECTORY)
        )
        if getattr(self, "_observation_cache_stamp", None) == stamp:
            return self._observation_cache_ids
        raw = {path.stem for path in RAW_DIRECTORY.glob("*.npz")}
        processed = {path.stem for path in PROCESSED_DIRECTORY.glob("*.npz")}
        provenance = {path.stem for path in PROVENANCE_DIRECTORY.glob("*.json")}
        self._observation_cache_stamp = stamp
        self._observation_cache_ids = frozenset(raw & processed & provenance)
        return self._observation_cache_ids

    @cached_property
    def ann01(self) -> LabModelState:
        state = load_scientific_model()
        if state is None:
            raise ApiError(503, "model_unavailable", "The persisted ANN-01 model is unavailable.")
        return state

    @cached_property
    def stage07_campaign(self) -> dict[str, Any]:
        return load_json(CAMPAIGN_PATH)

    def _selected_ann02_directory(self) -> Path:
        experiment_id = str(self.stage07_campaign["selected_experiment_id"])
        for experiment_path in (PROJECT_ROOT / "models" / "ann02").rglob("experiment.json"):
            try:
                payload = json.loads(experiment_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if str(payload.get("experiment_id")) == experiment_id:
                directory = experiment_path.parent.resolve()
                directory.relative_to(PROJECT_ROOT.resolve())
                return directory
        raise ApiError(503, "model_unavailable", "The Stage 07 selected ANN-02 checkpoint is unavailable.")

    @cached_property
    def ann02(self) -> ANN02Bundle:
        bundle = load_ann02(self._selected_ann02_directory(), device_preference="cpu")
        if bundle is None:
            raise ApiError(503, "model_unavailable", "The Stage 07 selected ANN-02 model is unavailable.")
        return bundle

    @cached_property
    def ann02_samples(self) -> dict[str, dict[str, Any]]:
        if not DEFAULT_DATASET_PATH.is_file():
            raise ApiError(503, "dataset_unavailable", "The local ANN-02 signal dataset is unavailable.")
        dataset = KeplerSignalDataset(split=None)
        samples = {str(dataset[index]["object_id"]): dataset[index] for index in range(len(dataset))}
        if any(str(sample["split"]) == "test" for sample in samples.values()):
            raise ApiError(503, "scientific_guardrail_violation", "Frozen test records were found in the serving dataset.")
        return samples

    def availability(self) -> dict[str, dict[str, bool]]:
        ann01_available = all(
            (PROJECT_ROOT / "experiments" / "stage04" / "selected_model" / name).is_file()
            for name in ("model.npz", "preprocessor.json", "experiment_config.json")
        )
        try:
            ann02_available = self._selected_ann02_directory().is_dir()
        except ApiError:
            ann02_available = False
        return {
            "models": {"ann-01": ann01_available, "ann-02": ann02_available},
            "artifacts": {
                "dataset_intelligence": DATASET_REPORT_PATH.is_file(),
                "model_comparison": COMPARISON_PATH.is_file(),
                "explainability": EXPLAINABILITY_PATH.is_file(),
                "tess_pilot": TESS_REPORT_PATH.is_file(),
            },
        }

    def datasets(self) -> list[dict[str, Any]]:
        overview = self.ann01_overview
        ann02 = load_json(DATASET_REPORT_PATH)
        tess = load_json(TESS_REPORT_PATH)
        return [
            {
                "id": "kepler-koi-cumulative",
                "mission": "Kepler",
                "kind": "catalog",
                "records_available_to_api": int(len(self.ann01_frame)),
                "objects": int(self.ann01_frame[OBJECT_ID_COLUMN].nunique()),
                "version_id": overview["version_id"],
                "frozen_test_excluded": True,
            },
            {
                "id": "ann02-kepler-q3",
                "mission": "Kepler",
                "kind": "processed_light_curves",
                **public_payload(ann02["dataset"]),
                "class_distribution": ann02["class_distribution"],
                "frozen_test_excluded": True,
            },
            {
                "id": "tess-experimental-pilot",
                "mission": "TESS",
                "kind": "isolated_experimental_light_curve",
                "target": tess.get("target"),
                "downloaded": bool(tess.get("downloaded")),
                "observation": public_payload(tess.get("observation", {})),
                "mixed_with_kepler_training": False,
            },
        ]

    def observatory_summary(self) -> dict[str, Any]:
        report = public_payload(load_json(DATASET_REPORT_PATH))
        return {
            "dataset": report["dataset"],
            "class_distribution": report["class_distribution"],
            "quality": report["quality"],
            "astronomy": report["astronomy"],
            "exclusions": report["exclusions"],
            "bias_analysis": report["bias_analysis"],
            "guardrails": report["guardrails"],
        }

    def list_objects(
        self,
        query: str | None,
        disposition: str | None,
        observed_only: bool,
        offset: int,
        limit: int,
    ) -> tuple[list[dict[str, Any]], int]:
        frame = self.catalog
        if disposition:
            frame = frame.loc[frame["koi_disposition"].eq(disposition)]
        if query:
            value = query.strip()
            if value:
                mask = (
                    frame["kepoi_name"].astype(str).str.contains(value, case=False, regex=False)
                    | frame["display_name"].astype(str).str.contains(value, case=False, regex=False)
                    | frame["kepler_name"].fillna("").astype(str).str.contains(value, case=False, regex=False)
                    | frame["kepid"].astype(str).str.contains(value, regex=False)
                )
                frame = frame.loc[mask]
        if observed_only:
            frame = frame.loc[frame["kepoi_name"].astype(str).isin(self.cached_observation_ids())]
        total = int(len(frame))
        selected = frame.iloc[offset : offset + limit]
        items = [self._object_summary(row) for _, row in selected.iterrows()]
        return items, total

    def _object_summary(self, row: pd.Series) -> dict[str, Any]:
        return {
            "object_id": str(row["kepoi_name"]),
            "display_name": str(row["display_name"]),
            "kic": int(row["kepid"]),
            "kepler_name": None if pd.isna(row["kepler_name"]) else str(row["kepler_name"]),
            "catalog_disposition": str(row["koi_disposition"]),
            "split": str(row["split"]),
            "period_days": None if pd.isna(row["koi_period"]) else float(row["koi_period"]),
            "has_cached_observation": str(row["kepoi_name"]) in self.cached_observation_ids(),
            "ra_deg": None if pd.isna(row.get("ra")) else float(row["ra"]),
            "dec_deg": None if pd.isna(row.get("dec")) else float(row["dec"]),
        }

    def object_metadata(self, identifier: str) -> dict[str, Any]:
        matches = find_koi(identifier, self.catalog)
        if not matches:
            raise ApiError(404, "object_not_found", "No accessible astronomical object matches the identifier.")
        if len(matches) > 1:
            raise ApiError(409, "ambiguous_identifier", "The identifier matches multiple astronomical objects.")
        metadata = matches[0]
        if metadata.split == "test":
            raise ApiError(404, "object_not_found", "No accessible astronomical object matches the identifier.")
        row = self.catalog.loc[self.catalog["kepoi_name"].astype(str).eq(metadata.object_id)].iloc[0]
        return {
            **metadata.to_dict(),
            "has_cached_observation": metadata.object_id in self.cached_observation_ids(),
            "ra_deg": None if pd.isna(row.get("ra")) else float(row["ra"]),
            "dec_deg": None if pd.isna(row.get("dec")) else float(row["dec"]),
        }

    def _artifact(self, identifier: str):
        metadata_dict = self.object_metadata(identifier)
        matches = find_koi(metadata_dict["object_id"], self.catalog)
        metadata = matches[0]
        if not artifact_is_cached(metadata.object_id):
            raise ApiError(404, "observation_not_found", "No cached local observation exists for this object.")
        try:
            return load_artifact(metadata)
        except (OSError, ValueError, KeyError) as exc:
            raise ApiError(503, "observation_unreadable", "The cached observation could not be read.") from exc

    def observation(self, identifier: str) -> dict[str, Any]:
        artifact = self._artifact(identifier)
        provenance = public_payload(artifact.provenance)
        return {
            "object": artifact.metadata.to_dict(),
            "mission": artifact.mission,
            "author": artifact.author,
            "cadence_seconds": artifact.cadence_seconds,
            "observation_start_bkjd": artifact.observation_start_bkjd,
            "observation_end_bkjd": artifact.observation_end_bkjd,
            "observation_span_days": artifact.observation_end_bkjd - artifact.observation_start_bkjd,
            "segments": list(artifact.quarters),
            "raw_records": artifact.raw_records,
            "processed_records": artifact.processed_records,
            "provenance": provenance,
        }

    def light_curve(
        self,
        identifier: str,
        view: Literal["raw", "processed", "phase", "local"],
        limit: int,
    ) -> dict[str, Any]:
        artifact = self._artifact(identifier)
        if view == "raw":
            arrays: dict[str, np.ndarray] = {
                "time_bkjd": artifact.raw_time_bkjd,
                "flux": artifact.raw_flux,
                "flux_error": artifact.raw_flux_error,
                "quality": artifact.raw_quality,
            }
        elif view == "processed":
            arrays = {
                "time_bkjd": artifact.clean_time_bkjd,
                "normalized_flux": artifact.normalized_flux,
                "detrended_flux": artifact.detrended_flux,
            }
        elif view == "phase":
            order = np.argsort(artifact.phase)
            arrays = {
                "phase": artifact.phase[order],
                "detrended_flux": artifact.detrended_flux[order],
            }
        else:
            arrays = {
                "phase_hours": artifact.local_phase_hours,
                "flux": artifact.local_flux,
            }
        records = min(array.size for array in arrays.values())
        indices = _sample_indices(records, limit)
        data: dict[str, Any] = {}
        for name, values in arrays.items():
            selected = np.asarray(values)[:records][indices]
            data[name] = selected.astype(int).tolist() if name == "quality" else _finite_list(selected.astype(float))
        return {
            "object_id": artifact.metadata.object_id,
            "mission": artifact.mission,
            "view": view,
            "records_total": records,
            "records_returned": int(indices.size),
            "downsampled": bool(records > indices.size),
            "data": data,
        }

    def models(self) -> list[dict[str, Any]]:
        comparison = load_json(COMPARISON_PATH)
        return [
            {
                "id": "ann-01",
                "name": self.ann01.model.engine_name,
                "framework": "NumPy",
                "architecture": self.ann01.model.architecture,
                "trainable_parameters": self.ann01.model.parameter_count,
                "experiment_id": "stability-004-seed-2026",
                "served_checkpoint": "Stage 04 selected scientific model",
                "shared_validation_metrics": comparison["ann01_metrics"],
                "definitive_scientific_validation": False,
            },
            {
                "id": "ann-02",
                "name": self.ann02.model.architecture()["name"],
                "framework": "PyTorch",
                "architecture": self.ann02.model.architecture(),
                "trainable_parameters": self.ann02.model.architecture()["trainable_parameters"],
                "experiment_id": self.ann02.experiment["experiment_id"],
                "served_checkpoint": self.stage07_campaign["selected_trial"],
                "seed": self.ann02.training_config.seed,
                "shared_validation_metrics": comparison["ann02_metrics"],
                "definitive_scientific_validation": False,
            },
        ]

    def model(self, model_id: str) -> dict[str, Any]:
        match = next((item for item in self.models() if item["id"] == model_id), None)
        if match is None:
            raise ApiError(404, "model_not_found", "Unknown ASTRAL model identifier.")
        return match

    def history(self, model_id: str, offset: int, limit: int) -> tuple[list[dict[str, Any]], int]:
        if model_id == "ann-01":
            history = self.ann01.model.training_history
        elif model_id == "ann-02":
            history = self.ann02.experiment.get("history", [])
        else:
            raise ApiError(404, "model_not_found", "Unknown ASTRAL model identifier.")
        return public_payload(history[offset : offset + limit]), len(history)

    def inference(self, model_id: str, object_id: str) -> dict[str, Any]:
        metadata = self.object_metadata(object_id)
        canonical = str(metadata["object_id"])
        notice = "Scores are uncalibrated model outputs and are not exoplanet confirmations or discoveries."
        if model_id == "ann-01":
            matches = self.ann01_frame.loc[self.ann01_frame[OBJECT_ID_COLUMN].eq(canonical)]
            if len(matches) != 1:
                raise ApiError(422, "incompatible_object", "The object is not compatible with ANN-01 inference.")
            result = infer_record(self.ann01, matches.iloc[0])
            return {
                "model_id": "ann-01",
                "model_experiment_id": "stability-004-seed-2026",
                "object_id": result.object_id,
                "kic": int(result.group_id),
                "original_disposition": result.original_disposition,
                "score": result.score,
                "threshold": result.threshold,
                "predicted_class": result.predicted_class,
                "input_kind": "Kepler KOI catalog features",
                "details": {
                    "raw_features": result.raw_features,
                    "normalized_features": result.normalized_features,
                    "activations": [values[0].tolist() for values in result.activations],
                },
                "scientific_notice": notice,
            }
        if model_id == "ann-02":
            sample = self.ann02_samples.get(canonical)
            if sample is None:
                raise ApiError(422, "incompatible_object", "The object has no ANN-02 signal representation.")
            result = infer_ann02(self.ann02, sample)
            return {
                "model_id": "ann-02",
                "model_experiment_id": self.ann02.experiment["experiment_id"],
                "object_id": canonical,
                "kic": int(sample["kic"]),
                "original_disposition": str(metadata["disposition"]),
                "score": result["score"],
                "threshold": result["threshold"],
                "predicted_class": result["predicted_class"],
                "input_kind": "Kepler Quarter 3 global/local luminosity views",
                "details": {
                    "split": str(sample["split"]),
                    "device": result["device"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "feature_map_shapes": {
                        name: list(values.shape) for name, values in result["feature_maps"].items()
                    },
                },
                "scientific_notice": notice,
            }
        raise ApiError(404, "model_not_found", "Unknown ASTRAL model identifier.")

    def neural_visualization(self, model_id: str, object_id: str) -> dict[str, Any]:
        """Expose bounded, real neural tensors for the publication interface."""
        metadata = self.object_metadata(object_id)
        canonical = str(metadata["object_id"])
        if model_id == "ann-01":
            matches = self.ann01_frame.loc[self.ann01_frame[OBJECT_ID_COLUMN].eq(canonical)]
            if len(matches) != 1:
                raise ApiError(422, "incompatible_object", "The object is not compatible with ANN-01 inference.")
            result = infer_record(self.ann01, matches.iloc[0])
            parameters: list[dict[str, Any]] = []
            for index, (weights, biases) in enumerate(
                zip(self.ann01.model.weights, self.ann01.model.biases, strict=True), start=1
            ):
                parameters.append(_tensor_state(f"layer_{index}.weights", "weight", weights))
                parameters.append(_tensor_state(f"layer_{index}.biases", "bias", biases.reshape(-1)))
            names = [str(layer["name"]) for layer in self.ann01.model.architecture]
            activations = [
                _tensor_state(f"{names[index]}.activation", "activation", values[0])
                for index, values in enumerate(result.activations)
            ]
            return {
                "model_id": model_id,
                "object_id": canonical,
                "architecture": self.ann01.model.architecture,
                "parameters": parameters,
                "activations": activations,
                "notes": [
                    "Weights and biases come from the persisted Stage 04 ANN-01 checkpoint.",
                    "Activations belong only to the requested astronomical object.",
                ],
            }
        if model_id == "ann-02":
            sample = self.ann02_samples.get(canonical)
            if sample is None:
                raise ApiError(422, "incompatible_object", "The object has no ANN-02 signal representation.")
            inference = infer_ann02(self.ann02, sample)
            parameters = [
                _tensor_state(name, "bias" if name.endswith("bias") else "weight", parameter.detach().cpu().numpy())
                for name, parameter in self.ann02.model.named_parameters()
            ]
            activations = [
                _tensor_state(name, "feature_map", values)
                for name, values in inference["feature_maps"].items()
            ]
            return {
                "model_id": model_id,
                "object_id": canonical,
                "architecture": self.ann02.model.architecture(),
                "parameters": parameters,
                "activations": activations,
                "notes": [
                    "Parameters come from the selected seed_stability checkpoint (seed 97).",
                    "Feature maps are captured from a real forward pass for the requested object.",
                    "The visualized tensors explain computation, not astrophysical causality.",
                ],
            }
        raise ApiError(404, "model_not_found", "Unknown ASTRAL model identifier.")

    def comparison(self) -> dict[str, Any]:
        return public_payload(load_json(COMPARISON_PATH))

    def explanation_index(self) -> dict[str, Any]:
        report = load_json(EXPLAINABILITY_PATH)
        return {
            "ann01_permutation_importance": report["ann01_permutation_importance"],
            "objects": [
                {
                    "object_id": item["object_id"],
                    "kic": item["kic"],
                    "split": item["split"],
                    "catalog_disposition": item["catalog_disposition"],
                }
                for item in report["representative_local_explanations"]
            ],
            "guardrails": report["guardrails"],
        }

    def explanation(self, object_id: str, include_saliency: bool, saliency_points: int) -> dict[str, Any]:
        report = load_json(EXPLAINABILITY_PATH)
        match = next(
            (
                item for item in report["representative_local_explanations"]
                if str(item["object_id"]).upper() == object_id.upper()
            ),
            None,
        )
        if match is None:
            raise ApiError(404, "explanation_not_found", "No persisted explanation exists for this object.")
        result = public_payload(match)
        saliency = result.get("ann02", {}).pop("saliency", None)
        if include_saliency and saliency:
            compact: dict[str, Any] = {}
            for name, values in saliency.items():
                array = np.asarray(values, dtype=float)
                indices = _sample_indices(array.size, saliency_points)
                compact[name] = {
                    "source_points": int(array.size),
                    "returned_points": int(indices.size),
                    "indices": indices.tolist(),
                    "values": _finite_list(array[indices]),
                }
            result["ann02"]["saliency"] = compact
        return result

    def list_experiments(self, stage: Literal["stage06", "stage07"], offset: int, limit: int) -> tuple[list[dict[str, Any]], int]:
        root = PROJECT_ROOT / "experiments" / stage
        files = sorted(
            path for path in root.rglob("*.json")
            if path.name != "latest_experiment.json" and path.name != "latest_campaign.json"
        )
        # A persisted run can be referenced by more than one JSON artifact
        # (for example, a detailed run and a compact campaign export).  The
        # experiment identity is semantic, not the artifact path, so expose it
        # once and prefer the record carrying the strongest provenance.
        rows_by_id: dict[str, dict[str, Any]] = {}
        for path in files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            experiment_id = payload.get("experiment_id") or payload.get("campaign_id")
            if experiment_id:
                identity = str(experiment_id)
                candidate = {
                    "experiment_id": identity,
                    "stage": stage,
                    "kind": "campaign" if "campaign_id" in payload else "training_run",
                    "generated_at_utc": payload.get("generated_at_utc"),
                    "selected": identity == str(self.stage07_campaign.get("selected_experiment_id")),
                }
                current = rows_by_id.get(identity)
                if current is None or (
                    current.get("generated_at_utc") is None
                    and candidate.get("generated_at_utc") is not None
                ):
                    rows_by_id[identity] = candidate
                elif candidate["selected"]:
                    current["selected"] = True
        rows = sorted(
            rows_by_id.values(),
            key=lambda row: str(row.get("generated_at_utc") or ""),
            reverse=True,
        )
        return rows[offset : offset + limit], len(rows)

    def experiment(self, experiment_id: str) -> dict[str, Any]:
        if not _EXPERIMENT_ID.fullmatch(experiment_id):
            raise ApiError(422, "invalid_experiment_id", "The experiment identifier is invalid.")
        for stage in ("stage07", "stage06"):
            root = PROJECT_ROOT / "experiments" / stage
            for path in root.rglob("*.json"):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                identity = payload.get("experiment_id") or payload.get("campaign_id")
                if str(identity) == experiment_id:
                    if "campaign_id" in payload:
                        return {
                            "experiment_id": str(payload["campaign_id"]),
                            "stage": stage,
                            "kind": "campaign",
                            "generated_at_utc": payload.get("generated_at_utc"),
                            "selected_trial": payload.get("selected_trial"),
                            "selected_experiment_id": payload.get("selected_experiment_id"),
                            "selection_metric": payload.get("selection_metric"),
                            "runs": [
                                {
                                    "name": run["name"],
                                    "changed_factor": run["changed_factor"],
                                    "experiment_id": run["experiment_id"],
                                    "best_epoch": run["best_epoch"],
                                    "epochs_completed": run["epochs_completed"],
                                    "duration_seconds": run["duration_seconds"],
                                    "validation_metrics": run["validation_metrics"],
                                }
                                for run in payload.get("runs", [])
                            ],
                            "guardrails": payload.get("guardrails", {}),
                            "limitations": payload.get("limitations", []),
                        }
                    return {
                        "experiment_id": str(payload["experiment_id"]),
                        "stage": stage,
                        "kind": "training_run",
                        "generated_at_utc": payload.get("generated_at_utc"),
                        "architecture": public_payload(payload.get("architecture")),
                        "configuration": public_payload(payload.get("training_config")),
                        "best_epoch": payload.get("best_epoch"),
                        "epochs_completed": payload.get("epochs_completed"),
                        "duration_seconds": payload.get("duration_seconds"),
                        "early_stopped": payload.get("early_stopped"),
                        "history": public_payload(payload.get("history", [])),
                        "validation_metrics": public_payload(payload.get("validation_metrics")),
                        "guardrails": payload.get("scientific_guardrails", payload.get("guardrails", {})),
                        "limitations": payload.get("limitations", []),
                    }
        raise ApiError(404, "experiment_not_found", "No persisted experiment matches the identifier.")

    def tess(self) -> dict[str, Any]:
        return public_payload(load_json(TESS_REPORT_PATH))
