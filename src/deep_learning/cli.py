"""Command-line workflows for bounded ANN-02 data, training, and inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.deep_learning.config import load_stage06_config
from src.deep_learning.dataset import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MANIFEST_PATH,
    KeplerSignalDataset,
    build_incremental_dataset,
)
from src.deep_learning.hardware import hardware_report
from src.deep_learning.training import (
    infer_ann02,
    load_ann02,
    train_ann02,
    validate_dataset_integrity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "ann02.json"


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASTRAL ANN-02 local signal intelligence")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("hardware", help="Inspect CPU/CUDA/PyTorch compatibility")
    commands.add_parser("status", help="Validate the current local ANN-02 dataset")
    commands.add_parser("build", help="Run the explicit bounded and resumable acquisition")
    commands.add_parser("train", help="Train ANN-02 on train/validation only")
    infer = commands.add_parser("infer", help="Infer one object already in the local dataset")
    infer.add_argument("object_id")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_config, model_config, training_config = load_stage06_config(args.config)
    if args.command == "hardware":
        _print(hardware_report())
    elif args.command == "status":
        payload: dict[str, Any] = {"dataset_available": DEFAULT_DATASET_PATH.is_file()}
        if DEFAULT_DATASET_PATH.is_file():
            payload["integrity"] = validate_dataset_integrity()
        if DEFAULT_MANIFEST_PATH.is_file():
            payload["manifest"] = json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
        _print(payload)
    elif args.command == "build":
        def progress(stage: str, index: int, total: int, message: str) -> None:
            print(f"[{stage}] {index}/{total} {message}", flush=True)

        _print(build_incremental_dataset(dataset_config, progress_callback=progress))
    elif args.command == "train":
        bundle = train_ann02(model_config, training_config)
        _print(bundle.experiment)
    elif args.command == "infer":
        bundle = load_ann02()
        if bundle is None:
            raise SystemExit("ANN-02 model is unavailable; run the train command first")
        dataset = KeplerSignalDataset(split=None)
        matches = [dataset[index] for index in range(len(dataset)) if dataset[index]["object_id"] == args.object_id]
        if not matches:
            raise SystemExit(f"Object {args.object_id!r} is not in the local ANN-02 dataset")
        result = infer_ann02(bundle, matches[0])
        result.pop("feature_maps")
        _print(result)


if __name__ == "__main__":
    main()
