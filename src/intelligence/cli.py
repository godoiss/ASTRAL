"""Local command line for Stage 07 scientific-intelligence workflows."""

from __future__ import annotations

import argparse
import json

from src.deep_learning.training import load_ann02
from src.intelligence.comparison import compare_ann01_ann02
from src.intelligence.dataset import build_dataset_intelligence
from src.intelligence.experiments import run_ann02_campaign
from src.intelligence.explainability import build_explainability_report
from src.intelligence.tess import run_tess_pilot, tess_compatibility_report
from src.lab.services import load_dataset_frame, load_scientific_model


def main() -> None:
    parser = argparse.ArgumentParser(description="ASTRAL Stage 07 Astronomical Intelligence")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("dataset-report")
    commands.add_parser("compare")
    commands.add_parser("campaign")
    commands.add_parser("explain")
    commands.add_parser("tess-compatibility")
    tess = commands.add_parser("tess-pilot")
    tess.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    if args.command == "dataset-report":
        payload = build_dataset_intelligence()
    elif args.command == "compare":
        ann01 = load_scientific_model()
        ann02 = load_ann02(device_preference="cpu")
        if ann01 is None or ann02 is None:
            raise SystemExit("Both persisted models are required")
        payload = compare_ann01_ann02(ann01, ann02, load_dataset_frame())
    elif args.command == "campaign":
        payload = run_ann02_campaign()
    elif args.command == "explain":
        ann01 = load_scientific_model()
        ann02 = load_ann02(device_preference="cpu")
        if ann01 is None or ann02 is None:
            raise SystemExit("Both persisted models are required")
        payload = build_explainability_report(ann01, ann02, load_dataset_frame())
    elif args.command == "tess-compatibility":
        payload = tess_compatibility_report()
    else:
        payload = run_tess_pilot(download=not args.no_download)
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
