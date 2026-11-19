"""Command-line interface for bounded ASTRAL discovery campaigns."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from src.discovery.campaign import load_campaign, run_campaign
from src.discovery.catalog import select_exploration_targets
from src.discovery.config import DiscoveryConfig
from src.training.logging_utils import configure_structured_logging


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "discovery_engine.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="ASTRAL Stage 10 Discovery Engine")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="Show the deterministic protected target plan")
    plan.add_argument("--sample-size", type=int)
    run = subparsers.add_parser("run", help="Execute a bounded campaign")
    run.add_argument("--sample-size", type=int)
    run.add_argument("--max-new-downloads", type=int)
    run.add_argument("--allow-network", action="store_true")
    run.add_argument("--campaign-id")
    inspect = subparsers.add_parser("inspect", help="Print the latest campaign report")
    inspect.add_argument("--path", type=Path)
    arguments = parser.parse_args()
    configure_structured_logging(arguments.log_level)
    config = DiscoveryConfig.load(arguments.config)
    if arguments.command == "inspect":
        payload = load_campaign(arguments.path)
        if payload is None:
            print(json.dumps({"status": "unavailable", "reason": "no campaign found"}))
            return 1
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    sample_size = arguments.sample_size or config.campaign.sample_size
    campaign = replace(
        config.campaign,
        sample_size=sample_size,
        max_new_downloads=min(config.campaign.max_new_downloads, sample_size),
    )
    config = replace(config, campaign=campaign)
    if arguments.command == "plan":
        payload = {
            "configuration": config.to_dict(),
            "targets": [item.to_dict() for item in select_exploration_targets(campaign)],
        }
    else:
        campaign = replace(
            campaign,
            allow_network=bool(arguments.allow_network),
            max_new_downloads=(
                campaign.max_new_downloads
                if arguments.max_new_downloads is None
                else arguments.max_new_downloads
            ),
        )
        config = replace(config, campaign=campaign)

        def progress(stage: str, index: int, total: int, message: str) -> None:
            print(f"[{index}/{total}] {stage}: {message}")

        payload = run_campaign(
            config,
            campaign_id=arguments.campaign_id,
            progress_callback=progress,
        )
        payload = {
            "campaign_id": payload["campaign_id"],
            "status": payload["status"],
            "campaign_path": str(
                Path("experiments") / "stage10" / "campaigns" / payload["campaign_id"] / "campaign.json"
            ).replace("\\", "/"),
            "summary": payload["summary"],
            "scientific_boundaries": payload["scientific_boundaries"],
        }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
