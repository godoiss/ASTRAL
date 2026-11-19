"""Command-line entry point for bounded Stage 11 verification."""

from __future__ import annotations

import json

from src.candidate_intelligence.engine import build_candidate_intelligence


def main() -> None:
    report = build_candidate_intelligence()
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
