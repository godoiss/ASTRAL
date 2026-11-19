"""Static integrity checks for the Stage 09 discovery-first design package."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESIGN_ROOT = PROJECT_ROOT / "design" / "stage09"
PREVIEW_ROOT = DESIGN_ROOT / "preview"


class PreviewParser(HTMLParser):
    """Collect structural and accessibility signals without browser I/O."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.duplicate_ids: set[str] = set()
        self.views = 0
        self.h1_count = 0
        self.primary_navigation_items = 0
        self._primary_nav_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        identifier = values.get("id")
        if identifier:
            if identifier in self.ids:
                self.duplicate_ids.add(identifier)
            self.ids.add(identifier)
        classes = set((values.get("class") or "").split())
        if tag == "nav" and "primary-nav" in classes:
            self._primary_nav_depth = 1
        elif self._primary_nav_depth:
            self._primary_nav_depth += 1
        if self._primary_nav_depth and tag == "button" and values.get("data-view"):
            self.primary_navigation_items += 1
        if "view" in classes:
            self.views += 1
        if tag == "h1":
            self.h1_count += 1

    def handle_endtag(self, tag: str) -> None:
        if self._primary_nav_depth:
            self._primary_nav_depth -= 1


def _relative_luminance(hex_color: str) -> float:
    channels = [int(hex_color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(foreground: str, background: str) -> float:
    high, low = sorted(
        (_relative_luminance(foreground), _relative_luminance(background)), reverse=True
    )
    return (high + 0.05) / (low + 0.05)


def test_preview_has_five_discovery_first_views_and_unique_ids() -> None:
    parser = PreviewParser()
    parser.feed((PREVIEW_ROOT / "index.html").read_text(encoding="utf-8"))

    assert parser.duplicate_ids == set()
    assert parser.views == 5
    assert parser.h1_count == 5
    assert parser.primary_navigation_items == 5
    assert {
        "view-discover",
        "view-observatory",
        "view-candidates",
        "view-neural",
        "view-research",
        "expedition-dialog",
    }.issubset(parser.ids)


def test_all_visual_directions_meet_core_text_contrast() -> None:
    tokens = json.loads((DESIGN_ROOT / "design_tokens.json").read_text(encoding="utf-8"))

    assert tokens["meta"]["status"] == "awaiting-visual-approval"
    assert tokens["meta"]["recommendedDirection"] == "atlas"
    assert tokens["spacingPx"] == [4, 8, 12, 16, 24, 32, 48, 64]
    assert tokens["radiusPx"] == [6, 8, 10, 14, 20]
    assert tokens["typography"]["sizePx"]["body"] >= 16
    for direction in tokens["direction"].values():
        assert _contrast(direction["text"], direction["canvas"]) >= 4.5
        assert _contrast(direction["textSoft"], direction["panel"]) >= 4.5
        assert _contrast(direction["accent"], direction["canvas"]) >= 4.5


def test_preview_is_non_executing_and_uses_only_audited_snapshots() -> None:
    script = (PREVIEW_ROOT / "app.js").read_text(encoding="utf-8")
    html = (PREVIEW_ROOT / "index.html").read_text(encoding="utf-8")
    combined = f"{script}\n{html}".lower()

    assert "fetch(" not in script
    assert "xmlhttprequest" not in script.lower()
    assert "websocket" not in script.lower()
    assert "0.4147668955" in script
    assert "0.7588245273" in script
    assert "0.8280777101" in script
    assert "0.6251904964" in script
    assert "no newly found candidates have been demonstrated" in combined
    assert "discovery engine is not implemented" in combined
    assert "no execution" in combined
    assert "showmodal" in script.lower()


def test_scientific_boundaries_and_schematic_disclaimer_are_visible() -> None:
    html = (PREVIEW_ROOT / "index.html").read_text(encoding="utf-8").lower()

    assert "not an observation or detected signal" in html
    assert "not confirmation probabilities" in html
    assert "not an exact topology recovered from a checkpoint" in html
    assert "no live activation, gradient, or weight" in html
    assert "frozen test data remains inaccessible" in html
    assert "draws no fictional series" in html


def test_stage09_does_not_introduce_a_nextjs_application() -> None:
    assert not (DESIGN_ROOT / "package.json").exists()
    assert not (DESIGN_ROOT / "next.config.js").exists()
    assert not (DESIGN_ROOT / "next.config.mjs").exists()
