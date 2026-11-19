# ASTRAL Stage 09 — Discovery-first design package

Status: **high-fidelity visual evaluation — direction not yet approved**

This directory is a design-only artifact. It does not contain Next.js, a
production client, scientific computation, API calls, model changes, or dataset
changes.

## Contents

- `design_tokens.json`: frontend-ready foundations for Atlas, Signal Room, and
  Quiet Orbit; Atlas is the recommended default.
- `preview/`: buildless HTML/CSS/JavaScript prototype for the five intended
  product areas: Discover, Observatory, Candidates, Neural Lab, and Research.
- `preview/assets/astral-mark.svg`: original compact orbit/star identity mark.
- `preview/atlas-*.png` and `preview/signal-room.png`: browser-rendered desktop,
  Neural Lab, scientific-area, theme, and 390 CSS px review captures.
- `docs/stage09-branding-design-system.md`: creative comparison, design system,
  UX architecture, scientific guardrails, component plan, and review gates.

The three directions share one layout and interaction model. The switch in the
preview is a live color-direction study, not three separate product concepts.

## Local preview

```powershell
cd design/stage09/preview
python -m http.server 3000 --bind 127.0.0.1
```

Open `http://127.0.0.1:3000`. The prototype intentionally has no local or remote
API dependency. Its three object scores and research metrics are labeled
snapshots copied from the audited Stage 08 artifacts; missing discovery and
light-curve services are shown as unavailable.

## Review boundary

Rendering successfully does not close Stage 09. Approval still requires visual
review of the identity, Neural Lab, typography/licensing, responsive and
keyboard behavior, and future campaign states.
