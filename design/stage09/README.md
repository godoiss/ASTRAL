# ASTRAL Stage 09 — Discovery-first design package

Status: **Atlas approved and connected to Stage 10 scientific services**

This directory began as the Stage 09 design artifact. In Stage 10, the approved
Atlas client was connected to the versioned FastAPI scientific services and is
served at `/app/`. It still contains no scientific computation: observations,
model scores, campaign evidence, and metrics come from typed API responses.
No Next.js dependency or dataset mutation is introduced by the client.

## Contents

- `design_tokens.json`: frontend-ready foundations for Atlas, Signal Room, and
  Quiet Orbit; Atlas is the recommended default.
- `preview/`: buildless HTML/CSS/JavaScript client for the five product
  product areas: Discover, Observatory, Candidates, Neural Lab, and Research.
- `preview/assets/astral-mark.svg`: original compact orbit/star identity mark.
- `preview/atlas-*.png` and `preview/signal-room.png`: browser-rendered desktop,
  Neural Lab, scientific-area, theme, and 390 CSS px review captures.
- `docs/stage09-branding-design-system.md`: creative comparison, design system,
  UX architecture, scientific guardrails, component plan, and review gates.

Atlas is the approved direction. Historical comparison captures remain in this
directory, but the running interface no longer exposes design-direction controls.

## Local preview

```powershell
$env:ASTRAL_DISCOVERY_EXECUTION_ENABLED = "true"
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/app/`. The client requests real persisted values
from the same-origin API. Campaign execution is cache-only and explicitly gated;
public deployments should leave it disabled until authentication is implemented.

## Review boundary

The interface does not convert a detector screen or neural score into scientific
confirmation. Frozen-test records remain inaccessible, and unmatched signals
remain explicitly inconclusive.
