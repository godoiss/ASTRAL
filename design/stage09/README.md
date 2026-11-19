# ASTRAL Stage 09 — Discovery-first design package

Status: **Atlas approved; buildless client retained as a historical design artifact**

This directory contains the Stage 09 design artifact and the superseded Stage 10
buildless client. The publication interface now lives in `web/` as a Next.js
application. The historical files remain available for design provenance but are
not served as the product interface.

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

## Publication client

```powershell
cd web
npm install
npm run dev
```

Open `http://127.0.0.1:3000`. The Next.js client proxies the versioned FastAPI
boundary and contains no independent scientific computation.

## Review boundary

The interface does not convert a detector screen or neural score into scientific
confirmation. Frozen-test records remain inaccessible, and unmatched signals
remain explicitly inconclusive.
