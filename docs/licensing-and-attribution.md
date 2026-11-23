# Licensing and attribution

## ASTRAL code

Original ASTRAL source code and documentation are currently distributed under the repository's **all-rights-reserved** `LICENSE`. Public visibility of this repository does not itself grant reuse rights. The owner may replace this with an approved open-source license in a future explicit legal decision.

## Scientific data and mission services

ASTRAL does not claim ownership of astronomical observations or catalog records. Scientific products retain their original provenance and identifiers.

- NASA Exoplanet Archive: <https://exoplanetarchive.ipac.caltech.edu/>
- Mikulski Archive for Space Telescopes (MAST): <https://archive.stsci.edu/>
- Kepler mission archive: <https://archive.stsci.edu/missions-and-data/kepler>
- TESS mission archive: <https://archive.stsci.edu/missions-and-data/tess>

Users publishing scientific work must follow the archive and mission citation guidance attached to the specific products they use. ASTRAL's reports are screening and engineering artifacts; they do not replace mission data citations.

## Third-party software and visual assets

Dependencies remain under their own licenses. Their inclusion does not change the ASTRAL source license. Important direct components include:

- Next.js and React;
- FastAPI, Pydantic and Uvicorn;
- NumPy, Pandas, Matplotlib, Astropy, Lightkurve, scikit-learn and PyTorch;
- Plotly.js;
- Lucide icons;
- Manrope Variable and Newsreader Variable fonts.

The exact dependency graph is reproducible from `requirements.txt` and `web/package-lock.json`. Redistributors must retain the notices required by each dependency. The font files bundled through `@fontsource` retain their upstream font licenses; Manrope and Newsreader are distributed under the SIL Open Font License 1.1. Lucide icons retain the upstream ISC license. Plotly.js retains its upstream MIT license.

No stock astronomical image, proprietary logo, or generative image asset is bundled as a scientific observation. The Atlas mark, layouts, gradients and code-native graphics are original repository assets.

## Trademarks and scientific interpretation

NASA, Kepler, TESS, MAST and third-party project names identify data sources and do not imply endorsement. ASTRAL model scores, screening detections and illustrations must not be represented as official mission confirmations or astronomical discoveries.

