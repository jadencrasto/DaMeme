# DaMeme engineering guide

## Standards

- Keep modules small, cohesive, typed, and clearly named. Prefer established libraries and minimal dependencies.
- Maintain clean boundaries; avoid premature optimization, speculative abstractions, and fabricated functionality.
- Preserve paper attribution, document deviations, record dataset/model provenance, and keep experiments reproducible.
- Every implemented feature needs proportionate tests. Run relevant checks, fix discovered issues, and report passing, failing, and unexecuted checks accurately.
- Work only in the currently authorized phase. Do not download datasets or model weights without explicit authorization.
- Never commit credentials. Validate external input, avoid unsafe file handling, respect source licenses, and never execute untrusted downloaded content.

## Layout and commands

- `backend/` is a Python 3.11 FastAPI service managed with `uv`.
- `frontend/` is a strict TypeScript React/Vite application managed with npm.
- Use `docs/` for source-backed architectural and research decisions. Keep models, datasets, indexes, artifacts, and generated assets out of Git.
- Before modifying a non-empty checkout, inspect `git status`; do not discard unrelated user work. Do not push without authorization.
