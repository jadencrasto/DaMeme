# Phase 8 automated-check summary

Executed on 2026-10-27 against commit `727125682b5d01b8c4bf13e03183e35f930ca9e3`.

| Check | Result | Duration / notes |
|---|---|---|
| `backend: uv run pytest -q` | 208 passed | 17.36 s; 2 scikit-learn L-BFGS-B deprecation warnings |
| `backend: uv run ruff check .` | failed | One pre-existing unsorted-import finding in `scripts/prepare_multimodal_ui_adapter.py:7` (`I001`); no change made |
| `frontend: npm run lint` | passed | No reported lint findings |
| `frontend: npm test` | passed with skips | 20 passed, 5 skipped, 0 failed; 7.28 s |
| `frontend: npm run build` | passed | TypeScript build and Vite production build completed in 1.04 s |

The five skipped frontend tests are in `src/App.test.tsx` and predate this evaluation. No application files were changed to address any finding.

## Live API validation — 2026-10-28

The user-started FastAPI service at `http://127.0.0.1:8000` completed **86/86 API checks**: 86 passed, 0 failed, and 0 skipped. This is integration evidence only; it does not assess classification accuracy or retrieval relevance.

| Check | Result | Notes |
|---|---:|---|
| Phase 5 readiness | passed | HTTP 200; ready with 5,469 bundle/image references |
| Text recommendation cases | 6/6 passed | One representative sentence per emotion; each HTTP 200 response contained five `DM-xxxxx` IDs |
| Direct-emotion cases | 6/6 passed | All supported emotion labels returned five `DM-xxxxx` IDs via the Phase 5 endpoint |
| Image responses | 60/60 passed | Every returned `image_url` delivered HTTP 200 and non-empty JPEG or PNG data |
| Repeated ranking order | 12/12 passed | Identical ordered IDs on a repeated request for every text and direct-emotion case |
| Invalid direct emotion | passed | HTTP 422 with the supported-emotion pattern validation error |
