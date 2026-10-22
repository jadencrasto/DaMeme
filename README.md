# DaMeme

DaMeme is a research-first, emotion-aware meme recommendation project. **Phase 2** adds a local, resumable dataset-management pipeline; it does not download research datasets, analyze meme content, or return recommendations.

## Repository map

- `backend/` — versioned FastAPI health endpoint and tests.
- `frontend/` — Vite React landing screen, Tailwind v4 setup, API health state.
- `docs/` — research analysis, architecture, data strategy, roadmap, and ADRs.
- `ml/`, `data/` — documented future boundaries, not datasets or models.

## Prerequisites (Windows 11 PowerShell)

- Python 3.11 (`py -3.11 --version`)
- [uv](https://docs.astral.sh/uv/) (`uv --version`)
- Node.js LTS and npm (`node --version`, `npm --version`)

## Dataset management (Phase 2)

The local-only ingestion CLI validates image files, keeps provenance, detects exact/near duplicates, checkpoints batches, exports manifests, and preserves supplied source splits/labels. It does not download anything. See [dataset sources](docs/dataset-sources.md), [schema](docs/dataset-schema.md), and [ingestion guide](docs/ingestion-pipeline.md).

```powershell
Set-Location backend
uv run python -m app.datasets.cli --help
uv run python -m app.datasets.cli --database ..\data\metadata.db init-db
```

## Configure

```powershell
Copy-Item .env.example .env
```

Values use the `DAMEME_` prefix. The backend reads `.env` from its working directory; either copy it into `backend/.env`, or set variables in the shell:

```powershell
$env:DAMEME_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
```

## Run the backend

```powershell
Set-Location backend
uv sync --group dev
uv run uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` or `http://localhost:8000/api/v1/health`.

## Run the frontend

In another PowerShell window:

```powershell
Set-Location frontend
npm install
npm run dev
```

Visit the Vite URL (normally `http://localhost:5173`). It checks `http://localhost:8000/api/v1/health`; override the API base URL with `VITE_API_BASE_URL` when required.

## Checks

```powershell
Set-Location backend
uv run ruff check .
uv run pytest

Set-Location ..\frontend
npm run lint
npm run build
```

## Troubleshooting

- **CORS/API offline:** start the backend on port 8000 and ensure its allowed origins include the Vite URL.
- **Python version mismatch:** install Python 3.11 and invoke `uv` from `backend/`.
- **Node install failure:** use an active Node LTS release, delete only `frontend/node_modules` if necessary, then run `npm install` again.
- **No recommendations appear:** expected in Phase 1; the search control is intentionally disabled.

See [architecture](docs/architecture.md), [research analysis](docs/research-analysis.md), and the [roadmap](docs/development-roadmap.md).
