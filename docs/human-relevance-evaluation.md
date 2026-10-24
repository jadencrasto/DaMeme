# Phase 7A.1 human relevance evaluation

This workflow collects human judgments for the five fixed Phase 7A queries. It
does not use similarity scores, cross-encoder scores, or source emotion and
sentiment annotations as ground truth.

## Create or load candidates

The checked-in `backend/evaluation_phase7a.json` is a usable candidate export.
To create a new one from a local catalog, run the existing evaluator and save
its standard output:

```powershell
Set-Location C:\Users\jason\OneDrive\Desktop\DaMeme\backend
uv run python -m app.catalog_evaluation --catalog memematch_1000_v2 --device cpu --modes semantic,hybrid,hybrid_cross_encoder > evaluation_phase7a.json
```

The evaluator uses the five fixed queries and includes ten results per mode.
The human-review workflow only requires the top five from each mode.

## Rate candidates

```powershell
Set-Location C:\Users\jason\OneDrive\Desktop\DaMeme\backend
uv run python -m app.human_evaluation serve --evaluation evaluation_phase7a.json --ratings evaluation_phase7a.ratings.json --catalog-root ..\data\retrieval\catalogs\memematch_1000_v2
```

Open `http://127.0.0.1:8090`. The page shows the query, filename, and image,
but deliberately hides retrieval mode, rank, and every model score. It combines
identical query--meme pairs across modes into one task. Ratings are immediately
saved to the supplied local JSON file, so closing and reopening the interface
resumes work; selecting another score edits that rating.

Use the required scale:

- 0 — Irrelevant
- 1 — Slightly relevant
- 2 — Relevant
- 3 — Highly relevant

## Produce a comparison report

```powershell
uv run python -m app.human_evaluation report --evaluation evaluation_phase7a.json --ratings evaluation_phase7a.ratings.json --output evaluation_phase7a.metrics.json
```

The command intentionally returns `complete: false` and lists missing tasks
until every unique top-five query--meme pair is rated. It does not calculate
partial final metrics.

When complete, the report gives each mode's Precision@5 and nDCG@5 per query,
plus their means across the five queries. Precision@5 treats ratings 2 and 3 as
relevant. nDCG@5 uses the full graded 0--3 rating scale and discounts lower
ranks. These values describe only the collected judgments; they do not prove a
mode is generally better beyond this query set.

## Sentiment diagnostic

`sentiment_relevance: null` in `evaluation_phase7a.json` is expected. The text
API accepts an optional `sentiment` parameter, defaults it to `None`, and
forwards it unchanged. The evaluation runner also explicitly passes
`sentiment=None`. `RecommendationService` therefore does not request a
positive, negative, or neutral metadata score and returns `None` for the field.

The catalog does contain source sentiment annotation columns. They are used
only when a caller explicitly supplies `sentiment=positive`, `negative`, or
`neutral`; the system does not infer one from query text or the selected emotion.
