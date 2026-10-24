# Phase 7B expanded moods

The six trained classifier labels remain `anger`, `fear`, `joy`, `sadness`,
`surprise`, and `disgust`. The 32 entries returned by `GET /api/v1/emotions`
are user-selectable retrieval context, not new classifier outputs.

The four taxonomy groups are Positive and uplifting, Sad and reflective, Stress
and negative, and Anticipation and neutral. Each API entry supplies a stable
ID, label, description, synonyms, heuristic classifier-weight mapping, and an
optional sentiment association. The mappings are hand-authored retrieval
heuristics, not learned labels or ground truth. In particular, anxiety, stress,
fear, embarrassment, and frustration retain distinct descriptions and query
context even where their optional six-label ranking mapping overlaps.

For a text request, `mood=anxiety` constructs:

```text
Situation: <the exact user text>
Selected mood context: uneasy concern about an uncertain outcome.
```

The original situation stays first; the short context is supplemental. If a
caller omits `mood`, query construction and ranking remain unchanged. A legacy
`emotion` request still has priority for six-label reranking.

## API and UI

```powershell
Invoke-RestMethod "http://localhost:8000/api/v1/emotions"
Invoke-RestMethod "http://localhost:8000/api/v1/recommendations/text?text=My%20presentation%20starts%20soon&mood=anxiety"
```

The React selector fetches this endpoint, organizes entries by group, supports
search and clearing, and sends the selected stable ID only for text requests.
Test it with the backend and frontend running locally, then select two moods
for the same situation and compare results. Different ranks demonstrate an
implemented influence, not improved relevance; use the Phase 7A.1 human-review
workflow for relevance judgments.

## Mood discrimination diagnostic

```powershell
Set-Location C:\Users\jason\OneDrive\Desktop\DaMeme\backend
uv run python -m app.mood_evaluation --catalog memematch_1000_v2 --mode semantic
```

It records the effective retrieval query, top-five IDs, candidate counts,
latency, warnings, and whether each contrasting pair changes ranking. It does
not calculate quality metrics.

Limitations: catalog annotations remain six-label source probabilities; no
32-label classifier was trained. Query expansion can only influence retrieval
through available text representations, and should be human-evaluated before
any quality claim. Future work could collect labeled mood judgments and test
calibrated, learned mappings.
