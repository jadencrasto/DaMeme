import json

import pytest

from app.human_evaluation import _image_path, build_dataset, load_ratings, metrics, save_ratings


def _export() -> dict[str, object]:
    return {
        "queries": [
            {
                "query": "passed exam",
                "modes": {
                    "semantic": {"results": [{"meme_id": name} for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg")]},
                    "hybrid": {"results": [{"meme_id": name} for name in ("a.jpg", "f.jpg", "c.jpg", "g.jpg", "h.jpg")]},
                    "hybrid_cross_encoder": {"results": [{"meme_id": name} for name in ("f.jpg", "a.jpg", "i.jpg", "b.jpg", "j.jpg")]},
                },
            }
        ]
    }


def test_dataset_deduplicates_query_meme_pairs_and_metrics_require_all_ratings() -> None:
    dataset = build_dataset(_export())
    assert len(dataset.items) == 10
    incomplete = metrics(dataset, {})
    assert incomplete["complete"] is False
    ratings = {key: 3 if item.meme_id in {"a.jpg", "f.jpg"} else 0 for key, item in dataset.items.items()}
    result = metrics(dataset, ratings)
    assert result["complete"] is True
    semantic = result["modes"]["semantic"]
    assert semantic["mean_precision_at_5"] == pytest.approx(0.2)
    assert semantic["mean_ndcg_at_5"] == pytest.approx(1.0)


def test_ndcg_uses_graded_ratings_and_rank_discount() -> None:
    exported = _export()
    exported["queries"][0]["modes"]["semantic"]["results"] = [
        {"meme_id": name} for name in ("b.jpg", "a.jpg", "c.jpg", "d.jpg", "e.jpg")
    ]
    dataset = build_dataset(exported)
    ratings = {key: 3 if item.meme_id == "a.jpg" else 0 for key, item in dataset.items.items()}
    result = metrics(dataset, ratings)
    assert result["modes"]["semantic"]["mean_precision_at_5"] == pytest.approx(0.2)
    assert result["modes"]["semantic"]["mean_ndcg_at_5"] == pytest.approx(1 / 1.5849625)


def test_ratings_persist_resume_and_reject_a_different_export(tmp_path) -> None:
    dataset = build_dataset(_export())
    ratings_path = tmp_path / "ratings.json"
    first_key = next(iter(dataset.items))
    save_ratings(ratings_path, dataset, {first_key: 2})
    assert load_ratings(ratings_path, dataset) == {first_key: 2}
    payload = json.loads(ratings_path.read_text(encoding="utf-8"))
    assert payload["ratings"][first_key] == 2
    other = _export()
    other["queries"][0]["query"] = "different query"
    with pytest.raises(ValueError, match="different evaluation"):
        load_ratings(ratings_path, build_dataset(other))


def test_invalid_rating_and_short_mode_results_are_rejected(tmp_path) -> None:
    dataset = build_dataset(_export())
    with pytest.raises(ValueError, match="0--3"):
        save_ratings(tmp_path / "ratings.json", dataset, {next(iter(dataset.items)): 4})
    malformed = _export()
    malformed["queries"][0]["modes"]["semantic"]["results"] = []
    with pytest.raises(ValueError, match="fewer than 5"):
        build_dataset(malformed)


def test_image_resolution_stays_under_the_configured_catalog_root(tmp_path) -> None:
    image = tmp_path / "images" / "safe.jpg"
    image.parent.mkdir()
    image.write_bytes(b"image")
    assert _image_path(tmp_path, "safe.jpg") == image
    assert _image_path(tmp_path, "..\\outside.jpg") is None
