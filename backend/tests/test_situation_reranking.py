from app.situation_reranking import (
    MAX_PROMOTION,
    candidate_evidence_text,
    rerank_candidates,
    situation_match,
)


def _rows() -> list[dict[str, object]]:
    return [{"source_key": key} for key in ("a", "b", "c", "d")]


def test_reranking_is_deterministic_and_bounded() -> None:
    text = {"a": "generic", "b": "interview meeting", "c": "interview", "d": "generic"}
    first = rerank_candidates(_rows(), query="job interview today", candidate_text=text)
    second = rerank_candidates(_rows(), query="job interview today", candidate_text=text)
    assert first == second
    assert {row["source_key"] for row in first} == {"a", "b", "c", "d"}
    assert max(abs(int(row["rank_displacement"])) for row in first) <= MAX_PROMOTION


def test_weak_or_ambiguous_evidence_falls_back_to_baseline_order() -> None:
    text = {key: "unrelated words" for key in ("a", "b", "c", "d")}
    result = rerank_candidates(_rows(), query="maybe something happened", candidate_text=text)
    assert [row["source_key"] for row in result] == ["a", "b", "c", "d"]
    assert all(row["rank_displacement"] == 0 for row in result)


def test_negated_concept_is_not_a_positive_signal() -> None:
    match = situation_match("not an interview reaction", "interview reaction meeting")
    assert not match.is_strong
    assert "interview" in match.negated_terms


def test_source_keys_and_baseline_rows_are_preserved() -> None:
    baseline = _rows()
    result = rerank_candidates(baseline, query="project deadline", candidate_text={"d": "project deadline"})
    assert {row["source_key"] for row in result} == {row["source_key"] for row in baseline}
    assert [row["source_key"] for row in baseline] == ["a", "b", "c", "d"]


def test_post_filter_reranking_preserves_safety_and_template_membership() -> None:
    # These rows stand for an already safety-filtered, one-per-template baseline list.
    baseline = [{"source_key": "safe-template-a"}, {"source_key": "safe-template-b"}]
    result = rerank_candidates(
        baseline,
        query="interview deadline",
        candidate_text={"safe-template-a": "interview", "safe-template-b": "deadline"},
    )
    assert {row["source_key"] for row in result} == {"safe-template-a", "safe-template-b"}


def test_template_format_metadata_is_not_situation_evidence() -> None:
    evidence = candidate_evidence_text(
        {
            "local_Extracted Text + Title": "a person waiting",
            "usage": "Popular Meme Format",
            "template_group_heuristic": "popular_meme_format",
        }
    )
    assert "format" not in evidence


def test_generic_tokens_are_excluded_from_situation_matching() -> None:
    match = situation_match("The group will need a format", "group will needed format")
    assert not match.is_strong
