from app.openclip_four_way import rrf


def test_rrf_deduplicates_and_uses_one_based_ranks() -> None:
    result = rrf([{"source_key":"a","raw_rank":1},{"source_key":"a","raw_rank":2},{"source_key":"b","raw_rank":2}], [{"source_key":"b","raw_rank":1},{"source_key":"c","raw_rank":2}])
    assert [row["source_key"] for row in result] == ["b", "a", "c"]
    assert result[0]["rrf_score"] == 1 / 62 + 1 / 61
