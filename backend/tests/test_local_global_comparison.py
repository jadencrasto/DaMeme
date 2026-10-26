from app.benchmark import BenchmarkQuery
from app.local_global_comparison import (
    check_source_key_alignment,
    development_queries,
    fuse_scores,
    rrf_scores,
)


def test_source_key_compatibility_requires_identical_order() -> None:
    check_source_key_alignment(["a", "b"], ["a", "b"])
    try:
        check_source_key_alignment(["a", "b"], ["b", "a"])
    except ValueError as error:
        assert "order" in str(error)
    else:
        raise AssertionError("source-key order mismatch must fail")


def test_missing_local_rows_use_global_score_not_artificial_zero() -> None:
    result = fuse_scores(
        {"available": 0.8, "missing": 0.0}, {"available": 0.2, "missing": 0.6}, {"available"}, 0.5
    )
    assert result == {"available": 0.5, "missing": 0.6}


def test_weighted_score_fusion() -> None:
    result = fuse_scores({"a": 0.8}, {"a": 0.2}, {"a"}, 0.7)
    assert result["a"] == 0.62


def test_rrf_fusion_and_deterministic_tie_order() -> None:
    assert rrf_scores(["b", "a"], ["a", "b"], k=60) == rrf_scores(["b", "a"], ["a", "b"], k=60)
    assert list(rrf_scores(["b"], ["a"], k=60)) == ["a", "b"]


def test_development_only_isolation() -> None:
    development = BenchmarkQuery("dev", "m", None, "c", "n", "development")
    held_out = BenchmarkQuery("test", "m", None, "c", "n", "test")
    assert development_queries([development, held_out]) == [development]
