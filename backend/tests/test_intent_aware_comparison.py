from app.benchmark import BenchmarkQuery
from app.intent_aware_comparison import (
    CANDIDATE_DEPTH,
    _fresh_preregistered_queries,
    retrieval_configuration,
)


def test_retrieval_configuration_parity_is_fixed() -> None:
    configuration = retrieval_configuration()
    assert configuration["candidate_depth"] == CANDIDATE_DEPTH == 100
    assert configuration["safety"] == "existing SafetyConfig"
    assert configuration["diversity"] == "existing one-per-template DiversityConfig"


def test_fresh_evaluation_selection_excludes_held_out_and_previously_rated_queries() -> None:
    queries = {
        "development-a": BenchmarkQuery("development-a", "m", None, "category-a", "n", "development"),
        "development-b": BenchmarkQuery("development-b", "m", None, "category-b", "n", "development"),
        "test": BenchmarkQuery("test", "held out", None, "category-a", "n", "test"),
    }
    assert _fresh_preregistered_queries(queries, {"development-a"}) == ["development-b"]
