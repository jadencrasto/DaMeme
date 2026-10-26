from app.benchmark import BenchmarkQuery
from app.session_repetition_ablation import build_sessions, rerank_with_session_penalty


class _Item:
    def __init__(self, meme_id: str, score: float):
        self.meme_id = meme_id
        self.final_score = score


def test_soft_penalty_demotes_but_does_not_exclude_recent_meme() -> None:
    result = rerank_with_session_penalty([_Item("a", 0.9), _Item("b", 0.88)], {"a"}, penalty=0.05)
    assert [item.meme_id for item in result] == ["b", "a"]
    assert {item.meme_id for item in result} == {"a", "b"}


def test_session_simulator_contains_related_unrelated_and_repeated_sequences() -> None:
    queries = {
        f"q{category}{index}": BenchmarkQuery(
            f"q{category}{index}", "m", None, category, "n", "development"
        )
        for category in ("a", "b")
        for index in range(2)
    }
    sessions = build_sessions(queries)
    assert {session["type"] for session in sessions} == {"related", "unrelated", "repeated"}
    assert {query_id for session in sessions for query_id in session["query_ids"]} == set(queries)
