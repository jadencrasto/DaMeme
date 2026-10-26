import pytest

from app.query_intent_audit import development_queries


def test_development_query_filter_excludes_test_rows() -> None:
    payload = {
        "queries": [{"id": f"d{index}", "split": "development"} for index in range(40)]
        + [{"id": "test", "split": "test"}]
    }
    assert len(development_queries(payload)) == 40


def test_development_query_filter_rejects_incomplete_development_set() -> None:
    with pytest.raises(ValueError, match="40 development"):
        development_queries({"queries": [{"id": "d", "split": "development"}]})
