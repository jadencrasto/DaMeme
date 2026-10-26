from pathlib import Path

import pytest

from app.product_feedback import ProductFeedback, feedback_analytics, record_product_feedback


def test_feedback_analytics_handles_empty_and_paginates(tmp_path: Path) -> None:
    database = tmp_path / "feedback.db"
    empty = feedback_analytics(database, limit=2, offset=0)
    assert empty["total"] == 0 and empty["useful_percentage"] is None and empty["recent"] == []
    for index, value in enumerate(("useful", "not_relevant", "useful")):
        record_product_feedback(database, ProductFeedback(f"request-{index}", f"meme-{index}.jpg", "joy", "a0", value))
    page = feedback_analytics(database, limit=2, offset=1)
    assert page["total"] == 3 and page["useful"] == 2 and page["not_relevant"] == 1
    assert page["useful_percentage"] == pytest.approx(200 / 3) and len(page["recent"]) == 2
