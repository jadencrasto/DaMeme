import sqlite3
from pathlib import Path

import pytest

from app.product_feedback import DuplicateProductFeedback, ProductFeedback, record_product_feedback


def test_product_feedback_is_deduplicated_and_isolated(tmp_path: Path) -> None:
    database = tmp_path / "product.db"
    feedback = ProductFeedback("request-1", "meme.jpg", "anxiety", "a0", "useful")
    record_product_feedback(database, feedback)
    with pytest.raises(DuplicateProductFeedback):
        record_product_feedback(database, feedback)
    connection = sqlite3.connect(database)
    try:
        row = connection.execute(
            "SELECT request_id, meme_id, selected_mood, retrieval_variant, feedback_value FROM product_feedback"
        ).fetchone()
        assert row == ("request-1", "meme.jpg", "anxiety", "a0", "useful")
        assert connection.execute("SELECT COUNT(*) FROM sqlite_master WHERE name = 'product_feedback'").fetchone()[0] == 1
    finally:
        connection.close()
