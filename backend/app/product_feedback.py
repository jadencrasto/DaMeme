"""Persistence for end-user feedback, intentionally separate from research ratings."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.datasets.database import connect, migrate


@dataclass(frozen=True)
class ProductFeedback:
    request_id: str
    meme_id: str
    mood: str | None
    retrieval_variant: str
    value: str


class DuplicateProductFeedback(ValueError):
    pass


def record_product_feedback(database: Path, feedback: ProductFeedback) -> None:
    connection = connect(database)
    try:
        migrate(connection)
        connection.execute(
            "INSERT INTO product_feedback(request_id, meme_id, selected_mood, retrieval_variant, feedback_value, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (feedback.request_id, feedback.meme_id, feedback.mood, feedback.retrieval_variant, feedback.value, datetime.now(UTC).isoformat()),
        )
        connection.commit()
    except sqlite3.IntegrityError as exc:
        connection.rollback()
        raise DuplicateProductFeedback("Feedback was already recorded for this recommendation.") from exc
    finally:
        connection.close()


def feedback_analytics(database: Path, *, limit: int, offset: int) -> dict[str, object]:
    connection = connect(database)
    try:
        migrate(connection)
        counts = dict(connection.execute("SELECT feedback_value, COUNT(*) AS count FROM product_feedback GROUP BY feedback_value"))
        total = sum(int(value) for value in counts.values())
        moods = [dict(row) for row in connection.execute("SELECT selected_mood AS mood, COUNT(*) AS count FROM product_feedback GROUP BY selected_mood ORDER BY count DESC, mood ASC")]
        recent = [dict(row) for row in connection.execute("SELECT request_id, meme_id, selected_mood, retrieval_variant, feedback_value, created_at FROM product_feedback ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset))]
        return {"total": total, "useful": int(counts.get("useful", 0)), "not_relevant": int(counts.get("not_relevant", 0)), "useful_percentage": (int(counts.get("useful", 0)) / total * 100) if total else None, "moods": moods, "recent": recent, "limit": limit, "offset": offset}
    finally:
        connection.close()
