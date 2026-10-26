from app.situation_reranking_impact import judged_metrics, top_five_impact


def test_top_five_impact_distinguishes_order_from_membership() -> None:
    internal = top_five_impact(["a", "b", "c", "d", "e"], ["b", "a", "c", "d", "e"])
    assert internal["top_five_order_changed"]
    assert not internal["top_five_membership_changed"]
    assert internal["overlap"] == 1.0
    membership = top_five_impact(["a", "b", "c", "d", "e"], ["a", "b", "c", "d", "f"])
    assert membership["top_five_membership_changed"]
    assert membership["entering"] == ["f"]
    assert membership["leaving"] == ["e"]


def test_judged_metrics_refuse_missing_ratings() -> None:
    ratings = {key: {"situation": 2, "tone": 2, "sendability": 2} for key in ("a", "b", "c", "d", "e")}
    assert judged_metrics(["a", "b", "c", "d", "e"], ["a", "b", "c", "d", "e"], ratings)
    assert judged_metrics(["a", "b", "c", "d", "e"], ["a", "b", "c", "d", "f"], ratings) is None
