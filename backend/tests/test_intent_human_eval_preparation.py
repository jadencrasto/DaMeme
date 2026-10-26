from app.intent_human_eval_preparation import _variant_top_five


def test_variant_top_five_uses_stable_filtered_ranks() -> None:
    record = {
        "variants": {
            "A3-LH": {
                "final_top_10": [
                    {"source_key": "a.jpg", "filtered_rank": 5},
                    {"source_key": "b.jpg", "filtered_rank": 6},
                ]
            }
        }
    }
    assert _variant_top_five(record, "A3-LH") == {"a.jpg"}
