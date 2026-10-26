from app.query_intent import interpret_query


def test_explicit_celebration_adds_one_short_label_and_preserves_original() -> None:
    result = interpret_query("I passed my driving test on the first try.", "Situation: I passed my driving test on the first try.", "pride")
    assert result.original_query == "I passed my driving test on the first try."
    assert result.intent_labels == ("celebration",)
    assert result.confidence == "high"
    assert result.retrieval_representation.endswith("Intent: celebration.")


def test_frustration_and_disbelief_cues_are_explicit() -> None:
    assert interpret_query("I missed the bus and feel frustrated.", "base").intent_labels == ("frustration_setback",)
    assert interpret_query("I was confused by what happened.", "base").intent_labels == ("disbelief_confusion",)


def test_sarcasm_requires_explicit_indicator() -> None:
    assert "sarcasm_irony" not in interpret_query("Great, the elevator stopped again.", "base").intent_labels
    assert interpret_query("That was sarcastic irony.", "base").intent_labels == ("sarcasm_irony",)


def test_ambiguous_low_confidence_falls_back_to_base_representation() -> None:
    result = interpret_query("Need a meme.", "Situation: Need a meme.", "joy")
    assert result.confidence == "low"
    assert result.retrieval_representation == "Situation: Need a meme."
    assert result.ambiguity_reasons


def test_selected_mood_is_retained_without_overriding_explicit_intent() -> None:
    result = interpret_query("I passed the exam.", "base mood", "frustration")
    assert result.selected_mood == "frustration"
    assert result.intent_labels == ("celebration",)


def test_evidence_spans_are_stable() -> None:
    result = interpret_query("I passed the exam.", "base")
    assert result.evidence[0] == {"kind": "intent", "label": "celebration", "span": "passed", "start": "2", "end": "8"}
