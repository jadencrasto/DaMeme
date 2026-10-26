"""Deterministic, conservative query-intent interpretation for offline research."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

RULE_VERSION = "phase7r-query-intent-v1"


@dataclass(frozen=True)
class IntentResult:
    original_query: str
    situation_event: str | None
    intent_labels: tuple[str, ...]
    desired_tone: str | None
    selected_mood: str | None
    send_as_response: str
    ambiguity_reasons: tuple[str, ...]
    confidence: str
    retrieval_representation: str
    evidence: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("celebration", ("passed", "won", "promotion", "promoted", "graduated", "first try", "achievement", "celebrate")),
    ("frustration_setback", ("missed", "stuck", "late", "failed", "broke", "out of stock", "frustrated", "annoyed", "except mine")),
    ("disbelief_confusion", ("confused", "surprised", "what happened", "not understanding", "how did", "didn't expect")),
    ("sarcasm_irony", ("sarcasm", "sarcastic", "ironically", "ironic")),
    ("comfort_empathy", ("comfort", "support", "feel better", "sorry you", "there for")),
    ("embarrassment_self_deprecation", ("embarrassed", "embarrassment", "my fault", "made a fool", "awkward")),
    ("relatable_humor", ("relatable", "every time", "when you", "me when")),
    ("conversational_reaction", ("send", "reply", "response", "react", "reaction", "respond")),
)
TONE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sarcastic", ("sarcasm", "sarcastic", "ironically", "ironic")),
    ("celebratory", ("celebrate", "celebration", "excited")),
    ("comforting", ("comfort", "support", "sympathetic")),
)


def _matches(text: str, phrases: tuple[str, ...], kind: str, label: str) -> list[dict[str, str]]:
    values = []
    for phrase in phrases:
        match = re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, flags=re.IGNORECASE)
        if match:
            values.append({"kind": kind, "label": label, "span": match.group(0), "start": str(match.start()), "end": str(match.end())})
    return values


def interpret_query(original_query: str, mood_expanded_query: str, selected_mood: str | None = None) -> IntentResult:
    """Add one intent label only when explicit lexical evidence is unambiguous."""
    text = original_query.strip()
    if not text:
        raise ValueError("original query must be non-empty")
    evidence: list[dict[str, str]] = []
    labels: list[str] = []
    for label, phrases in RULES:
        matches = _matches(text, phrases, "intent", label)
        if matches:
            labels.append(label)
            evidence.extend(matches)
    tone_matches = [(tone, _matches(text, phrases, "tone", tone)) for tone, phrases in TONE_RULES]
    desired_tone = next((tone for tone, matches in tone_matches if matches), None)
    evidence.extend(match for _, matches in tone_matches for match in matches)
    send_as_response = "yes" if "conversational_reaction" in labels else "unknown"
    ambiguity = []
    if not labels:
        ambiguity.append("no explicit communicative-intent cue")
    if len(labels) > 1:
        ambiguity.append("multiple explicit intent cues")
    confidence = "high" if len(labels) == 1 and not ambiguity else ("medium" if labels else "low")
    retrieval = mood_expanded_query
    if confidence == "high":
        retrieval = f"{mood_expanded_query}\nIntent: {labels[0].replace('_', ' ')}."
    return IntentResult(
        original_query=original_query,
        situation_event=text,
        intent_labels=tuple(labels),
        desired_tone=desired_tone,
        selected_mood=selected_mood,
        send_as_response=send_as_response,
        ambiguity_reasons=tuple(ambiguity),
        confidence=confidence,
        retrieval_representation=retrieval,
        evidence=tuple(sorted(evidence, key=lambda item: (int(item["start"]), item["label"]))),
    )
