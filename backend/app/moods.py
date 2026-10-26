"""Static, documented user-selectable mood taxonomy; it is not a classifier."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Mood:
    id: str
    label: str
    group: str
    description: str
    synonyms: tuple[str, ...]
    classifier_weights: dict[str, float]
    sentiment: str | None = None


def _mood(id: str, group: str, description: str, synonyms: tuple[str, ...], weights: dict[str, float], sentiment: str | None = None) -> Mood:
    return Mood(id, id.replace("_", " ").title(), group, description, synonyms, weights, sentiment)


MOODS = (
    _mood("joy", "Positive and uplifting", "a happy, cheerful moment", ("happy", "happiness", "delight"), {"joy": 1}, "positive"),
    _mood("excitement", "Positive and uplifting", "energized anticipation about something happening", ("thrilled", "pumped"), {"joy": .75, "surprise": .25}, "positive"),
    _mood("amusement", "Positive and uplifting", "finding a situation funny or ridiculous", ("funny", "humor"), {"joy": .7, "surprise": .3}, "positive"),
    _mood("love", "Positive and uplifting", "affection, warmth, or caring connection", ("affection", "fondness"), {"joy": .8}, "positive"),
    _mood("gratitude", "Positive and uplifting", "appreciation for help, luck, or another person", ("thankful", "appreciation"), {"joy": .75}, "positive"),
    _mood("pride", "Positive and uplifting", "satisfaction in an achievement or effort", ("accomplished", "achievement"), {"joy": .8}, "positive"),
    _mood("confidence", "Positive and uplifting", "self-assurance before or after a challenge", ("self_assured", "bold"), {"joy": .55, "surprise": .1}, "positive"),
    _mood("relief", "Positive and uplifting", "release after worry, pressure, or uncertainty ends", ("reassured", "unburdened"), {"joy": .55, "fear": .25}, "positive"),
    _mood("sadness", "Sad and reflective", "feeling down after a loss, setback, or difficult moment", ("sad", "unhappy"), {"sadness": 1}, "negative"),
    _mood("loneliness", "Sad and reflective", "feeling isolated or disconnected from others", ("isolated", "alone"), {"sadness": .8}, "negative"),
    _mood("disappointment", "Sad and reflective", "an outcome falling short of expectations", ("let_down", "disappointed"), {"sadness": .65, "anger": .15}, "negative"),
    _mood("nostalgia", "Sad and reflective", "reflecting warmly or wistfully on the past", ("wistful", "remembering"), {"sadness": .35, "joy": .25}, "neutral"),
    _mood("guilt", "Sad and reflective", "regret about one's own action or omission", ("remorse", "regret"), {"sadness": .65}, "negative"),
    _mood("embarrassment", "Sad and reflective", "self-conscious discomfort after a social mistake", ("awkward", "humiliated"), {"fear": .45, "sadness": .35}, "negative"),
    _mood("grief", "Sad and reflective", "deep sadness following serious loss", ("mourning", "bereavement"), {"sadness": .95}, "negative"),
    _mood("boredom", "Sad and reflective", "under-stimulated waiting or lack of interest", ("bored", "restless"), {"sadness": .35}, "neutral"),
    _mood("anger", "Stress and negative", "anger at unfairness, conflict, or obstruction", ("mad", "furious"), {"anger": 1}, "negative"),
    _mood("frustration", "Stress and negative", "blocked effort or repeated difficulty", ("annoyed", "fed_up"), {"anger": .7, "sadness": .15}, "negative"),
    _mood("anxiety", "Stress and negative", "uneasy concern about an uncertain outcome", ("anxious", "worried"), {"fear": .75, "sadness": .1}, "negative"),
    _mood("fear", "Stress and negative", "feeling threatened or afraid of danger", ("scared", "afraid"), {"fear": 1}, "negative"),
    _mood("stress", "Stress and negative", "pressure from demands, deadlines, or overload", ("stressed", "pressure"), {"fear": .45, "anger": .2, "sadness": .15}, "negative"),
    _mood("disgust", "Stress and negative", "strong aversion or revulsion", ("grossed_out", "repulsed"), {"disgust": 1}, "negative"),
    _mood("jealousy", "Stress and negative", "resentful comparison or fear of losing attention", ("envious", "envy"), {"anger": .45, "sadness": .3}, "negative"),
    _mood("overwhelm", "Stress and negative", "too many demands or feelings at once", ("overwhelmed", "swamped"), {"fear": .35, "sadness": .3, "anger": .15}, "negative"),
    _mood("surprise", "Anticipation and neutral", "an unexpected event or revelation", ("surprised", "shocked"), {"surprise": 1}, "neutral"),
    _mood("curiosity", "Anticipation and neutral", "interest in finding out or understanding", ("curious", "intrigued"), {"surprise": .45, "joy": .15}, "neutral"),
    _mood("hope", "Anticipation and neutral", "wanting a positive uncertain outcome", ("hopeful", "wishful"), {"joy": .45, "fear": .15}, "positive"),
    _mood("anticipation", "Anticipation and neutral", "waiting expectantly for a coming event", ("expectant", "looking_forward"), {"surprise": .45, "joy": .3}, "neutral"),
    _mood("optimism", "Anticipation and neutral", "positive expectation about what comes next", ("optimistic", "positive_outlook"), {"joy": .5}, "positive"),
    _mood("confusion", "Anticipation and neutral", "not understanding what happened or what to do", ("confused", "puzzled"), {"surprise": .5, "fear": .15}, "neutral"),
    _mood("trust", "Anticipation and neutral", "confidence in a person, plan, or process", ("reliance", "faith"), {"joy": .3}, "positive"),
    _mood("calm", "Anticipation and neutral", "peaceful steadiness without urgency", ("relaxed", "peaceful"), {"joy": .25, "fear": .05}, "neutral"),
)
MOODS_BY_ID = {mood.id: mood for mood in MOODS}
_ALIASES = {alias.lower().replace("-", "_").replace(" ", "_"): mood.id for mood in MOODS for alias in (mood.id, *mood.synonyms)}


def resolve_mood(value: str | None) -> Mood | None:
    if value is None or not value.strip():
        return None
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    mood = MOODS_BY_ID.get(_ALIASES.get(key, ""))
    if mood is None:
        raise ValueError("mood must be one of the supported mood identifiers")
    return mood


def mood_query(query: str, mood: Mood | None) -> str:
    """Add compact, explainable context without replacing the situation text."""
    if mood is None:
        return query
    return f"Situation: {query}\nSelected mood context: {mood.description}."


def mood_only_query(mood: Mood) -> str:
    """Build the fixed A0 query used when a user intentionally provides only a mood.

    This is taxonomy metadata, not an inferred situation or an emotion prediction.
    """
    related_terms = ", ".join((mood.id.replace("_", " "), *mood.synonyms))
    return f"Selected mood: {mood.label}.\nMood description: {mood.description}.\nRelated terms: {related_terms}."


def primary_classifier_emotion(mood: Mood | None) -> str | None:
    return max(mood.classifier_weights, key=mood.classifier_weights.get) if mood and mood.classifier_weights else None
