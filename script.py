"""
script.py — script analysis with act-based emotion detection

Returns not one emotion but a list of (emotion, tension) per act,
so the video grade can evolve across the runtime of the clip.
"""

import re
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword → canonical emotion
# Uses word-boundary regex — "war" will NOT match "toward"
# ---------------------------------------------------------------------------
KEYWORD_TO_EMOTION = {
    # happy
    "love":        "happy",
    "romance":     "happy",
    "joy":         "happy",
    "happiness":   "happy",
    "hope":        "happy",
    "cheerful":    "happy",
    "comedy":      "happy",
    "celebration": "happy",
    "triumph":     "happy",
    "laugh":       "happy",
    "smile":       "happy",
    "warm":        "happy",

    # sad
    "sad":         "sad",
    "grief":       "sad",
    "loss":        "sad",
    "melancholy":  "sad",
    "tragedy":     "sad",
    "noir":        "sad",
    "weep":        "sad",
    "sorrow":      "sad",
    "mourn":       "sad",
    "lonely":      "sad",

    # angry
    "anger":       "angry",
    "rage":        "angry",
    "fury":        "angry",
    "conflict":    "angry",
    "violence":    "angry",
    "revenge":     "angry",
    "danger":      "angry",
    "fight":       "angry",
    "attack":      "angry",
    "shout":       "angry",
    "slam":        "angry",

    # fearful — full horror/thriller screenplay vocabulary
    "fear":        "fearful",
    "terror":      "fearful",
    "suspense":    "fearful",
    "dread":       "fearful",
    "paranoia":    "fearful",
    "thriller":    "fearful",
    "panic":       "fearful",
    "horror":      "fearful",
    "shadow":      "fearful",
    "shadows":     "fearful",
    "darkness":    "fearful",
    "dark":        "fearful",
    "flicker":     "fearful",
    "distort":     "fearful",
    "distorted":   "fearful",
    "glitch":      "fearful",
    "scream":      "fearful",
    "tremble":     "fearful",
    "trembles":    "fearful",
    "freeze":      "fearful",
    "frozen":      "fearful",
    "stiff":       "fearful",
    "static":      "fearful",
    "ghost":       "fearful",
    "haunt":       "fearful",
    "creep":       "fearful",
    "whisper":     "fearful",
    "silence":     "fearful",
    "watching":    "fearful",
    "pulse":       "fearful",
    "hum":         "fearful",
    "crackle":     "fearful",
    "violent":     "fearful",

    # neutral
    "neutral":     "neutral",
    "documentary": "neutral",
    "drama":       "neutral",
}

# Words that indicate high-tension moments regardless of base emotion
TENSION_MARKERS = {
    "suddenly", "violently", "rapidly", "abruptly", "screams", "slams",
    "crashes", "explodes", "shatters", "lurches", "spins",
    "cut to black", "smash cut", "everything goes dark",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_act_boundaries(text: str) -> list:
    """
    Return list of (start, end) char index pairs for 3 acts.
    Snaps to scene headings (INT./EXT.) when available near the 1/3 marks,
    otherwise falls at exactly 33% and 66%.
    """
    n = len(text)
    heading_positions = [
        m.start() for m in re.finditer(r'(?m)^(INT\.|EXT\.|INT/EXT\.)', text)
    ]

    snapped = [0]
    for target in [n // 3, 2 * n // 3]:
        window = n * 0.15
        candidates = [p for p in heading_positions if abs(p - target) < window]
        if candidates:
            snapped.append(min(candidates, key=lambda p: abs(p - target)))
        else:
            blank = text.rfind('\n\n', target - 200, target + 200)
            snapped.append(blank if blank != -1 else target)
    snapped.append(n)

    return [(snapped[i], snapped[i + 1]) for i in range(len(snapped) - 1)]


def _tension_score(segment: str) -> float:
    """
    Return 0.0–1.0 representing escalation intensity in this segment.
    Driven by tension marker words and ALL-CAPS stage direction emphasis.
    """
    lower = segment.lower()
    hits = sum(1 for m in TENSION_MARKERS if m in lower)
    caps_words = len(re.findall(r'\b[A-Z]{3,}\b', segment))
    raw = hits * 2 + min(caps_words, 10)
    return min(raw / 20.0, 1.0)


def _score_segment(text: str) -> defaultdict:
    """Score text segment using word-boundary regex matching."""
    lower = text.lower()
    scores = defaultdict(int)
    for keyword, canonical in KEYWORD_TO_EMOTION.items():
        count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', lower))
        if count:
            scores[canonical] += count
    return scores


def _dominant_emotion(scores: defaultdict) -> str:
    if not scores:
        return "neutral"
    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_emotion(script_text: str) -> str:
    """Single emotion for the whole script — simple fallback mode."""
    scores = _score_segment(script_text)
    result = _dominant_emotion(scores)
    logger.debug(f"Single emotion: {result} | scores: {dict(scores)}")
    return result


def extract_acts(script_text: str) -> list:
    """
    Split the script into 3 acts and return an emotion + tension profile
    for each.

    Returns list of 3 dicts:
      [
        {"act": 1, "emotion": "fearful", "tension": 0.1, "start": 0.0,  "end": 0.33},
        {"act": 2, "emotion": "fearful", "tension": 0.5, "start": 0.33, "end": 0.66},
        {"act": 3, "emotion": "fearful", "tension": 0.9, "start": 0.66, "end": 1.0},
      ]

    "start"/"end" are 0.0–1.0 positions in the VIDEO timeline.
    "tension" modulates vignette + grain beyond the base emotion profile.
    """
    if not script_text or not script_text.strip():
        return [
            {"act": i+1, "emotion": "neutral", "tension": float(i)/2,
             "start": i/3.0, "end": (i+1)/3.0}
            for i in range(3)
        ]

    boundaries = _find_act_boundaries(script_text)
    n = len(script_text)
    acts = []

    for i, (start_char, end_char) in enumerate(boundaries):
        segment = script_text[start_char:end_char]
        scores  = _score_segment(segment)
        emotion = _dominant_emotion(scores)
        tension = _tension_score(segment)

        acts.append({
            "act":     i + 1,
            "emotion": emotion,
            "tension": tension,
            "start":   start_char / n,
            "end":     end_char / n,
        })
        logger.debug(
            f"Act {i+1}: emotion={emotion}, tension={tension:.2f}, "
            f"scores={dict(scores)}, chars={start_char}–{end_char}"
        )

    # If all acts resolved to the same emotion (common for short scripts),
    # enforce a tension progression so the grade still evolves visually.
    if len({a["emotion"] for a in acts}) == 1:
        logger.debug("All acts same emotion — enforcing tension progression")
        acts[0]["tension"] = max(acts[0]["tension"], 0.1)
        acts[1]["tension"] = max(acts[1]["tension"], 0.45)
        acts[2]["tension"] = max(acts[2]["tension"], 0.85)

    return acts
