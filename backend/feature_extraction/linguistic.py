"""Extract 11 linguistic features from a transcript, matching the training schema.

Features produced (exactly as named in summary_stats.csv):
    text_filler_rate
    text_sentiment_positive, text_sentiment_negative
    text_first_person_singular_rate
    text_first_person_plural_rate
    text_third_person_rate
    text_negation_rate
    text_cognitive_process_rate
    text_exclusion_rate
    text_word_count
    text_mean_word_length
"""

import re

import numpy as np
from transformers import pipeline


# ── DistilBERT sentiment (matches training: top_k=None returns both scores) ──
_SENTIMENT = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    top_k=None,
)


# ── Filler words (matches training regex) ───────────────────────────
FILLER_PATTERN = re.compile(
    r"\b(uh+|um+|er+|ah+|hmm+|uhm+|mm+)\b",
    flags=re.IGNORECASE,
)


# ── LIWC-style categories (matches training add_features_cell.py) ───
LIWC_CATEGORIES = {
    "first_person_singular": [
        r"\bI\b", r"\bme\b", r"\bmy\b", r"\bmine\b", r"\bmyself\b", r"\bI'm\b",
        r"\bI've\b", r"\bI'd\b", r"\bI'll\b",
    ],
    "first_person_plural": [
        r"\bwe\b", r"\bus\b", r"\bour\b", r"\bours\b", r"\bourselves\b",
        r"\bwe're\b", r"\bwe've\b", r"\bwe'd\b", r"\bwe'll\b",
    ],
    "third_person": [
        r"\bhe\b", r"\bshe\b", r"\bthey\b", r"\bthem\b", r"\bhim\b", r"\bher\b",
        r"\bhis\b", r"\bhers\b", r"\btheir\b", r"\btheirs\b",
        r"\bhe's\b", r"\bshe's\b", r"\bthey're\b", r"\bthey've\b",
    ],
    "negation": [
        r"\bno\b", r"\bnot\b", r"\bnever\b", r"\bnothing\b", r"\bnone\b",
        r"\bnobody\b", r"\bnowhere\b", r"\bneither\b", r"\bnor\b",
        r"n't\b",
    ],
    "cognitive_process": [
        r"\bthink\b", r"\bknow\b", r"\bbecause\b", r"\breason\b", r"\bcause\b",
        r"\beffect\b", r"\bwhy\b", r"\bhow\b", r"\bunderstand\b",
        r"\brealise\b", r"\brealize\b", r"\bbelieve\b", r"\bconsider\b",
    ],
    "exclusion": [
        r"\bbut\b", r"\bexcept\b", r"\bwithout\b", r"\bunless\b",
        r"\bhowever\b", r"\balthough\b", r"\bthough\b",
    ],
}


def _word_tokenize(text: str) -> list[str]:
    """Extract words using the same pattern as training."""
    return re.findall(r"\b[\w']+\b", text.lower())


def _compute_liwc_rates(text: str, word_count: int) -> dict:
    """Compute per-category match rates (matches/word_count)."""
    text_lower = text.lower()
    rates = {}
    for category, patterns in LIWC_CATEGORIES.items():
        combined = "|".join(patterns)
        matches = len(re.findall(combined, text_lower, flags=re.IGNORECASE))
        rates[f"text_{category}_rate"] = matches / max(word_count, 1)
    return rates


def _compute_sentiment(text: str) -> dict:
    """Run DistilBERT SST-2 and return positive and negative confidence scores."""
    # pipeline with top_k=None returns [[{label, score}, {label, score}]]
    result = _SENTIMENT(text)

    # Unwrap outer list (batch dimension)
    scores = result[0] if isinstance(result[0], list) else result

    sentiment = {"text_sentiment_positive": 0.0, "text_sentiment_negative": 0.0}
    for item in scores:
        label = item["label"].upper()
        if label == "POSITIVE":
            sentiment["text_sentiment_positive"] = float(item["score"])
        elif label == "NEGATIVE":
            sentiment["text_sentiment_negative"] = float(item["score"])
    return sentiment


def extract_linguistic(transcript: str) -> dict:
    """Extract all 11 linguistic features from a transcript string.

    Args:
        transcript: The text to analyse (typically Whisper output)

    Returns:
        Dict with exactly the 11 keys expected by the trained model.
    """
    if not isinstance(transcript, str) or not transcript.strip():
        # Return zero-valued features rather than failing — allows the pipeline
        # to continue on edge cases (e.g., silent video).
        zero_features = {
            "text_filler_rate": 0.0,
            "text_sentiment_positive": 0.0,
            "text_sentiment_negative": 0.0,
            "text_word_count": 0,
            "text_mean_word_length": 0.0,
        }
        for category in LIWC_CATEGORIES:
            zero_features[f"text_{category}_rate"] = 0.0
        return zero_features

    # Word-level tokenisation
    words = _word_tokenize(transcript)
    word_count = len(words)

    # ── Structural features ────────────────────────────────────────
    features = {
        "text_word_count": word_count,
        "text_mean_word_length": float(np.mean([len(w) for w in words])) if words else 0.0,
    }

    # ── Filler rate ────────────────────────────────────────────────
    filler_matches = len(FILLER_PATTERN.findall(transcript))
    features["text_filler_rate"] = filler_matches / max(word_count, 1)

    # ── LIWC-style rates ───────────────────────────────────────────
    features.update(_compute_liwc_rates(transcript, word_count))

    # ── Sentiment ──────────────────────────────────────────────────
    features.update(_compute_sentiment(transcript))

    return features
