# sentiment_utils.py
from textblob import TextBlob
from typing import Tuple

def analyze_sentiment(text: str) -> Tuple[str, float]:
    """Return sentiment label + polarity score"""
    text = (text or "").strip()
    p = TextBlob(text).sentiment.polarity
    if p > 0.05:
        return "positive", float(p)
    elif p < -0.05:
        return "negative", float(p)
    else:
        return "neutral", float(p)

