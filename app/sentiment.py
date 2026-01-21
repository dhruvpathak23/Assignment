"""
sentiment.py

Handles sentiment analysis for call transcripts using Hugging Face Transformers.
Designed for API usage (model loaded once, fast inference).
"""

from typing import List
from transformers import pipeline


# -----------------------------------
# Model configuration
# -----------------------------------

# Pretrained sentiment model (robust + lightweight)
SENTIMENT_MODEL_NAME = (
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

# Load the sentiment pipeline ONCE at import time.
# This avoids expensive reloads on every API request.
_sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=SENTIMENT_MODEL_NAME
)


def analyze_sentiment(texts: List[str]) -> str:
    """
    Analyze overall sentiment of a call based on transcript text.

    Strategy:
    - Run sentiment on each text chunk
    - Aggregate results using majority vote
    - Return a single label: POSITIVE / NEGATIVE / NEUTRAL

    Parameters
    ----------
    texts : List[str]
        List of transcribed text segments from the call.

    Returns
    -------
    str
        Overall call sentiment.
    """

    if not texts:
        # Defensive default when transcript is empty
        return "NEUTRAL"

    positive = 0
    negative = 0

    for text in texts:
        # Truncate long text to avoid transformer token limits
        truncated_text = text[:512]

        result = _sentiment_pipeline(truncated_text)[0]
        label = result.get("label", "").upper()

        if label == "POSITIVE":
            positive += 1
        elif label == "NEGATIVE":
            negative += 1

    # Majority vote logic
    if positive > negative:
        return "POSITIVE"
    elif negative > positive:
        return "NEGATIVE"
    else:
        return "NEUTRAL"
