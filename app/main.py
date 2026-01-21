"""
main.py

FastAPI entry point for the Call Sentiment Analyzer API.
Handles request validation, orchestration, and response formatting.
"""

import os
import tempfile
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException

from app.speech_to_text import transcribe_audio
from app.sentiment import analyze_sentiment
from app.metrics import analyze_metrics
from app.utils import validate_audio_file


# -----------------------------------
# FastAPI app initialization
# -----------------------------------

app = FastAPI(
    title="Call Sentiment Analyzer API",
    description=(
        "Analyze call audio to extract sentiment, speaker dominance, "
        "conversation quality metrics, and actionable insights."
    ),
    version="1.0.0",
)


# -----------------------------------
# API endpoint
# -----------------------------------

@app.post("/analyze-call")
async def analyze_call(file: UploadFile = File(...)):
    """
    Upload an audio file and receive call sentiment + quality metrics.

    Supported formats: wav, mp3, m4a
    """

    # --- Validate uploaded file ---
    validate_audio_file(file)

    temp_audio_path = None

    try:
        # --- Save uploaded audio to a temporary file ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_audio_path = tmp.name

        # --- Speech-to-text ---
        segments = transcribe_audio(temp_audio_path)

        if not segments:
            raise HTTPException(
                status_code=422,
                detail="No speech detected in the uploaded audio.",
            )

        # --- Extract transcript text for sentiment analysis ---
        texts = [seg["text"] for seg in segments if seg.get("text")]

        # --- Sentiment analysis ---
        sentiment = analyze_sentiment(texts)

        # --- Call-quality metrics ---
        metrics = analyze_metrics(segments, sentiment)

        # --- Final API response ---
        return {
            "filename": file.filename,
            "sentiment": sentiment,
            "metrics": metrics,
        }

    finally:
        # --- Ensure temp file cleanup ---
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
