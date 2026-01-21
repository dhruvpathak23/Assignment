from fastapi import FastAPI, UploadFile, File, HTTPException
import tempfile
import shutil
import os

from app.speech_to_text import transcribe_audio
from app.sentiment import analyze_sentiment
from app.metrics import analyze_metrics

app = FastAPI(
    title="Call Sentiment Analyzer API",
    description="Analyze call audio for sentiment and conversation quality",
    version="1.0.0",
)


@app.post("/analyze-call")
async def analyze_call(file: UploadFile = File(...)):
    """
    Accepts an audio file (wav/mp3),
    returns sentiment + call quality metrics as JSON.
    """

    # --- Basic validation ---
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Upload wav, mp3, or m4a.",
        )

    # --- Save uploaded file to a temp location ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        temp_audio_path = tmp.name

    try:
        # --- Speech-to-text ---
        segments = transcribe_audio(temp_audio_path)

        if not segments:
            raise HTTPException(
                status_code=422,
                detail="No speech detected in the audio.",
            )

        # --- Extract raw text for sentiment ---
        texts = [seg["text"] for seg in segments if seg.get("text")]

        # --- Sentiment analysis ---
        sentiment = analyze_sentiment(texts)

        # --- Call-quality metrics ---
        metrics = analyze_metrics(segments, sentiment)

        return {
            "filename": file.filename,
            "sentiment": sentiment,
            "metrics": metrics,
        }

    finally:
        # --- Always clean up temp file ---
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

