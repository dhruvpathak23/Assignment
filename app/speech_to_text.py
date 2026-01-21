"""
speech_to_text.py

Handles automatic speech recognition (ASR) using OpenAI Whisper.
This module is intentionally lightweight and API-safe.
"""

import whisper
from typing import List, Dict

# ------------------------------
# Model configuration
# ------------------------------

# Use a small model for faster inference on CPU (good for demos & deployment)
WHISPER_MODEL_NAME = "tiny"

# Load the Whisper model ONCE at import time.
# This avoids reloading the model on every API request.
_whisper_model = whisper.load_model(WHISPER_MODEL_NAME)


def transcribe_audio(audio_path: str) -> List[Dict]:
    """
    Transcribe an audio file into speech segments using Whisper.

    Parameters
    ----------
    audio_path : str
        Path to the audio file (wav/mp3/m4a).

    Returns
    -------
    List[Dict]
        A list of Whisper segments.
        Each segment contains:
        - start (float): start time in seconds
        - end (float): end time in seconds
        - text (str): transcribed text
    """

    # Perform transcription
    result = _whisper_model.transcribe(
        audio_path,
        fp16=False,           # Disable FP16 for CPU compatibility
        word_timestamps=False # Segment-level timestamps are sufficient
    )

    # Whisper always returns a dict; segments may be empty if no speech
    segments = result.get("segments", [])

    # Normalize output to ensure consistent structure
    cleaned_segments = []
    for seg in segments:
        cleaned_segments.append({
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": str(seg.get("text", "")).strip()
        })

    return cleaned_segments
