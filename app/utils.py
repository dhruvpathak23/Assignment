"""
utils.py

Shared utility helpers used across the API.
Kept intentionally small and dependency-free.
"""

import os
from fastapi import UploadFile, HTTPException


# -----------------------------------
# File validation helpers
# -----------------------------------

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a"}


def validate_audio_file(file: UploadFile) -> None:
    """
    Validate uploaded audio file.

    Ensures:
    - File has a name
    - File extension is supported

    Raises HTTPException if invalid.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file has no filename."
        )

    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file format. "
                "Please upload wav, mp3, or m4a."
            )
        )


# -----------------------------------
# Numeric helpers
# -----------------------------------

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers.

    Prevents ZeroDivisionError and NaN propagation.
    """
    if denominator == 0:
        return default
    return numerator / denominator


# -----------------------------------
# Text helpers
# -----------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize text for NLP processing.

    - Strips whitespace
    - Converts to lowercase

    Keeps logic simple and predictable.
    """
    return text.strip().lower()
