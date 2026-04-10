import whisper

def transcribe_audio(audio_path: str) -> List[Dict]:
    # Load model locally to save boot-up RAM
    model = whisper.load_model("tiny")
    
    result = model.transcribe(audio_path, fp16=False)
    segments = result.get("segments", [])
    
    return [{
        "start": float(seg.get("start", 0.0)),
        "end": float(seg.get("end", 0.0)),
        "text": str(seg.get("text", "")).strip()
    } for seg in segments]
