I built a Call Quality Analyzer designed to run efficiently on the free Google Colab tier within ~30 seconds. The system downloads the provided YouTube test file, preprocesses it into mono 16 kHz audio, and applies Whisper (tiny) for fast speech-to-text with timestamps.

From the transcript, I compute key metrics:

Talk-time ratio (weighted by both duration and word count)

Number of questions asked (using regex + interrogative cues)

Longest continuous monologue

Overall call sentiment (positive/negative/neutral, averaged across turns)

A lightweight rule-based insight generator then suggests one actionable improvement (e.g., balance participation, ask more open-ended questions, handle objections). The notebook includes inline comments, runs fully on free Colab, and produces both printed results and a CSV export of speaker turns.

The design prioritizes speed, clarity, and reproducibility while leaving room for enhancements like pyannote.audio diarization for accurate speaker attribution and faster-whisper/OpenAI API for higher transcription accuracy.

This solution demonstrates practical AI/ML application in real-world voice analytics while meeting the given constraints.
