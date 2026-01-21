# Call Sentiment Analyzer API

An end-to-end ML-powered API that analyzes call audio to extract:
- Overall call sentiment
- Speaker dominance (talk-time ratio)
- Number of questions asked
- Longest monologue duration
- Actionable coaching insights

Built using **Whisper (ASR)**, **Transformers (NLP)**, and **FastAPI**.

---

## Features
- Upload call audio (wav/mp3/m4a)
- Automatic speech-to-text
- Sentiment analysis
- Conversation quality metrics
- JSON response via REST API

---

## API Endpoint

### POST `/analyze-call`

**Request**
- Form-data
- Key: `file`
- Value: audio file (wav/mp3/m4a)

**Response**
```json
{
  "filename": "call.wav",
  "sentiment": "POSITIVE",
  "metrics": {
    "talk_time_ratio": { "A": 0.72, "B": 0.28 },
    "num_questions": 5,
    "longest_monologue_s": 42.3,
    "insight": "Speaker A is dominating..."
  }
}

Run Locally :
pip install -r requirements.txt
uvicorn app.main:app --reload


Open:

http://127.0.0.1:8000/docs
