# Call Sentiment Analyzer API 🎧📊

A production-ready **ML-powered backend API** that analyzes call audio to extract
sentiment, conversation quality metrics, and actionable coaching insights.

Built with **FastAPI**, **OpenAI Whisper**, and **Hugging Face Transformers**.

---

## 🚀 Features

- Upload call audio (`wav`, `mp3`, `m4a`)
- Automatic speech-to-text (ASR)
- Call-level sentiment analysis
- Speaker dominance (talk-time ratio)
- Question count detection
- Longest monologue detection
- Actionable conversation insights
- REST API with Swagger documentation

---

## 🧠 Tech Stack

- **Backend:** FastAPI
- **ASR:** OpenAI Whisper (tiny, CPU-safe)
- **NLP:** Transformers (DistilBERT – SST-2)
- **Language:** Python 3.8+
- **Deployment:** Render (Docker-free)

---

## 📌 API Endpoint

### `POST /analyze-call`

Upload a call recording and receive sentiment + conversation metrics.

### Request
- **Content-Type:** `multipart/form-data`
- **Field:** `file`
- **Value:** audio file (`wav`, `mp3`, `m4a`)

### Sample Response
```json
{
  "filename": "call.wav",
  "sentiment": "POSITIVE",
  "metrics": {
    "talk_time_ratio": {
      "A": 0.71,
      "B": 0.29
    },
    "num_questions": 6,
    "longest_monologue_s": 38.4,
    "insight": "Speaker A is dominating (~71.0%). Encourage the other speaker to participate more."
  }
}

▶️ Run Locally
pip install -r requirements.txt
uvicorn app.main:app --reload


Open Swagger UI:

http://127.0.0.1:8000/docs
