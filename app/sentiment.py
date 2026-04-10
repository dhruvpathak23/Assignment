from transformers import pipeline

def analyze_sentiment(texts: List[str]) -> str:
    if not texts:
        return "NEUTRAL"
    
    # Load model only when needed
    sentiment_pipe = pipeline(
        "sentiment-analysis", 
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )
    
    positive = 0
    negative = 0
    for text in texts:
        result = sentiment_pipe(text[:512])[0]
        if result['label'].upper() == "POSITIVE":
            positive += 1
        elif result['label'].upper() == "NEGATIVE":
            negative += 1
            
    return "POSITIVE" if positive > negative else "NEGATIVE" if negative > positive else "NEUTRAL"
