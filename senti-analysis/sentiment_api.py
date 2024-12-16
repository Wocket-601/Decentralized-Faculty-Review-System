from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import re
import string
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

# Load the trained model and vectorizer
def load_model_and_vectorizer(model_path="sentiment_model.pkl", vectorizer_path="vectorizer.pkl"):
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    with open(vectorizer_path, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Preprocess input text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Predict sentiment for a single review
def predict_sentiment(review, model, vectorizer):
    review_cleaned = preprocess_text(review)
    X = vectorizer.transform([review_cleaned])
    predicted_sentiment = model.predict(X)[0]
    return predicted_sentiment

# Calculate sentiment scores
def calculate_scores(reviews, model, vectorizer):
    sentiment_scores = {
        "happy": 100,
        "sad": 20,
        "angry": 30,
        "fearful": 40,
        "neutral": 50,
        "cheated": 10,
        "loved": 90,
        "attached": 80,
        "bored": 25,
        "safe": 95,
        "lustful": 70,
        "independent": 85,
        "alone": 15,
        "powerless": 5,
        "focused": 90,
        "demoralized": 10,
        "obsessed": 50,
        "average": 60,
        "embarrassed": 35,
        "esteemed": 95,
        "surprise": 60
    }

    sentiment_counts = {}
    total_score = 0

    for review in reviews:
        sentiment = predict_sentiment(review, model, vectorizer)
        score = sentiment_scores.get(sentiment, 50)  # Default to 50 if sentiment not mapped
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        total_score += score

    mean_score = total_score / len(reviews)
    return mean_score, sentiment_counts

# Define input schema
class ReviewInput(BaseModel):
    reviews: List[str]

# Load the model and vectorizer at startup
model, vectorizer = load_model_and_vectorizer()

@app.post("/analyze")
async def analyze_sentiment(input_data: ReviewInput):
    try:
        reviews = input_data.reviews
        if not reviews:
            raise HTTPException(status_code=400, detail="No reviews provided.")

        mean_score, sentiment_distribution = calculate_scores(reviews, model, vectorizer)

        return {
            "mean_score": mean_score,
            "sentiment_distribution": sentiment_distribution
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sentiment_api:app", host="127.0.0.1", port=8000, reload=True)
