from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

# Load saved model and vectorizer
model_dir = r'D:\Sparsh\ML_Projects\Fake_News_Detection\Model'
nb_model = joblib.load(os.path.join(model_dir, 'nb_fake_news_model.joblib'))
tfidf = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.joblib'))


app = FastAPI(title="Fake News Detection API", version="1.0")


# Request body schema
class NewsArticle(BaseModel):
    title: str = None   # optional
    text: str           # main content for prediction


#Routes
@app.get("/")
def home():
    return {"message": "Welcome to the Fake News Detection API!"}


@app.post("/predict")
def predict_news(article: NewsArticle):
    # Combine title and text if title exists
    content = article.text if article.title is None else article.title + " " + article.text

    # Transform using TF-IDF
    vector = tfidf.transform([content])

    # Make prediction
    pred = nb_model.predict(vector)[0]

    # Convert label to readable format
    label = "Real" if pred == 1 else "Fake"

    # Return response
    return {
        "prediction": label
    }

