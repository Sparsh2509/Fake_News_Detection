from fastapi import FastAPI # type: ignore
from pydantic import BaseModel # type: ignore
import joblib # type: ignore


# Load saved model and vectorizer

nb_model = joblib.load("nb_fake_news_model.joblib")
tfidf = joblib.load("tfidf_vectorizer.joblib")


app = FastAPI(title="Fake News Detection API")


class NewsItem(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Fake News Detection API is running!"}

label_map = {0: "Fake", 1: "Real"}

@app.post("/predict")
def predict_news(news: NewsItem):
    # Transform input
    transformed_text = tfidf.transform([news.text])
    prediction_num = nb_model.predict(transformed_text)[0]
    proba = nb_model.predict_proba(transformed_text)[0]

    # Get confidence percentage
    prediction = label_map[int(prediction_num)]
    confidence = float(round(max(proba) * 100, 2))

    # Generate response message
    if prediction == "Fake":
        message = "The article seems suspicious or misleading. Verify before sharing."
    else:
        message = "The article seems reliable based on the model analysis."

    advice = "Note: This prediction is based on the model and may not be 100% accurate. Always verify with trusted sources."


    return {
        "prediction": prediction,
        "confidence": f"{confidence}%",
        "message": message,

        "Final_Suggestion" :{
            "Suggestion": advice
        }
    }