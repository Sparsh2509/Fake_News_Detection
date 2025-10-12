# 📰 Fake News Detection

A machine learning-powered web API using FastAPI to detect whether a news article is Fake or Real.
This project is based on the dataset by kaggle [Fake-and-Real-News-Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).

---

## 🚀 Features

- Predicts Fake or Real news based on article content.
- Provides confidence score for prediction.
- Returns human-readable advisory messages for user awareness.
- Built with Naive Bayes classifier and TF-IDF feature extraction.
- Organized input and output for easy API integration.

---

## 🧠 Machine Learning Prediction (Naive Bayes)

- Uses a trained Naive Bayes model with TF-IDF features to predict news credibility.
- Outputs:
  - Prediction → Fake or Real
  - Confidence (%) → How confident the model is
  - Advisory messages → Suggestions to verify information

- Achieves an accuracy of ~93.24%.

---

## 💡 Prediction Interpretation

- Fake → "The article seems suspicious or misleading. Verify before sharing."
- Real → "The article seems reliable based on the model analysis."
- Final Suggestion → "Note: This prediction is based on the model and may not be 100% accurate. Always verify with trusted sources."


---

## 📂 Project Structure

```
Heart_Disease_Prediction/
├── app.py                                              # FastAPI backend logic
├── Heart_Disease_Predict_Model.py                      # Model training script
├── Heart_disease_cleveland_new.csv                     # Cleaned cleveland dataset used for training
├── randomforest_heart_model.joblib                     # Trained Random forest classifier model
├── feature_columns.joblib                              # Saved feature order for clean prediction
├── requirements.txt                                    # Python package dependencies
└── README.md                                           # Project documentation
```

---

## 🛠️ Installation

### Prerequisites:
- Python 3.9.5

### Setup:
```bash
# Repository Name
Heart_Disease_Risk_Prediction

# Create virtual environment
python -m venv venv
venv\Scripts\activate 

# Install dependencies
pip install -r requirements.txt
```

---

## 🏋️‍♂️ Model Training

To train or retrain the model using the dataset:

```bash
python Heart_Disease_Predict_Model.py
```

This will generate the model and features order files:
- `randomforest_heart_model.joblib `
- `feature_columns.joblib`

---

## 🚦 Running the API

Start the FastAPI server:
```bash
uvicorn app:app --reload
```

Navigate to:
- Swagger Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Root: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Render: [https://heart-disease-risk-prediction-hpkk.onrender.com](https://heart-disease-risk-prediction-hpkk.onrender.com)

---

## 📥 API Usage

### Endpoint:
```
POST /predict
```

### Request Body Example:
```json
{
  "age": 56,
  "sex": 1,
  "cp": 2,
  "trestbps": 150,
  "chol": 260,
  "fbs": 1,
  "restecg": 1,
  "thalach": 120,
  "exang": 1,
  "oldpeak": 2.5,
  "slope": 2,
  "ca": 2,
  "thal": 2
}

```

### Sample Response:
```json
{
  "ml_prediction": {
    "heart_disease_probability": "82.5%",
    "no_disease_probability": "17.5%",
    "ml_risk_message": "High risk — please consult a cardiologist immediately"
  },
  "score_prediction": {
    "risk_score": 13,
    "risk_level": "High Risk",
    "threshold_flags": [
      "Older age (>50 years)",
      "High resting blood pressure (>140 mm Hg)",
      "High cholesterol level (>240 mg/dl)",
      "Low max heart rate (<130 bpm)",
      "Exercise-induced angina detected",
      "Significant ST depression (>1.5)"
    ]
  },
  "final_advice": "The ML model and scoring system together indicate your overall heart risk. For accurate diagnosis, please consult a healthcare professional."
}

```

---

## 🧠 Model Overview

- Algorithm: `Random Forest Classifier`
- Input Features: 13
- Target: `Target` (0 or 1)
- Evaluation: Accuracy ~ 90.20%

---

## 📘 Dataset Info

- Source: Heart Disease Cleveland UCI
- [Heart Disease Cleveland UCI](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci?resource=download)

---

