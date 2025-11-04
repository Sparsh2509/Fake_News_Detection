# 📰 Fake News Detection

A machine learning-powered web API using FastAPI to detect whether a news article is Fake or Real.
This project is based on the dataset by kaggle [Fake-and-Real-News-Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).

### Deployed Project Using API on Streamlit 

 Test APP HERE -
 [https://fakenewsdetection-fcsvxd6c9n6kxyprviuvtl.streamlit.app/](https://fakenewsdetection-fcsvxd6c9n6kxyprviuvtl.streamlit.app/)

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
Fake_News_Detection/
├── Dataset/
│   └── NLP_final_news_dataset.csv                # Cleaned dataset
├── nb_fake_news_model.joblib                     # Trained Naive Bayes model 
├── tfidf_vectorizer.joblib                       # Saved TF-IDF vectorizer                       
├── NLP_Dataset_preprocessing.py                  # Dataset cleaning and preprocessing
├── Tranformation_text_to_num.py                  # TF-IDF vectorization
├── Train_and_Test_of_Transform_Data.py           # Train/test split and evaluation
├── Fake_news_detection_model.py                  # Train and save model
├── app.py                                        # FastAPI backend
├── requirements.txt                              # Python dependencies
├── streamlit_app.py                              # Deployed App Using API On Streamlit
└── README.md                                     # Project documentation
```

---

## 🛠️ Installation

### Prerequisites:
- Python 3.13.7

### Setup:
```bash
# Repository Name
Fake_News_Detection

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
python Fake_news_detection_model.py
```
Preprocess the dataset
- ` NLP_Dataset_preprocessing.py `

Transform text to numeric TF-IDF features
- ` Tranformation_text_to_num.py `

Train and evaluate model
- ` Train_and_Test_of_Transform_Data.py `

This will generate the model and saved TF-IDF vectorizer files:
- `nb_fake_news_model.joblib `
- `tfidf_vectorizer.joblib`

---

## 🚦 Running the API

Start the FastAPI server:
```bash
uvicorn app:app --reload
```

Navigate to:
- Swagger Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Root: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- AWS EB API URL: [http://fake-news-detection2-env.eba-gdmi2vhg.ap-south-1.elasticbeanstalk.com](http://fake-news-detection2-env.eba-gdmi2vhg.ap-south-1.elasticbeanstalk.com)

---

## 📥 API Usage

### Endpoint:
```
POST /predict
```

### Request Body Example:
```json
{
    "text": "An anonymous insider revealed that Coca-Cola will halt production worldwide after their secret formula was leaked on the dark web. The company has not yet responded to the allegations."
}

```

### Sample Response:
```json
{
  "prediction": "Fake",
    "confidence": "81.97%",
    "message": "The article seems suspicious or misleading. Verify before sharing.",
    "Final_Suggestion": {
        "Suggestion": "Note: This prediction is based on the model and may not be 100% accurate. Always verify with trusted sources."
    }
}

```

---

## 🧠 Model Overview

- Algorithm: `Multinomial Naive Bayes`
- Feature Extraction: `TF-IDF`
- Input Features: News text
- Target: `Target` (0 or 1) (Fake : 0 or Real : 1)
- Evaluation: Accuracy ~ 93.24%

---

## 📘 Dataset Info

- Source: Fake-and-Real-News-Dataset
- [Fake-and-Real-News-Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

