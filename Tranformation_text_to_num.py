# Step 1: Feature Extraction using TF-IDF
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore

# Load NLP cleaned dataset
df = pd.read_csv(r'D:\Sparsh\ML_Projects\Fake_News_Detection\Dataset\NLP_final_news_dataset.csv')  # your cleaned CSV

# Use the 'cleaned_text' column as input
texts = df['cleaned_text']

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000,  # top 5000 words
                        ngram_range=(1,2),   # unigrams + bigrams
                        stop_words='english')

# Fit and transform the text data
X = tfidf.fit_transform(texts)

print("TF-IDF feature matrix shape:", X.shape)
