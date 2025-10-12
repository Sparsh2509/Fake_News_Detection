# TF-IDF Feature Extraction
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

# Load the cleaned dataset
df = pd.read_csv(r'D:\Sparsh\ML_Projects\Fake_News_Detection\Dataset\NLP_final_news_dataset.csv')

# Check for missing values in 'cleaned_text' 
print("Number of missing values in 'cleaned_text':", df['cleaned_text'].isna().sum())

# Replace NaN with empty strings 
texts = df['cleaned_text'].fillna('')

# Initialize TF-IDF Vectorizer 
tfidf = TfidfVectorizer(
    max_features=5000,     # keep top 5000 words
    ngram_range=(1,2),     # unigrams + bigrams
    stop_words='english'   # remove common English stopwords
)

# Fit and transform the text data
X = tfidf.fit_transform(texts)

# store feature names for reference 
feature_names = tfidf.get_feature_names_out()

# Print result 
print("TF-IDF feature matrix shape:", X.shape)
print("Number of features (words):", len(feature_names))

# Optional: preview first 10 features 
print("First 10 features:", feature_names[:10])
