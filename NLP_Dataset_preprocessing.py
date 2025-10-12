# Preprocessing for Fake News Detection without punkt_tab issue
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Download stopwords and wordnet only ---
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# --- Load your CSV ---
df = pd.read_csv(r'D:\Sparsh\ML_Projects\Fake_News_Detection\Dataset\final_news_dataset_clean.csv')  # replace with your file path

# --- Initialize lemmatizer and stopwords ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- Simple Python tokenizer function ---
def simple_tokenize(text):
    # Split on spaces
    return text.split()

# --- Preprocessing function ---
def clean_text(text):
    if pd.isnull(text):
        return ""
    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation & numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # 3. Tokenization (Python split instead of nltk.word_tokenize)
    tokens = simple_tokenize(text)
    # 4. Remove stopwords & lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # 5. Join back to string
    text = ' '.join(tokens)
    # 6. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Apply preprocessing ---
df['cleaned_text'] = df['text'].apply(clean_text)  # replace 'text' with your column name

# --- Save cleaned dataset ---
df.to_csv('final_news_dataset_cleaned.csv', index=False)

print("Preprocessing complete! Cleaned CSV saved as 'final_news_dataset_cleaned.csv'")
