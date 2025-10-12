import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download stopwords and wordnet only 
nltk.download('stopwords')
nltk.download('wordnet')

# Load your CSV 
df = pd.read_csv(r'D:\Sparsh\ML_Projects\Fake_News_Detection\Dataset\final_news_dataset_clean.csv')  # replace with your file path

# Initialize lemmatizer and stopwords 
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Simple Python tokenizer function 
def simple_tokenize(text):
    # Split on spaces
    return text.split()

# Preprocessing function 
def clean_text(text):
    if pd.isnull(text):
        return ""
    # Lowercase
    text = text.lower()
    # Remove punctuation & numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenization (Python split instead of nltk.word_tokenize)
    tokens = simple_tokenize(text)
    # Remove stopwords & lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # Join back to string
    text = ' '.join(tokens)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply preprocessing 
df['cleaned_text'] = df['text'].apply(clean_text)  # replace 'text' with your column name

# Save cleaned dataset
df.to_csv('NLP_final_news_dataset.csv', index=False)
print("Preprocessing complete! Cleaned CSV saved as 'NLP_final_news_dataset.csv'")
