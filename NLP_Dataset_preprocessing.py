import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (only once)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load your CSV
df = pd.read_csv(r'D:\Sparsh\ML_Projects\Fake_News_Detection\Dataset\final_news_dataset_clean.csv')  # replace with your file path

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if pd.isnull(text):
        return ""
    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation & numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # 3. Tokenization
    tokens = nltk.word_tokenize(text)
    # 4. Remove stopwords & lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # 5. Join back to string
    text = ' '.join(tokens)
    # 6. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply preprocessing to the column containing news text (replace 'text' with your column name)
df['cleaned_text'] = df['text'].apply(clean_text)

# Save cleaned dataset
df.to_csv('NLP_final_news_dataset.csv', index=False)

print("Preprocessing complete! Cleaned CSV saved as 'NLP_final_news_dataset_.csv'")

