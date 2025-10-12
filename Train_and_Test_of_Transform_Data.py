
# Train/Test Split
import pandas as pd
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore

# Load cleaned dataset
df = pd.read_csv(r'D:\Sparsh\ML_Projects\Fake_News_Detection\Dataset\NLP_final_news_dataset.csv')

# Fill missing cleaned_text
texts = df['cleaned_text'].fillna('')

# Target labels: Fake=0, Real=1
y = df['label']  # make sure 'label' column exists

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
X = tfidf.fit_transform(texts)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Print shapes to verify
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)
