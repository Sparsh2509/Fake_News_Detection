
# Step 3: Train & Evaluate Naive Bayes Classifier
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load cleaned dataset 
df = pd.read_csv(r'D:\Sparsh\ML_Projects\Fake_News_Detection\Dataset\NLP_final_news_dataset.csv')

# Prepare text and labels
texts = df['cleaned_text'].fillna('')  # handle missing values
y = df['label']  # target: Fake=0, Real=1

# TF-IDF Vectorization 
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
X = tfidf.fit_transform(texts)

# Train/Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize and Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Predictions 
y_pred = nb_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test set:", round(accuracy*100, 2), "%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
