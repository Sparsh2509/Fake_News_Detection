import pandas as pd

# Load merged dataset
df = pd.read_csv("D:\Sparsh\ML_Projects\Fake_News_Detection\Dataset\final_news_dataset.csv")

print("Before cleaning:", df.shape)

# Drop rows where 'text' or 'label' is missing
df = df.dropna(subset=['text', 'label'])

# Step 2: Fill missing 'title' or 'subject' with placeholder (optional)
df['title'] = df['title'].fillna("No Title")
df['subject'] = df['subject'].fillna("Unknown")
df['date'] = df['date'].fillna("Unknown")

# Drop duplicate rows (to avoid repetition)
df = df.drop_duplicates(subset=['text'])

# Reset index after cleaning
df = df.reset_index(drop=True)

print("After cleaning:", df.shape)

# Save cleaned dataset
df.to_csv("final_news_dataset_clean.csv", index=False)
print("✅ Cleaned dataset saved as final_news_dataset_clean.csv")
