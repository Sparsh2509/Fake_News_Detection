import pandas as pd

file_path = r"D:\Sparsh\ML_Projects\Fake_News_Detection\Dataset\final_news_dataset.csv"

# Step 1: Read CSV safely — force pandas to treat commas inside text properly
# 'error_bad_lines=False' is deprecated, so use 'on_bad_lines="skip"'
# Try different quotechar just in case — many CSVs use double quotes
df = pd.read_csv(file_path, sep=',', quotechar='"', engine='python', on_bad_lines='skip')

print("Columns found:", df.columns.tolist())
print("Before cleaning:", df.shape)

# Step 2: Remove unnamed columns automatically
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Step 3: Ensure we have the expected 5 columns only
expected_cols = ['title', 'text', 'subject', 'date', 'label']
df = df[[col for col in df.columns if col in expected_cols]]

# Step 4: Handle missing data
df['title'] = df['title'].fillna("No Title")
df['text'] = df['text'].fillna("")
df['subject'] = df['subject'].fillna("Unknown")
df['date'] = df['date'].fillna("Unknown")
df['label'] = df['label'].fillna("Unknown")

# Step 5: Drop duplicate news texts
df = df.drop_duplicates(subset=['text']).reset_index(drop=True)

print("After cleaning:", df.shape)

# Step 6: Save cleaned dataset
output_path = "final_news_dataset_clean.csv"
df.to_csv(output_path, index=False, encoding='utf-8')

print(f"✅ Cleaned dataset saved as {output_path}")
