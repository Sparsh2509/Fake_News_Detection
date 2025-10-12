import pandas as pd

# ✅ Use raw string to avoid escape issues in Windows path
file_path = r"D:\Sparsh\ML_Projects\Fake_News_Detection\Dataset\final_news_dataset.csv"

# Step 1: Safely read the file using 'python' engine and skip bad lines
df = pd.read_csv(file_path, engine="python", header=None, on_bad_lines='skip')

# Step 2: Merge all text columns into one column
df['merged'] = df.apply(lambda x: ' '.join(str(v) for v in x if pd.notna(v)), axis=1)

# Step 3: Extract label (0 or 1 at the end)
df['label'] = df['merged'].str.extract(r'(\b[01]\b)$')
df['label'] = df['label'].fillna(method='ffill').fillna(method='bfill')

# Step 4: Clean text (remove the label number from the end)
df['text'] = df['merged'].str.replace(r'\b[01]\b$', '', regex=True).str.strip()

# Step 5: Keep only text and label
df = df[['text', 'label']]

# Step 6: Remove missing or blank values
df.dropna(subset=['text', 'label'], inplace=True)
df = df[df['text'].str.strip() != '']

# Step 7: Save cleaned dataset
output_path = r"D:\Sparsh\ML_Projects\Fake_News_Detection\Dataset\final_news_dataset_cleaned.csv"
df.to_csv(output_path, index=False)

print(f"✅ Cleaned dataset saved successfully at:\n{output_path}")
print(f"📊 Final shape: {df.shape}")
print(df.head(5))
