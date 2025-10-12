import pandas as pd

# ✅ Use raw string for Windows path
file_path = r"D:\Sparsh\ML_Projects\Fake_News_Detection\Dataset\final_news_dataset.csv"

# Step 1: Read file safely using 'python' engine
df = pd.read_csv(file_path, engine="python", header=None, on_bad_lines=False)

# Step 2: Merge all extra columns into one text column (join with space)
df['merged'] = df.apply(lambda x: ' '.join(str(v) for v in x if pd.notna(v)), axis=1)

# Step 3: Try to extract label (last value like 0/1)
df['label'] = df['merged'].str.extract(r'(\b[01]\b)$')
df['label'] = df['label'].fillna(method='ffill').fillna(method='bfill')

# Step 4: Remove label number from text part
df['text'] = df['merged'].str.replace(r'\b[01]\b$', '', regex=True).str.strip()

# Step 5: Drop merged column, keep clean ones
df = df[['text', 'label']]

# Step 6: Clean missing rows
df.dropna(subset=['text', 'label'], inplace=True)
df = df[df['text'].str.strip() != '']

# Step 7: Save cleaned dataset
output_path = r"D:\Sparsh\ML_Projects\Fake_News_Detection\Dataset\final_news_dataset_cleaned.csv"
df.to_csv(output_path, index=False)

print(f"✅ Cleaned dataset saved at:\n{output_path}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
