import pandas as pd

# Loading both datasets
fake = pd.read_csv("D:\Sparsh\ML_Projects\Fake_News_Detection\Dataset\Fake.csv")
true = pd.read_csv("D:\Sparsh\ML_Projects\Fake_News_Detection\Dataset\True.csv")

# Add labels 
# True as 1 
# False as 0 

fake["label"] = 0
true["label"] = 1 

# Taking 10000 random rows from both dataset
fake_sample = fake.sample(10000, random_state=42)
true_sample = true.sample(10000, random_state=42)