import pandas as pd
import re
import os

# Load the data
data = pd.read_csv("data/social_media_trends.csv")

# Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and punctuation
    text = re.sub(r'\W', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Combine columns for text representation
data['text'] = data['Hashtag'] + " " + data['Platform'] + " " + data['Post_Type']
data['clean_text'] = data['text'].apply(preprocess_text)

# Save preprocessed data
os.makedirs("data/processed", exist_ok=True)
data.to_csv("data/processed/clean_social_media_trends.csv", index=False)