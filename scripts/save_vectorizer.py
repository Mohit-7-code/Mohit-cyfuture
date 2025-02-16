import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Load data
data = pd.read_csv("data/processed/clean_social_media_trends.csv")

# Create and fit the vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(data['clean_text'])

# Save the vectorizer
VECTOR_PATH = "models/vectorizer.pkl"
os.makedirs("models", exist_ok=True)
with open(VECTOR_PATH, "wb") as f:
    pickle.dump(vectorizer, f)

print(f"Vectorizer saved successfully at: {VECTOR_PATH}")
