import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os

def preprocess_data(input_file, output_file, vectorizer_path):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Load data
    data = pd.read_csv(input_file)

    if "cleaned_text" not in data.columns:
        raise ValueError("Input file must contain a 'cleaned_text' column")

    texts = data["cleaned_text"]

    # Fit CountVectorizer
    vectorizer = CountVectorizer(max_features=5000)
    vectorized_data = vectorizer.fit_transform(texts)

    # Save processed data as a DataFrame
    processed_df = pd.DataFrame(vectorized_data.toarray(), columns=vectorizer.get_feature_names_out())
    processed_df["likes"] = data["likes"]  # Assuming "likes" column exists in input file
    processed_df.to_csv(output_file, index=False)

    # Save the fitted vectorizer
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    preprocess_data(
        input_file=os.path.join(BASE_DIR, "data", "raw", "raw_data.csv"),
        output_file=os.path.join(BASE_DIR, "data", "processed", "processed_data.csv"),
        vectorizer_path=os.path.join(BASE_DIR, "models", "vectorizer.pkl")
    )
