import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer

class TrendPredictor:
    def __init__(self, model_path, vectorizer_path):
        self.model = load_model(model_path)
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

    def predict_engagement(self, hashtag, platform, post_type):
        input_text = f"{hashtag} {platform} {post_type}"
        vectorized_text = self.vectorizer.transform([input_text]).toarray()
        prediction = self.model.predict(vectorized_text)
        return prediction[0][0]

# Example usage
if __name__ == "__main__":
    predictor = TrendPredictor("models/lstm_model.h5", "models/vectorizer.pkl")
    hashtag = "#example"
    platform = "Twitter"
    post_type = "Text"
    engagement = predictor.predict_engagement(hashtag, platform, post_type)
    print(f"Predicted Engagement: {engagement:.2f}")