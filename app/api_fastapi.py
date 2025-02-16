from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from app.utils import load_model_and_vectorizer

# Load model and vectorizer
model, vectorizer = load_model_and_vectorizer()

# Define FastAPI app
app = FastAPI()

# Input schema
class TrendInput(BaseModel):
    hashtag: str
    platform: str
    post_type: str

@app.post("/predict")
def predict(input: TrendInput):
    try:
        # Combine features for prediction
        input_text = f"{input.hashtag} {input.platform} {input.post_type}"
        # Vectorize input text
        vectorized_text = vectorizer.transform([input_text]).toarray()
        # Predict
        prediction = model.predict(vectorized_text)
        return {"predicted_engagement": float(prediction[0][0])}
    except Exception as e:
        return {"error": str(e)}