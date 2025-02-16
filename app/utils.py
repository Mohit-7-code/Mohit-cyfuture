import os
import pickle
from tensorflow.keras.models import load_model

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lstm_model.h5")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

from tensorflow.keras.losses import MeanSquaredError

def load_model_and_vectorizer():
    # Load vectorizer
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    # Load model with custom objects
    model = load_model(MODEL_PATH, custom_objects={"mse": MeanSquaredError()})

    return model, vectorizer