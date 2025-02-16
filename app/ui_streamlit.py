import streamlit as st
from app.utils import load_model_and_vectorizer

# Load model and vectorizer
model, vectorizer = load_model_and_vectorizer()

# Predefined weights for hashtags, platforms, and post types
HASHTAG_WEIGHTS = {
    "#love": 1.2,
    "#food": 1.1,
    "#travel": 1.3,
    "#fitness": 1.4,
    "#mohit": 0.9,  # Example custom weight
    "default": 1.0
}

PLATFORM_WEIGHTS = {
    "Twitter": 0.8,
    "Instagram": 1.5,
    "Facebook": 1.2
}

POST_TYPE_WEIGHTS = {
    "Image": 1.1,
    "Video": 1.5,
    "Text": 0.9
}

# Streamlit UI
st.title("Social Media Trend Predictor")

hashtag = st.text_input("Enter Hashtag:").strip().lower()
platform = st.selectbox("Select Platform:", ["Twitter", "Instagram", "Facebook"])
post_type = st.selectbox("Select Post Type:", ["Image", "Video", "Text"])

if st.button("Predict Engagement"):
    # Combine inputs into a single string
    input_text = f"{hashtag} {platform} {post_type}"
    
    # Debugging Outputs
    st.write("Debugging Info:")
    st.write("Hashtag:", hashtag)
    st.write("Platform:", platform)
    st.write("Post Type:", post_type)
    st.write("Combined Input:", input_text)
    
    try:
        # Vectorize input text
        vectorized_text = vectorizer.transform([input_text]).toarray()
        st.write("Vectorized Input:", vectorized_text)
        
        # Base Prediction from Model
        base_prediction = model.predict(vectorized_text)[0][0]
        
        # Adjust prediction using weights
        hashtag_weight = HASHTAG_WEIGHTS.get(hashtag, HASHTAG_WEIGHTS["default"])
        platform_weight = PLATFORM_WEIGHTS[platform]
        post_type_weight = POST_TYPE_WEIGHTS[post_type]
        
        adjusted_prediction = base_prediction * hashtag_weight * platform_weight * post_type_weight
        
        # Display debugging information for weights
        st.write("Hashtag Weight:", hashtag_weight)
        st.write("Platform Weight:", platform_weight)
        st.write("Post Type Weight:", post_type_weight)
        st.write("Base Prediction:", base_prediction)
        
        # Final Prediction
        st.success(f"Predicted Engagement: {adjusted_prediction:.2f}")
        
    except Exception as e:
        st.error(f"Error: {e}")