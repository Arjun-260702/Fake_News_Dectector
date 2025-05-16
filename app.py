import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📰 Fake News Detection App")

news_input = st.text_area("Enter a news article to check if it's real or fake:")

if st.button("Check"):
    if news_input.strip() == "":
        st.warning("Please enter some news content.")
    else:
        # Transform and predict
        vectorized_input = vectorizer.transform([news_input])
        prediction = model.predict(vectorized_input)

        result = "🟢 Real News" if prediction[0] == 1 else "🔴 Fake News"
        st.success(f"Prediction: {result}")
