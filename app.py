import streamlit as st
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="TruthScan AI - Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# Load trained model and vectorizer
@st.cache_resource
def load_models():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

try:
    model, vectorizer = load_models()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

# App heading
st.title("🔍 TruthScan AI")
st.subheader("Fake News Detection System")

st.write(
    "Check whether a news article is REAL or FAKE "
    "using Machine Learning."
)

st.divider()

# User input
news_text = st.text_area(
    "📰 Enter your news article here:",
    height=200,
    placeholder="Paste your news article here..."
)

# Prediction
if st.button("Check News", use_container_width=True):

    if not news_text.strip():
        st.warning("Please enter a news article first.")

    else:
        try:
            # Convert text into numerical features
            news_vector = vectorizer.transform([news_text])

            # Predict news class
            prediction = model.predict(news_vector)[0]

            st.subheader("Prediction Result")

            # Assumption: 0 = Fake, 1 = Real
            if prediction == 0:
                st.error("🚨 FAKE NEWS DETECTED")
                st.write(
                    "The model classified this article as Fake News."
                )

            elif prediction == 1:
                st.success("✅ REAL NEWS")
                st.write(
                    "The model classified this article as Real News."
                )

            else:
                st.info(f"Predicted class: {prediction}")

            # Display confidence if supported
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(news_vector)[0]
                confidence = max(probabilities) * 100

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

        except Exception as e:
            st.error(f"Prediction error: {e}")

st.divider()

st.caption(
    "TruthScan AI | Machine Learning Based Fake News Detection"
)
