"""
Model Loader Utilities


This module loads the trained Machine Learning
model and TF-IDF vectorizer.

The model is cached so it is loaded only once,
making the Streamlit application much faster.
"""

import joblib
import streamlit as st
from pathlib import Path


# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "perceptron_model.pkl"

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"


# ==========================================================
# Load Model
# ==========================================================

@st.cache_resource
def load_model():
    """
    Load the trained Perceptron model and
    TF-IDF vectorizer.

    Returns
    -------
    tuple
        (model, tfidf_vectorizer)
    """

    try:

        model = joblib.load(MODEL_PATH)

        vectorizer = joblib.load(VECTORIZER_PATH)

        return model, vectorizer

    except FileNotFoundError as e:

        st.error("❌ Model files were not found.")

        st.info(
            """
            Make sure the following files exist:

            models/
            ├── perceptron_model.pkl
            └── tfidf_vectorizer.pkl
            """
        )

        st.stop()

    except Exception as e:

        st.error(f"❌ Error loading model: {e}")

        st.stop()


# ==========================================================
# Model Information
# ==========================================================

def get_model_info():
    """
    Returns metadata about the trained model.
    """

    return {

        "Algorithm": "Perceptron",

        "Vectorizer": "TF-IDF",

        "Features": "10,000",

        "N-Grams": "Unigrams + Bigrams",

        "Classes": {

            0: "Ham",

            1: "Spam"

        }

    }


# ==========================================================
# Prediction Helper
# ==========================================================

def predict_email(email_text):
    """
    Predict whether an email is Spam or Ham.

    Parameters
    ----------
    email_text : str
        Preprocessed email text.

    Returns
    -------
    dict
        Prediction result.
    """

    model, vectorizer = load_model()

    vector = vectorizer.transform([email_text])

    prediction = model.predict(vector)[0]

    score = model.decision_function(vector)[0]

    label = "Spam" if prediction == 1 else "Ham"

    return {

        "prediction": prediction,

        "label": label,

        "decision_score": float(score),

        "vector": vector,

        "model": model,

        "vectorizer": vectorizer

    }


# ==========================================================
# Confidence Score
# ==========================================================

def calculate_confidence(decision_score):
    """
    Convert decision score into an approximate
    confidence percentage.

    Note:
    Perceptron does not provide probabilities.
    This value is only for visualization.
    """

    confidence = min(abs(decision_score) / 5, 1.0)

    return round(confidence * 100, 2)