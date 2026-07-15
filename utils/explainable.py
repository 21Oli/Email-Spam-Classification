"""
============================================================
Explainable AI Utilities
Author: Oli Bekele

This module provides Explainable AI (XAI) utilities for the
Email Spam Classification project.

It explains Perceptron predictions by calculating how much
each TF-IDF feature contributed to the final decision.

Contribution Formula
--------------------
Contribution = TF-IDF × Model Weight

Positive Contribution  -> Pushes prediction toward Spam

Negative Contribution  -> Pushes prediction toward Ham
============================================================
"""

import pandas as pd
import numpy as np


# ==========================================================
# Explain Prediction
# ==========================================================

def explain_prediction(model, vectorizer, vector):
    """
    Explain a prediction by calculating feature contributions.

    Parameters
    ----------
    model : sklearn.linear_model.Perceptron
        Trained Perceptron model.

    vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
        Trained TF-IDF vectorizer.

    vector : scipy sparse matrix
        TF-IDF vector of a single email.

    Returns
    -------
    pandas.DataFrame

    Columns
    -------
    Word
    Feature Type
    TF-IDF
    Weight
    Contribution
    Direction
    """

    feature_names = vectorizer.get_feature_names_out()

    tfidf_values = vector.toarray()[0]

    weights = model.coef_[0]

    indices = np.where(tfidf_values > 0)[0]

    rows = []

    for idx in indices:

        word = feature_names[idx]

        tfidf = tfidf_values[idx]

        weight = weights[idx]

        contribution = tfidf * weight

        # ----------------------------------------------
        # Feature Type
        # ----------------------------------------------

        feature_type = (
            "Bigram"
            if " " in word
            else "Unigram"
        )

        # ----------------------------------------------
        # Prediction Direction
        # ----------------------------------------------

        if contribution > 0:

            direction = "Spam"

        elif contribution < 0:

            direction = "Ham"

        else:

            direction = "Neutral"

        rows.append({

            "Word": word,

            "Feature Type": feature_type,

            "TF-IDF": round(tfidf, 4),

            "Weight": round(weight, 4),

            "Contribution": round(contribution, 4),

            "Direction": direction

        })

    df = pd.DataFrame(rows)

    if df.empty:

        return df

    # Sort by absolute contribution

    df = df.sort_values(

        by="Contribution",

        key=np.abs,

        ascending=False

    ).reset_index(drop=True)

    return df


# ==========================================================
# Top Spam Words
# ==========================================================

def top_spam_words(df, top_n=10):
    """
    Return the words that contributed the most
    toward a Spam prediction.

    Parameters
    ----------
    df : pandas.DataFrame

    top_n : int

    Returns
    -------
    pandas.DataFrame
    """

    if df.empty:

        return df

    spam_df = df[

        df["Contribution"] > 0

    ].copy()

    spam_df = spam_df.sort_values(

        by="Contribution",

        ascending=False

    )

    return spam_df.head(top_n)


# ==========================================================
# Top Ham Words
# ==========================================================

def top_ham_words(df, top_n=10):
    """
    Return the words that contributed the most
    toward a Ham prediction.

    Parameters
    ----------
    df : pandas.DataFrame

    top_n : int

    Returns
    -------
    pandas.DataFrame
    """

    if df.empty:

        return df

    ham_df = df[

        df["Contribution"] < 0

    ].copy()

    ham_df = ham_df.sort_values(

        by="Contribution",

        ascending=True

    )

    return ham_df.head(top_n)

# ==========================================================
# Prediction Summary
# ==========================================================

def prediction_summary(df):
    """
    Generate a summary of the feature contributions.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    dict
        Dictionary containing summary statistics.
    """

    if df.empty:

        return {

            "Total Features": 0,

            "Spam Features": 0,

            "Ham Features": 0,

            "Unigrams": 0,

            "Bigrams": 0

        }

    spam_features = (df["Contribution"] > 0).sum()

    ham_features = (df["Contribution"] < 0).sum()

    unigrams = (df["Feature Type"] == "Unigram").sum()

    bigrams = (df["Feature Type"] == "Bigram").sum()

    return {

        "Total Features": len(df),

        "Spam Features": spam_features,

        "Ham Features": ham_features,

        "Unigrams": unigrams,

        "Bigrams": bigrams

    }


# ==========================================================
# Global Feature Importance
# ==========================================================

def global_feature_importance(model, vectorizer, top_n=20):
    """
    Return the globally most influential words learned
    by the Perceptron model.

    Parameters
    ----------
    model : trained Perceptron

    vectorizer : trained TF-IDF Vectorizer

    top_n : int

    Returns
    -------
    tuple
        (top_spam_words, top_ham_words)
    """

    feature_names = vectorizer.get_feature_names_out()

    weights = model.coef_[0]

    feature_types = [

        "Bigram" if " " in word else "Unigram"

        for word in feature_names

    ]

    importance = pd.DataFrame({

        "Word": feature_names,

        "Feature Type": feature_types,

        "Weight": weights

    })

    spam = (

        importance

        .sort_values(

            by="Weight",

            ascending=False

        )

        .head(top_n)

        .reset_index(drop=True)

    )

    ham = (

        importance

        .sort_values(

            by="Weight",

            ascending=True

        )

        .head(top_n)

        .reset_index(drop=True)

    )

    return spam, ham


# ==========================================================
# Decision Explanation
# ==========================================================

def explain_decision(score):
    """
    Convert a Perceptron decision score into
    an easy-to-understand explanation.

    Parameters
    ----------
    score : float

    Returns
    -------
    dict
    """

    abs_score = abs(score)

    if abs_score >= 5:

        confidence = "Very High"

    elif abs_score >= 3:

        confidence = "High"

    elif abs_score >= 1:

        confidence = "Moderate"

    else:

        confidence = "Low"

    prediction = "Spam" if score > 0 else "Ham"

    if abs_score >= 5:

        interpretation = (
            "The model is extremely confident in this prediction."
        )

    elif abs_score >= 3:

        interpretation = (
            "The prediction is made with high confidence."
        )

    elif abs_score >= 1:

        interpretation = (
            "The prediction is reasonably confident."
        )

    else:

        interpretation = (
            "The email is close to the decision boundary and "
            "may be difficult to classify."
        )

    return {

        "Prediction": prediction,

        "Confidence": confidence,

        "Score": round(score, 4),

        "Interpretation": interpretation

    }