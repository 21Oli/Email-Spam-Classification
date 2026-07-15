"""
Visualization Utilities


This module contains reusable visualization functions
for the Email Spam Classification project.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ==========================================================
# Confidence Gauge
# ==========================================================

def plot_confidence_gauge(confidence):
    """
    Create a horizontal confidence gauge.

    Parameters
    ----------
    confidence : float
        Value between 0 and 100

    Returns
    -------
    matplotlib.figure.Figure
    """

    confidence = max(0, min(confidence, 100))

    fig, ax = plt.subplots(figsize=(8, 1.2))

    ax.barh(
        y=0,
        width=confidence,
        height=0.5
    )

    ax.set_xlim(0, 100)

    ax.set_xticks([0, 20, 40, 60, 80, 100])

    ax.set_yticks([])

    ax.set_xlabel("Confidence (%)")

    ax.set_title("Prediction Confidence")

    return fig


# ==========================================================
# Top Feature Contributions
# ==========================================================

def plot_feature_contributions(
        explanation_df,
        top_n=10):
    """
    Plot the top feature contributions.

    Parameters
    ----------
    explanation_df : DataFrame

    top_n : int

    Returns
    -------
    matplotlib.figure.Figure
    """

    if explanation_df.empty:

        return None

    df = explanation_df.copy()

    df = df.reindex(
        df["Contribution"].abs().sort_values(
            ascending=False
        ).index
    ).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [
        "green" if c > 0 else "red"
        for c in df["Contribution"]
    ]

    ax.barh(
        df["Word"],
        df["Contribution"],
        color=colors
    )

    ax.set_title("Top Influential Words")

    ax.set_xlabel("Contribution")

    ax.invert_yaxis()

    return fig


# ==========================================================
# Global Feature Importance
# ==========================================================

def plot_global_importance(
        spam_df,
        ham_df):
    """
    Plot global spam and ham words.

    Parameters
    ----------
    spam_df : DataFrame

    ham_df : DataFrame

    Returns
    -------
    tuple
        (spam_fig, ham_fig)
    """

    spam_fig, spam_ax = plt.subplots(figsize=(8,6))

    spam_ax.barh(
        spam_df["Word"],
        spam_df["Weight"]
    )

    spam_ax.set_title(
        "Top Spam Words"
    )

    spam_ax.invert_yaxis()


    ham_fig, ham_ax = plt.subplots(figsize=(8,6))

    ham_ax.barh(
        ham_df["Word"],
        ham_df["Weight"]
    )

    ham_ax.set_title(
        "Top Ham Words"
    )

    ham_ax.invert_yaxis()

    return spam_fig, ham_fig


# ==========================================================
# Prediction Distribution
# ==========================================================

def plot_prediction_distribution(history_df):
    """
    Plot prediction counts.

    Parameters
    ----------
    history_df : DataFrame

    Returns
    -------
    matplotlib.figure.Figure
    """

    if history_df.empty:

        return None

    counts = history_df["Prediction"].value_counts()

    fig, ax = plt.subplots(figsize=(5,5))

    ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90
    )

    ax.set_title(
        "Prediction Distribution"
    )

    return fig


# ==========================================================
# Decision Score Plot
# ==========================================================

def plot_decision_score(score):
    """
    Visualize the Perceptron decision score.

    Parameters
    ----------
    score : float

    Returns
    -------
    matplotlib.figure.Figure
    """

    fig, ax = plt.subplots(figsize=(8,1.4))

    ax.barh(
        y=0,
        width=score
    )

    ax.axvline(
        x=0,
        linestyle="--"
    )

    ax.set_yticks([])

    ax.set_xlabel(
        "Decision Score"
    )

    ax.set_title(
        "Perceptron Decision Function"
    )

    return fig


# ==========================================================
# Performance Metrics
# ==========================================================

def metrics_dataframe():
    """
    Return evaluation metrics.

    Returns
    -------
    pandas.DataFrame
    """

    return pd.DataFrame({

        "Metric":[

            "Accuracy",

            "Precision",

            "Recall",

            "F1 Score"

        ],

        "Value":[

            0.9768,

            0.9821,

            0.9686,

            0.9753

        ]

    })


# ==========================================================
# Confusion Matrix Data
# ==========================================================

def confusion_matrix_dataframe():
    """
    Returns confusion matrix values.

    Replace these values if you
    calculate your own confusion matrix.
    """

    return pd.DataFrame(

        [

            [20058, 371],

            [576, 17758]

        ],

        columns=["Predicted Ham","Predicted Spam"],

        index=["Actual Ham","Actual Spam"]

    )