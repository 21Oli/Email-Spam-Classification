# ==========================================================
# Model Analytics Dashboard
# ==========================================================

import numpy as np
import pandas as pd
import streamlit as st

from utils.model_loader import load_model
from utils.explainable import global_feature_importance

st.set_page_config(

    page_title="Model Analytics",

    page_icon="📊",

    layout="wide"

)

# ==========================================================
# Load Model
# ==========================================================

model, vectorizer = load_model()

feature_names = vectorizer.get_feature_names_out()

weights = model.coef_[0]

# ==========================================================
# Vocabulary
# ==========================================================

feature_types = [

    "Bigram"

    if " " in word

    else "Unigram"

    for word in feature_names

]

vocab_df = pd.DataFrame({

    "Word": feature_names,

    "Feature Type": feature_types,

    "Weight": weights

})

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("📊 Model Analytics")

st.sidebar.markdown("---")

st.sidebar.success("""

Inspect what the
Perceptron learned.

""")

st.sidebar.markdown("### Model")

st.sidebar.write("Algorithm : Perceptron")

st.sidebar.write("Vectorizer : TF-IDF")

st.sidebar.write("Features : 10,000")

st.sidebar.write("N-Grams : (1,2)")

st.sidebar.markdown("---")

# ==========================================================
# Header
# ==========================================================

st.title("📊 Model Analytics Dashboard")

st.caption("""

Understand what the
machine learning model
has learned from
the training data.

""")

st.divider()

# ==========================================================
# Model Overview
# ==========================================================

st.header("🧠 Model Overview")

total_features = len(vocab_df)

unigrams = (

    vocab_df["Feature Type"]

    ==

    "Unigram"

).sum()

bigrams = (

    vocab_df["Feature Type"]

    ==

    "Bigram"

).sum()

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(

        "Algorithm",

        "Perceptron"

    )

with c2:

    st.metric(

        "Vocabulary",

        f"{total_features:,}"

    )

with c3:

    st.metric(

        "Unigrams",

        f"{unigrams:,}"

    )

with c4:

    st.metric(

        "Bigrams",

        f"{bigrams:,}"

    )

st.divider()

# ==========================================================
# Weight Statistics
# ==========================================================

st.header("📈 Model Weight Statistics")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric(
        "Maximum",
        f"{weights.max():.4f}"
    )

with c2:
    st.metric(
        "Minimum",
        f"{weights.min():.4f}"
    )

with c3:
    st.metric(
        "Mean",
        f"{weights.mean():.4f}"
    )

with c4:
    st.metric(
        "Median",
        f"{np.median(weights):.4f}"
    )

with c5:
    st.metric(
        "Std Dev",
        f"{weights.std():.4f}"
    )

st.divider()

# ==========================================================
# Weight Distribution
# ==========================================================

st.subheader("📊 Distribution of Model Weights")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10,4))

ax.hist(
    weights,
    bins=50
)

ax.set_xlabel("Weight")

ax.set_ylabel("Number of Features")

ax.set_title("Perceptron Weight Distribution")

st.pyplot(fig)

st.divider()

# ==========================================================
# Global Feature Importance
# ==========================================================

spam_df, ham_df = global_feature_importance(

    model,

    vectorizer,

    top_n=20

)

left, right = st.columns(2)

# ----------------------------------------------------------
# Spam Features
# ----------------------------------------------------------

with left:

    st.subheader("🚨 Strongest Spam Features")

    st.dataframe(

        spam_df,

        use_container_width=True,

        hide_index=True

    )

# ----------------------------------------------------------
# Ham Features
# ----------------------------------------------------------

with right:

    st.subheader("✅ Strongest Ham Features")

    st.dataframe(

        ham_df,

        use_container_width=True,

        hide_index=True

    )

st.divider()

# ==========================================================
# Search Vocabulary
# ==========================================================

st.header("🔍 Search Learned Vocabulary")

keyword = st.text_input(

    "Search Feature",

    placeholder="Type a word..."

)

filtered_vocab = vocab_df.copy()

if keyword.strip():

    filtered_vocab = filtered_vocab[

        filtered_vocab["Word"]

        .str.contains(

            keyword,

            case=False,

            na=False

        )

    ]

st.dataframe(

    filtered_vocab,

    use_container_width=True,

    hide_index=True,

    height=400

)

st.divider()

# ==========================================================
# Download Vocabulary
# ==========================================================

csv = vocab_df.to_csv(index=False)

st.download_button(

    label="📥 Download Model Vocabulary",

    data=csv,

    file_name="model_vocabulary.csv",

    mime="text/csv"

)

# ==========================================================
# Feature Type Distribution
# ==========================================================

st.header("📚 Vocabulary Analysis")

left, right = st.columns(2)

with left:

    st.subheader("Feature Types")

    feature_counts = (

        vocab_df["Feature Type"]

        .value_counts()

    )

    fig, ax = plt.subplots(figsize=(5,5))

    ax.pie(

        feature_counts,

        labels=feature_counts.index,

        autopct="%1.1f%%",

        startangle=90

    )

    ax.axis("equal")

    st.pyplot(fig)

with right:

    st.subheader("Vocabulary Summary")

    summary = pd.DataFrame({

        "Metric":[

            "Total Features",

            "Unigrams",

            "Bigrams"

        ],

        "Value":[

            total_features,

            unigrams,

            bigrams

        ]

    })

    st.dataframe(

        summary,

        hide_index=True,

        use_container_width=True

    )

st.divider()

# ==========================================================
# Top Positive & Negative Weights
# ==========================================================

st.header("🔥 Strongest Learned Features")

left, right = st.columns(2)

with left:

    st.subheader("Top Spam Features")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.barh(

        spam_df["Word"],

        spam_df["Weight"]

    )

    ax.invert_yaxis()

    st.pyplot(fig)

with right:

    st.subheader("Top Ham Features")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.barh(

        ham_df["Word"],

        ham_df["Weight"]

    )

    ax.invert_yaxis()

    st.pyplot(fig)

st.divider()

# ==========================================================
# Scatter Plot
# ==========================================================

st.header("📈 Feature Weight Scatter Plot")

fig, ax = plt.subplots(figsize=(10,4))

ax.scatter(

    range(len(weights)),

    weights,

    s=8

)

ax.set_xlabel("Feature Index")

ax.set_ylabel("Weight")

ax.set_title("Perceptron Learned Weights")

st.pyplot(fig)

st.divider()

# ==========================================================
# Model Summary
# ==========================================================

st.header("🧠 Model Summary")

summary = pd.DataFrame({

    "Property":[

        "Classifier",

        "Vectorizer",

        "Vocabulary Size",

        "Classes",

        "Feature Types",

        "Maximum Weight",

        "Minimum Weight",

        "Average Weight"

    ],

    "Value":[

        "Perceptron",

        "TF-IDF",

        total_features,

        "Spam / Ham",

        "Unigram + Bigram",

        round(weights.max(),4),

        round(weights.min(),4),

        round(weights.mean(),4)

    ]

})

st.dataframe(

    summary,

    use_container_width=True,

    hide_index=True

)

st.divider()

# ==========================================================
# Footer
# ==========================================================

st.markdown("""

<div style='text-align:center;'>

## 📊 Model Analytics Dashboard

Built with

**Python • Scikit-learn • TF-IDF • Perceptron • Streamlit**

Explainable AI • NLP • Machine Learning

© 2026 Oli Bakala

</div>

""",

unsafe_allow_html=True)