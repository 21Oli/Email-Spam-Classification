import streamlit as st

# ==========================================================
# Page Configuration
# ==========================================================
st.set_page_config(
    page_title="Email Spam Classification",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# Custom CSS
# ==========================================================
st.markdown("""
<style>

.main-title{
    font-size:42px;
    font-weight:700;
    text-align:center;
    color:#2E86C1;
}

.subtitle{
    text-align:center;
    font-size:20px;
    color:gray;
    margin-bottom:25px;
}

.metric-card{
    background-color:#F8F9FA;
    padding:20px;
    border-radius:12px;
    text-align:center;
    border:1px solid #E5E5E5;
}

.feature-box{
    padding:18px;
    border-radius:10px;
    border:1px solid #DDDDDD;
    background:#FAFAFA;
    margin-bottom:15px;
}

.footer{
    text-align:center;
    color:gray;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# Sidebar
# ==========================================================
st.sidebar.title("📌 Navigation")

st.sidebar.success(
    """
Choose a page from the sidebar.

📧 Single Prediction

📂 Batch Prediction

📊 Model Analytics

📈 Performance

ℹ About
"""
)

st.sidebar.markdown("---")

st.sidebar.info("""
### Machine Learning Model

Algorithm:
**Perceptron**

Feature Engineering:
**TF-IDF**

Features:
**10,000**

N-Grams:
**1–2**
""")

# ==========================================================
# Header
# ==========================================================
st.markdown(
    "<h1 class='main-title'>📧 Email Spam Classification System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>Machine Learning Powered Email Spam Detection using Perceptron + TF-IDF</p>",
    unsafe_allow_html=True
)

st.divider()

# ==========================================================
# Dashboard Metrics
# ==========================================================
st.subheader("📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Accuracy",
        "97.68%"
    )

with col2:
    st.metric(
        "Precision",
        "98.21%"
    )

with col3:
    st.metric(
        "Recall",
        "96.86%"
    )

with col4:
    st.metric(
        "F1 Score",
        "97.53%"
    )

st.divider()

# ==========================================================
# Dataset Information
# ==========================================================
left, right = st.columns(2)

with left:

    st.subheader("📊 Dataset Information")

    st.markdown("""
- Total Emails: **193,812**

- Ham Emails: **102,159**

- Spam Emails: **91,653**

- Features Used:
    - TF-IDF
    - 10,000 Features
    - Unigrams + Bigrams
""")

with right:

    st.subheader("🤖 Model Information")

    st.markdown("""
- Algorithm:
    - Perceptron

- Text Processing:
    - Lowercase
    - Remove URLs
    - Remove HTML
    - Remove Stopwords
    - Lemmatization

- Feature Extraction:
    - TF-IDF Vectorizer
""")

st.divider()

# ==========================================================
# Features
# ==========================================================

st.subheader("✨ Application Features")

c1, c2 = st.columns(2)

with c1:

    with st.container(border=True):
        st.markdown("### 📧 Single Email Prediction")
        st.write("Predict whether an email is **Spam** or **Ham** from a single email message.")

    with st.container(border=True):
        st.markdown("### 📂 Batch Prediction")
        st.write("Upload a CSV file and classify hundreds or thousands of emails at once.")

    with st.container(border=True):
        st.markdown("### 📊 Model Analytics")
        st.write("Explore the learned vocabulary, feature importance, and influential words.")

with c2:

    with st.container(border=True):
        st.markdown("### 📈 Performance Dashboard")
        st.write("View the confusion matrix, classification report, and evaluation metrics.")

    with st.container(border=True):
        st.markdown("### 🧠 Explainable AI")
        st.write("Understand why the model predicted an email as **Spam** or **Ham**.")

    with st.container(border=True):
        st.markdown("### ☁️ Deployment Ready")
        st.write("Built with Streamlit and ready for deployment to Streamlit Community Cloud.")

st.divider()

# ==========================================================
# Workflow
# ==========================================================
st.subheader("⚙ Machine Learning Pipeline")

st.markdown("""

Email Input

⬇

Text Preprocessing

⬇

TF-IDF Vectorization

⬇

Perceptron Classification

⬇

Spam / Ham Prediction

""")

st.divider()

# ==========================================================
# Getting Started
# ==========================================================
st.subheader("🚀 Getting Started")

st.info("""
Use the navigation menu on the left.

1. 📧 Single Prediction
    - Predict one email.

2. 📂 Batch Prediction
    - Upload a CSV file.

3. 📊 Model Analytics
    - Explore learned features.

4. 📈 Performance
    - View model evaluation.

5. ℹ About
    - Learn more about the project.
""")

# ==========================================================
# Footer
# ==========================================================
st.divider()

st.markdown(
"""
<div class="footer">

Developed by <b>Oli Bakala</b>

Machine Learning | Natural Language Processing | Streamlit

</div>
""",
unsafe_allow_html=True
)