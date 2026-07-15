# ==========================================================
# About the Project
# ==========================================================

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("ℹ️ About")

st.sidebar.markdown("---")

st.sidebar.success(
    """
    Email Spam Classification
    using Machine Learning
    """
)

st.sidebar.markdown("### Application")

st.sidebar.write("Version : 1.0")

st.sidebar.write("Model : Perceptron")

st.sidebar.write("Vectorizer : TF-IDF")

st.sidebar.markdown("---")

# ==========================================================
# Header
# ==========================================================

st.title("ℹ️ About This Project")

st.caption(
    """
    A complete Machine Learning application for
    Email Spam Detection using the Perceptron Algorithm.
    """
)

st.divider()

# ==========================================================
# Project Overview
# ==========================================================

st.header("📌 Project Overview")

st.write(
"""
This application classifies email messages into **Spam** or **Ham**
using Natural Language Processing (NLP) and Machine Learning.

The project demonstrates an end-to-end machine learning workflow,
including data preprocessing, feature engineering using TF-IDF,
model training with the Perceptron algorithm, evaluation,
Explainable AI, and deployment through Streamlit.

The goal is to provide accurate and interpretable spam detection
while showcasing the complete machine learning pipeline.
"""
)

st.divider()

# ==========================================================
# Application Features
# ==========================================================

st.header("🚀 Application Features")

features = pd.DataFrame({

    "Feature":[

        "📧 Single Email Prediction",

        "📂 Batch Prediction",

        "🧠 Explainable AI",

        "📊 Model Analytics",

        "📈 Performance Dashboard",

        "📥 CSV Export",

        "🔍 Vocabulary Search",

        "📋 Interactive Tables"

    ],

    "Description":[

        "Predict Spam or Ham for one email",

        "Predict hundreds or thousands of emails",

        "Explain why the model made a prediction",

        "Explore learned feature weights",

        "View evaluation metrics",

        "Download prediction results",

        "Search learned vocabulary",

        "Interactive data exploration"

    ]

})

st.dataframe(

    features,

    use_container_width=True,

    hide_index=True

)

st.divider()

# ==========================================================
# Technologies
# ==========================================================

st.header("🛠️ Technologies Used")

tech = pd.DataFrame({

    "Technology":[

        "Python",

        "Streamlit",

        "Scikit-learn",

        "Pandas",

        "NumPy",

        "Matplotlib",

        "Joblib"

    ],

    "Purpose":[

        "Programming Language",

        "Web Application",

        "Machine Learning",

        "Data Analysis",

        "Numerical Computing",

        "Visualization",

        "Model Serialization"

    ]

})

st.dataframe(

    tech,

    use_container_width=True,

    hide_index=True

)

st.divider()

# ==========================================================
# Machine Learning Pipeline
# ==========================================================

st.header("🧠 Machine Learning Pipeline")

pipeline = pd.DataFrame({

    "Step":[

        "1",

        "2",

        "3",

        "4",

        "5",

        "6",

        "7",

        "8"

    ],

    "Process":[

        "Load Dataset",

        "Exploratory Data Analysis (EDA)",

        "Text Preprocessing",

        "TF-IDF Vectorization",

        "Train-Test Split",

        "Train Perceptron Model",

        "Model Evaluation",

        "Deploy with Streamlit"

    ],

    "Description":[

        "Load the Spam/Ham email dataset.",

        "Explore class distribution and text statistics.",

        "Clean text by removing punctuation, stopwords, numbers, and applying stemming.",

        "Convert cleaned emails into numerical TF-IDF features using unigrams and bigrams.",

        "Split the dataset into training and testing sets.",

        "Train the Perceptron classifier on the TF-IDF vectors.",

        "Evaluate the model using Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.",

        "Build an interactive web application for prediction and analysis."

    ]

})

st.dataframe(

    pipeline,

    use_container_width=True,

    hide_index=True

)

st.divider()

# ==========================================================
# Dataset Information
# ==========================================================

st.header("📚 Dataset Information")

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(

        "Total Emails",

        "193,850"

    )

with c2:

    st.metric(

        "Ham Emails",

        "102,159"

    )

with c3:

    st.metric(

        "Spam Emails",

        "91,691"

    )

with c4:

    st.metric(

        "Vocabulary",

        "10,000"

    )

st.info("""

**Dataset Summary**

• Total Emails: **193,850**

• Classes: **Ham (0)** and **Spam (1)**

• Feature Extraction: **TF-IDF**

• N-Grams: **(1,2)**

• Train/Test Split: **80% / 20%**

""")

st.divider()

# ==========================================================
# Model Information
# ==========================================================

st.header("🤖 Model Information")

model_info = pd.DataFrame({

    "Property":[

        "Algorithm",

        "Learning Method",

        "Vectorizer",

        "Features",

        "Classification",

        "Output",

        "Deployment"

    ],

    "Value":[

        "Perceptron",

        "Supervised Learning",

        "TF-IDF",

        "10,000",

        "Binary",

        "Spam / Ham",

        "Streamlit"

    ]

})

st.dataframe(

    model_info,

    use_container_width=True,

    hide_index=True

)

st.divider()

# ==========================================================
# Project Structure
# ==========================================================

st.header("📂 Project Structure")

st.code("""

Email-Spam-Classifier/

│

├── app.py

├── pages/

│   ├── 📧_Single_Prediction.py

│   ├── 📂_Batch_Prediction.py

│   ├── 📊_Model_Analytics.py

│   ├── 📈_Performance.py

│   └── ℹ️_About.py

│

├── utils/

│   ├── model_loader.py

│   ├── explain.py

│   ├── visualization.py

│

├── models/

│   ├── perceptron_model.pkl

│   └── tfidf_vectorizer.pkl

│

├── results/

│   ├── metrics.json

│   ├── classification_report.csv

│   ├── confusion_matrix.npy

│   └── test_predictions.csv

│

├── requirements.txt

├── README.md

""",

language="text")

st.divider()

# ==========================================================
# Model Performance Summary
# ==========================================================

st.header("🏆 Model Performance")

performance = pd.DataFrame({

    "Metric":[

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score"

    ],

    "Value":[

        "97.68%",

        "98.21%",

        "96.86%",

        "97.53%"

    ]

})

st.dataframe(

    performance,

    use_container_width=True,

    hide_index=True

)

st.success(
"""
The Perceptron classifier achieved excellent performance on the testing
dataset. The high Accuracy, Precision, Recall, and F1 Score indicate
that the model is effective at distinguishing Spam from Ham emails.
"""
)

st.divider()

# ==========================================================
# Future Improvements
# ==========================================================

st.header("🎯 Future Improvements")

future = pd.DataFrame({

    "Future Enhancement":[

        "Compare multiple Machine Learning algorithms",

        "Deep Learning using LSTM or Transformers",

        "Multilingual Spam Detection",

        "Real-time Email Classification API",

        "Continuous Model Retraining",

        "Cloud Deployment"

    ],

    "Description":[

        "Evaluate Logistic Regression, SVM, Naive Bayes, and Random Forest.",

        "Investigate deep learning models for improved text understanding.",

        "Support spam detection in multiple languages.",

        "Develop a REST API for real-time predictions.",

        "Automatically update the model using newly labeled data.",

        "Deploy using Streamlit Community Cloud or Docker."

    ]

})

st.dataframe(

    future,

    use_container_width=True,

    hide_index=True

)

st.divider()

# ==========================================================
# Author
# ==========================================================

st.header("👨‍💻 Author")

left, right = st.columns([1,2])

with left:

    st.metric(

        "Developer",

        "Oli Bakala"

    )

with right:

    st.write("""
**Project**

Email Spam Classification Using Perceptron

**Program**

Data Science and Artificial Intelligence

**Application**

Interactive Machine Learning Dashboard built with Streamlit.

This project demonstrates an end-to-end machine learning workflow,
including data preprocessing, feature engineering, model training,
evaluation, Explainable AI, analytics, and deployment.
""")

st.divider()

# ==========================================================
# Footer
# ==========================================================

st.markdown(
"""
<div style="text-align:center; padding:15px;">

<h3>📧 Email Spam Classification Using Perceptron</h3>

<p>
Built with
<strong>Python • Scikit-learn • Streamlit • TF-IDF • Perceptron</strong>
</p>

<p>
Machine Learning • Natural Language Processing • Explainable AI
</p>

<hr>

<p>
© 2026 Oli Bakala
</p>

</div>
""",
unsafe_allow_html=True
)
