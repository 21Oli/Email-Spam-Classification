# ==========================================================
# Performance Dashboard
# ==========================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(

    page_title="Performance",

    page_icon="📈",

    layout="wide"

)

# ==========================================================
# Metrics
# ==========================================================

accuracy = 0.9768
precision = 0.9821
recall = 0.9686
f1 = 0.9753

train_size = 155049
test_size = 38763
vocabulary = 10000

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("📈 Performance")

st.sidebar.markdown("---")

st.sidebar.success("""

Model Evaluation Dashboard

""")

st.sidebar.markdown("### Dataset")

st.sidebar.write(f"Training : {train_size:,}")

st.sidebar.write(f"Testing : {test_size:,}")

st.sidebar.write(f"Vocabulary : {vocabulary:,}")

st.sidebar.markdown("---")

# ==========================================================
# Header
# ==========================================================

st.title("📈 Model Performance Dashboard")

st.caption("""

Evaluation metrics for the trained
Perceptron classifier.

""")

st.divider()

# ==========================================================
# Metrics
# ==========================================================

st.header("🏆 Performance Metrics")

c1,c2,c3,c4 = st.columns(4)

with c1:

    st.metric(

        "Accuracy",

        f"{accuracy*100:.2f}%"

    )

with c2:

    st.metric(

        "Precision",

        f"{precision*100:.2f}%"

    )

with c3:

    st.metric(

        "Recall",

        f"{recall*100:.2f}%"

    )

with c4:

    st.metric(

        "F1 Score",

        f"{f1*100:.2f}%"

    )

st.divider()

# ==========================================================
# Performance Bar Chart
# ==========================================================

st.header("📊 Metric Comparison")

metrics = pd.DataFrame({

    "Metric":[

        "Accuracy",

        "Precision",

        "Recall",

        "F1"

    ],

    "Score":[

        accuracy,

        precision,

        recall,

        f1

    ]

})

fig, ax = plt.subplots(figsize=(8,4))

ax.bar(

    metrics["Metric"],

    metrics["Score"]

)

ax.set_ylim(0.9,1)

st.pyplot(fig)

st.divider()

# ==========================================================
# Classification Report
# ==========================================================

st.header("📋 Classification Report")

report_df = pd.DataFrame({

    "Class":[

        "Ham",

        "Spam",

        "Macro Avg",

        "Weighted Avg"

    ],

    "Precision":[

        0.97,

        0.98,

        0.98,

        0.98

    ],

    "Recall":[

        0.98,

        0.97,

        0.98,

        0.98

    ],

    "F1-Score":[

        0.98,

        0.98,

        0.98,

        0.98

    ]

})

st.dataframe(

    report_df,

    use_container_width=True,

    hide_index=True

)

st.divider()

# ==========================================================
# Confusion Matrix
# ==========================================================

st.header("📊 Confusion Matrix")

# Values computed from your evaluation
tn = 19987
fp = 442
fn = 576
tp = 17758

cm = [

    [tn, fp],

    [fn, tp]

]

fig, ax = plt.subplots(figsize=(5,5))

image = ax.imshow(cm)

ax.set_xticks([0,1])

ax.set_yticks([0,1])

ax.set_xticklabels(["Ham","Spam"])

ax.set_yticklabels(["Ham","Spam"])

ax.set_xlabel("Predicted")

ax.set_ylabel("Actual")

ax.set_title("Confusion Matrix")

for i in range(2):

    for j in range(2):

        ax.text(

            j,

            i,

            f"{cm[i][j]:,}",

            ha="center",

            va="center",

            fontsize=12,

            fontweight="bold"

        )

st.pyplot(fig)

st.divider()

# ==========================================================
# Dataset Overview
# ==========================================================

st.header("📚 Evaluation Dataset")

c1,c2,c3 = st.columns(3)

with c1:

    st.metric(

        "Training Samples",

        f"{train_size:,}"

    )

with c2:

    st.metric(

        "Testing Samples",

        f"{test_size:,}"

    )

with c3:

    st.metric(

        "Vocabulary",

        f"{vocabulary:,}"

    )

st.divider()

# ==========================================================
# Metric Interpretation
# ==========================================================

st.header("🧠 Performance Interpretation")

st.success("""

✅ Accuracy of **97.68%** indicates that the model correctly classified
almost all emails in the testing dataset.

""")

st.info("""

🎯 Precision of **98.21%** means that when the model predicts **Spam**,
it is usually correct.

""")

st.info("""

📬 Recall of **96.86%** shows that the model successfully detects
most Spam emails while missing very few.

""")

st.success("""

🏆 F1 Score of **97.53%** demonstrates an excellent balance between
Precision and Recall.

""")

st.warning("""

Although the model performs extremely well,
a few difficult emails near the decision boundary
may still be misclassified.

""")

st.divider()

# ==========================================================
# Overall Performance Summary
# ==========================================================

st.header("📈 Overall Summary")

summary = pd.DataFrame({

    "Metric":[

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "Training Samples",

        "Testing Samples",

        "Vocabulary Size"

    ],

    "Value":[

        f"{accuracy*100:.2f}%",

        f"{precision*100:.2f}%",

        f"{recall*100:.2f}%",

        f"{f1*100:.2f}%",

        f"{train_size:,}",

        f"{test_size:,}",

        f"{vocabulary:,}"

    ]

})

st.dataframe(

    summary,

    hide_index=True,

    use_container_width=True

)

st.divider()

# ==========================================================
# Footer
# ==========================================================

st.markdown("""

<div style="text-align:center">

## 📈 Model Performance Dashboard

Built with

**Python • Scikit-learn • TF-IDF • Perceptron • Streamlit**

Evaluation • NLP • Machine Learning

© 2026 Oli Bakala

</div>

""",

unsafe_allow_html=True)