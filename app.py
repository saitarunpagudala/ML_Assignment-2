import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

st.set_page_config(page_title="Breast Cancer ML App", layout="wide")

st.title("🧠 Breast Cancer Classification App")
st.markdown("""
Upload a **test CSV file** and select a trained model 
to evaluate tumor classification (Malignant vs Benign).
""")

uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

model_option = st.selectbox(
    "Choose Classification Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ],
    index=0
)

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    if "target" not in data.columns:
        st.error("Uploaded CSV must contain a 'target' column.")
    else:
        X = data.drop("target", axis=1)
        y = data["target"]

        model = joblib.load(f"model/{model_option}.pkl")
        y_pred = model.predict(X)

        st.markdown("## 📊 Model Evaluation Results")

        # --- Metrics Summary ---
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy:.2f}")
        col2.metric("Precision", f"{precision:.2f}")
        col3.metric("Recall", f"{recall:.2f}")
        col4.metric("F1 Score", f"{f1:.2f}")

        st.markdown("---")

        # --- Classification Report ---
        st.markdown("### 📄 Detailed Classification Report")
        st.text(classification_report(y, y_pred))

        st.markdown("---")

        # --- Confusion Matrix ---
        st.markdown("### 🔍 Confusion Matrix")

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            fig, ax = plt.subplots(figsize=(3,3))

            sns.heatmap(
                    confusion_matrix(y, y_pred),
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=False,  # 🔥 removes color bar (huge space saver)
                    annot_kws={"size": 9}
            )
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Actual", fontsize=8)

            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)

        st.markdown("---")

        # --- Explanation Section ---
        st.markdown("### ℹ️ Metric Explanation")

        st.write("""
        - **Accuracy**: Overall correctness of the model.
        - **Precision**: Out of predicted positives, how many are correct.
        - **Recall**: Out of actual positives, how many were detected.
        - **F1 Score**: Harmonic mean of precision and recall.
        - **Confusion Matrix**: Shows correct and incorrect predictions.
        """)
