🧠 Breast Cancer Classification – ML Assignment 2
📌 Problem Statement

Build and compare multiple machine learning classification models to predict whether a breast tumor is Malignant (0) or Benign (1) using diagnostic features.

The project also includes deployment of an interactive Streamlit web application for model evaluation.

📊 Dataset

Breast Cancer Wisconsin (Diagnostic) Dataset

Instances: 569

Features: 30 numerical features

Target Variable:

0 → Malignant

1 → Benign

The dataset satisfies assignment constraints (≥12 features and ≥500 samples).

🤖 Models Implemented

The following six classification models were trained and saved as .pkl files:

Logistic Regression

Decision Tree

K-Nearest Neighbors (KNN)

Naive Bayes (Gaussian)

Random Forest

XGBoost

All models were trained using a train-test split and evaluated on the test set.

📈 Evaluation Metrics

Each model was evaluated using:

Accuracy

Precision

Recall

F1 Score

Classification Report

Confusion Matrix

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 0.97     | 0.97      | 0.99   | 0.98     |
| Decision Tree       | 0.93     | 0.94      | 0.94   | 0.94     |
| KNN                 | 0.95     | 0.96      | 0.96   | 0.96     |
| Naive Bayes         | 0.96     | 0.96      | 0.99   | 0.97     |
| Random Forest       | 0.96     | 0.96      | 0.99   | 0.97     |
| XGBoost             | 0.96     | 0.96      | 0.97   | 0.97     |


(Results may vary slightly depending on random state.)

🔍 Observations

Logistic Regression performs strongly as a baseline model.

Decision Tree shows slightly lower generalization performance.

KNN performs well after feature scaling.

Naive Bayes performs competitively despite independence assumptions.

Random Forest and XGBoost provide consistently high performance due to ensemble learning.

Overall, ensemble methods achieve strong stability and accuracy.

🌐 Streamlit Web Application Features

The deployed Streamlit app provides:

CSV test dataset upload

Downloadable sample test dataset

Model selection dropdown

Evaluate Model button

Display of:

Accuracy

Precision

Recall

F1 Score

Detailed Classification Report

Confusion Matrix

The results update only when the Evaluate Model button is clicked.

📁 Project Structure
│-- app.py
│-- requirements.txt
│-- README.md
│-- test_data.csv
│-- model/
    │-- train_models.py
    │-- scaler.pkl
    │-- Logistic Regression.pkl
    │-- Decision Tree.pkl
    │-- KNN.pkl
    │-- Naive Bayes.pkl
    │-- Random Forest.pkl
    │-- XGBoost.pkl

🚀 Deployment

The application is deployed using Streamlit Community Cloud, connected to the GitHub repository.
