🧠 Breast Cancer Classification – ML Assignment 2
📌 Problem Statement

Build and compare multiple machine learning classification models to predict whether a breast tumor is Malignant or Benign using diagnostic features.
The project also includes deployment of an interactive Streamlit web application.

📊 Dataset

Breast Cancer Wisconsin (Diagnostic) Dataset

Instances: 569

Features: 30 numerical features

Target:

0 → Malignant

1 → Benign

The dataset satisfies assignment constraints (≥12 features, ≥500 samples).

🤖 Models Implemented

Logistic Regression

Decision Tree

K-Nearest Neighbors (KNN)

Naive Bayes (Gaussian)

Random Forest (Ensemble)

XGBoost (Ensemble)

📈 Evaluation Metrics

Each model was evaluated using:

Accuracy

AUC Score

Precision

Recall

F1 Score

Matthews Correlation Coefficient (MCC)

📊 Model Comparison
Model	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.96	0.99	0.97	0.97	0.97	0.92
Decision Tree	0.93	0.92	0.94	0.95	0.94	0.87
KNN	0.95	0.98	0.96	0.96	0.96	0.90
Naive Bayes	0.93	0.97	0.94	0.94	0.94	0.86
Random Forest	0.97	0.99	0.98	0.98	0.98	0.94
XGBoost	0.97	0.99	0.98	0.98	0.98	0.95

(Results may vary slightly depending on random state.)

🔍 Observations

Logistic Regression performs strongly as a baseline model.

Decision Tree shows slight overfitting.

KNN performs well after feature scaling.

Naive Bayes performs reasonably but assumes feature independence.

Random Forest and XGBoost achieve the best overall performance due to ensemble learning.

🌐 Streamlit App Features

CSV dataset upload

Model selection dropdown

Display of evaluation metrics

Confusion matrix & classification report

📁 Project Structure
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
    │-- train_models.py
    │-- *.pkl files
