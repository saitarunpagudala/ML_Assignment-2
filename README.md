Breast Cancer Classification – ML Assignment 2

**Problem Statement**

The objective of this project is to build and compare multiple machine learning classification models to predict whether a breast tumor is Malignant (0) or Benign (1) using diagnostic medical features.

The project also includes deployment of an interactive Streamlit web application that allows users to upload test data and evaluate trained models.

**Dataset Description**

The dataset used is the Breast Cancer Wisconsin (Diagnostic) Dataset.

| Attribute       | Value                     |
| --------------- | ------------------------- |
| Total Instances | 569                       |
| Total Features  | 30 numerical features     |
| Target Classes  | 0 → Malignant, 1 → Benign |
| Problem Type    | Binary Classification     |

The dataset satisfies the assignment requirements of having more than 12 features and more than 500 samples.

The dataset was split into training and test sets, and feature scaling was applied where required before training the models.

**Models Implemented**

The following six classification models were trained on the same dataset:

| Model Type              | Model Name                |
| ----------------------- | ------------------------- |
| Linear Model            | Logistic Regression       |
| Tree-Based Model        | Decision Tree             |
| Distance-Based Model    | K-Nearest Neighbors (KNN) |
| Probabilistic Model     | Naive Bayes (Gaussian)    |
| Ensemble Model          | Random Forest             |
| Ensemble Boosting Model | XGBoost                   |

All models were trained using a consistent train-test split and evaluated on the test dataset.

**Evaluation Metrics**

Each model was evaluated using the following metrics:

Accuracy

AUC Score

Precision

Recall

F1 Score

Matthews Correlation Coefficient (MCC)

**Model Performance Comparison**
| Model               | Accuracy | AUC  | Precision | Recall | F1 Score | MCC  |
| ------------------- | -------- | ---- | --------- | ------ | -------- | ---- |
| Logistic Regression | 0.97     | 0.99 | 0.97      | 0.99   | 0.98     | 0.94 |
| Decision Tree       | 0.93     | 0.92 | 0.94      | 0.94   | 0.94     | 0.87 |
| KNN                 | 0.95     | 0.98 | 0.96      | 0.96   | 0.96     | 0.90 |
| Naive Bayes         | 0.96     | 0.97 | 0.96      | 0.99   | 0.97     | 0.92 |
| Random Forest       | 0.96     | 0.99 | 0.96      | 0.99   | 0.97     | 0.93 |
| XGBoost             | 0.96     | 0.99 | 0.96      | 0.97   | 0.97     | 0.93 |

**Observations on Model Performance**
| Model               | Observation                                                                            |
| ------------------- | -------------------------------------------------------------------------------------- |
| Logistic Regression | Performs strongly as a baseline model and achieves excellent recall and F1 score.      |
| Decision Tree       | Slightly lower performance compared to other models and may show signs of overfitting. |
| KNN                 | Performs well after feature scaling and maintains balanced precision and recall.       |
| Naive Bayes         | Performs competitively despite strong independence assumptions.                        |
| Random Forest       | Achieves stable and high performance due to ensemble averaging.                        |
| XGBoost             | Provides strong generalization and competitive performance using boosting techniques.  |

**Streamlit Web Application Features**

The deployed Streamlit application provides:

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

The application is deployed using Streamlit Community Cloud and connected to the GitHub repository.


** Project Structure**
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


