
# ============================================================
# DIABETES PREDICTION USING MACHINE LEARNING
# MODEL COMPARISON WITH IQR-FIRST PREPROCESSING
# ============================================================
#
# Research objective:
# Compare five machine learning algorithms using the
# improved preprocessing order identified in Experiment B.
#
# Preprocessing:
# 1. IQR outlier clipping
# 2. Median imputation
# 3. Standard scaling
# 4. Classification model
#
# Models:
# 1. Logistic Regression
# 2. Decision Tree
# 3. Random Forest
# 4. K-Nearest Neighbors
# 5. Support Vector Machine
#
# Evaluation:
# Stratified 5-fold cross-validation
#
# IMPORTANT:
# This is an experimental script.
# It does not overwrite the existing trained model.
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import pandas as pd
import numpy as np

from pathlib import Path

# Pipeline connects preprocessing and classification steps.
from sklearn.pipeline import Pipeline

# Tools for reproducible stratified cross-validation.
from sklearn.model_selection import StratifiedKFold, cross_validate

# Base classes for creating a custom sklearn transformer.
from sklearn.base import BaseEstimator, TransformerMixin

# Preprocessing tools.
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Five classification algorithms.
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Evaluation metrics.
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score
)


# ------------------------------------------------------------
# 2. LOAD THE DATASET
# ------------------------------------------------------------

# Load the original diabetes dataset.
df = pd.read_csv("data/diabetes.csv")

# Separate the eight input features from the target.
X = df.drop("Outcome", axis=1)

# Store the target class.
# 0 = negative class; 1 = positive class.
y = df["Outcome"]


# ------------------------------------------------------------
# 3. REPLACE INVALID ZERO VALUES
# ------------------------------------------------------------

# These measurements use zero as a missing-value indicator
# in this dataset's preprocessing convention.
columns_with_invalid_zero = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Make a separate copy of the feature dataset.
X_clean = X.copy()

# Replace the selected zero values with NaN.
# The transformation is a fixed cleaning rule.
X_clean[columns_with_invalid_zero] = (
    X_clean[columns_with_invalid_zero].replace(0, np.nan)
)


# ------------------------------------------------------------
# 4. CREATE THE IQR CLIPPER
# ------------------------------------------------------------

class IQRClipper(BaseEstimator, TransformerMixin):

    """
    Clips feature values outside their IQR boundaries.

    IMPORTANT:
    - fit() learns boundaries from the training fold.
    - transform() applies those learned boundaries.
    - Pandas quantile() ignores NaN values.
    - Therefore, missing values do not distort the
      IQR calculations in this experiment.
    """

    def __init__(self):

        # Store the learned lower and upper boundaries.
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

        # Store the original feature order.
        self.feature_names_in_ = None


    def fit(self, X, y=None):

        # Convert the input into a DataFrame.
        X_df = pd.DataFrame(X)

        # Remember the feature names and their order.
        self.feature_names_in_ = list(X_df.columns)

        # Calculate IQR boundaries for each feature.
        for column in X_df.columns:

            # First quartile (25th percentile).
            Q1 = X_df[column].quantile(0.25)

            # Third quartile (75th percentile).
            Q3 = X_df[column].quantile(0.75)

            # Interquartile range.
            IQR = Q3 - Q1

            # Lower outlier boundary.
            self.lower_bounds_[column] = Q1 - (1.5 * IQR)

            # Upper outlier boundary.
            self.upper_bounds_[column] = Q3 + (1.5 * IQR)

        # Return the fitted transformer.
        return self


    def transform(self, X):

        # Convert input into a DataFrame using the
        # feature order learned during fit().
        X_df = pd.DataFrame(
            X,
            columns=self.feature_names_in_
        ).copy()

        # Clip values using the learned boundaries.
        for column in X_df.columns:

            X_df[column] = X_df[column].clip(
                lower=self.lower_bounds_[column],
                upper=self.upper_bounds_[column]
            )

        # Return the transformed numerical data.
        return X_df.to_numpy()


# ------------------------------------------------------------
# 5. DEFINE STRATIFIED 5-FOLD CROSS-VALIDATION
# ------------------------------------------------------------

# Split the dataset into five folds.
# Each fold approximately preserves the original
# proportions of the two target classes.
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ------------------------------------------------------------
# 6. DEFINE THE FIVE CLASSIFICATION MODELS
# ------------------------------------------------------------

models = {

    # Linear classification algorithm.
    "Logistic Regression": LogisticRegression(
        random_state=42
    ),

    # Tree-based classification algorithm.
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=5
    ),

    # Ensemble of multiple decision trees.
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    # Distance-based classification algorithm.
    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        metric="euclidean"
    ),

    # Support Vector Machine with an RBF kernel.
    "SVM": SVC(
        kernel="rbf",
        C=1.0,
        random_state=42
    )
}


# ------------------------------------------------------------
# 7. DEFINE EVALUATION METRICS
# ------------------------------------------------------------

scoring = {

    # Overall proportion of correct predictions.
    "accuracy": "accuracy",

    # Correct positive predictions among all predicted positives.
    "precision": make_scorer(
        precision_score,
        zero_division=0
    ),

    # Correctly identified positives among actual positives.
    "recall": make_scorer(
        recall_score,
        zero_division=0
    ),

    # Harmonic mean of precision and recall.
    "f1": make_scorer(
        f1_score,
        zero_division=0
    ),

    # Ability to rank positive cases above negative cases.
    "roc_auc": "roc_auc"
}


# ------------------------------------------------------------
# 8. STORE EXPERIMENT RESULTS
# ------------------------------------------------------------

# Each model's cross-validation results will be stored here.
results = []


# ------------------------------------------------------------
# 9. RUN MODEL COMPARISON
# ------------------------------------------------------------

print("\n================================================")
print("MODEL COMPARISON WITH IQR-FIRST PREPROCESSING")
print("================================================")

print("\nDataset records:", len(X_clean))
print("Input features:", X_clean.shape[1])
print("Cross-validation folds:", skf.n_splits)


# Evaluate each classification algorithm.
for model_name, model in models.items():

    print("\n-----------------------------------------------")
    print("Model:", model_name)
    print("-----------------------------------------------")

    # Create a complete pipeline for the current model.
    pipeline = Pipeline([

        # Step 1: Learn and apply IQR boundaries
        # using observed training-fold values.
        ("iqr_clipping", IQRClipper()),

        # Step 2: Replace remaining missing values
        # using the training-fold median.
        ("imputer", SimpleImputer(strategy="median")),

        # Step 3: Standardize feature values.
        ("scaler", StandardScaler()),

        # Step 4: Train the selected classifier.
        ("model", model)
    ])

    # Evaluate the complete pipeline on the same five folds.
    cv_results = cross_validate(
        estimator=pipeline,
        X=X_clean,
        y=y,
        cv=skf,
        scoring=scoring,
        return_train_score=False,
        error_score="raise"
    )

    # Calculate mean and sample standard deviation
    # for every evaluation metric.
    model_result = {

        "Model": model_name,

        "Accuracy Mean": np.mean(cv_results["test_accuracy"]),
        "Accuracy Std": np.std(cv_results["test_accuracy"], ddof=1),

        "Precision Mean": np.mean(cv_results["test_precision"]),
        "Precision Std": np.std(cv_results["test_precision"], ddof=1),

        "Recall Mean": np.mean(cv_results["test_recall"]),
        "Recall Std": np.std(cv_results["test_recall"], ddof=1),

        "F1 Mean": np.mean(cv_results["test_f1"]),
        "F1 Std": np.std(cv_results["test_f1"], ddof=1),

        "ROC-AUC Mean": np.mean(cv_results["test_roc_auc"]),
        "ROC-AUC Std": np.std(cv_results["test_roc_auc"], ddof=1)
    }

    # Add the current model's results to the list.
    results.append(model_result)

    # Display the average evaluation metrics.
    print(f"Mean Accuracy : {model_result['Accuracy Mean']:.4f}")
    print(f"Mean Precision: {model_result['Precision Mean']:.4f}")
    print(f"Mean Recall   : {model_result['Recall Mean']:.4f}")
    print(f"Mean F1-Score : {model_result['F1 Mean']:.4f}")
    print(f"Mean ROC-AUC  : {model_result['ROC-AUC Mean']:.4f}")


# ------------------------------------------------------------
# 10. CREATE THE MODEL COMPARISON TABLE
# ------------------------------------------------------------

# Convert the collected results into a DataFrame.
comparison_df = pd.DataFrame(results)

# Round numerical values to four decimal places.
comparison_df = comparison_df.round(4)


# ------------------------------------------------------------
# 11. DISPLAY AND SAVE THE RESULTS
# ------------------------------------------------------------

print("\n================================================")
print("FINAL MODEL COMPARISON - IQR FIRST")
print("================================================")

# Display the comparison table.
print(comparison_df.to_string(index=False))

# Ensure the results folder exists.
Path("results").mkdir(parents=True, exist_ok=True)

# Save the model comparison results as a CSV file.
comparison_df.to_csv(
    "results/model_comparison_iqr_first.csv",
    index=False
)

print("\nResults saved to:")
print("results/model_comparison_iqr_first.csv")

print("\n================================================")
print("MODEL COMPARISON COMPLETED")
print("================================================")
