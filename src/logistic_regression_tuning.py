
# ============================================================
# DIABETES PREDICTION USING MACHINE LEARNING
# LOGISTIC REGRESSION HYPERPARAMETER TUNING
# ============================================================
#
# Research objective:
# Find better Logistic Regression settings using
# GridSearchCV and stratified 5-fold cross-validation.
#
# Preprocessing:
# 1. IQR outlier clipping
# 2. Median imputation
# 3. Standard scaling
#
# Parameters to investigate:
# 1. Regularization strength (C)
# 2. Regularization type (L1 and L2)
# 3. Class weighting (None and balanced)
#
# Evaluation metrics:
# Accuracy, Precision, Recall, F1-score and ROC-AUC.
#
# IMPORTANT:
# - Preprocessing remains inside the cross-validation pipeline.
# - The existing trained model is not overwritten.
# - This is an experimental model-selection script.
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

# Pandas handles dataset loading and result tables.
import pandas as pd

# NumPy provides numerical calculations.
import numpy as np

# Path creates the results directory when necessary.
from pathlib import Path

# Pipeline connects preprocessing and classification.
from sklearn.pipeline import Pipeline

# StratifiedKFold creates balanced validation folds.
# GridSearchCV tests parameter combinations automatically.
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# Base classes for our custom IQR transformer.
from sklearn.base import BaseEstimator, TransformerMixin

# Missing-value handling and feature scaling.
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Logistic Regression classifier.
from sklearn.linear_model import LogisticRegression

# Evaluation metric functions.
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score
)


# ------------------------------------------------------------
# 2. LOAD THE DATASET
# ------------------------------------------------------------

# Read the original diabetes dataset.
df = pd.read_csv("data/diabetes.csv")

# Separate the input features from the target column.
X = df.drop("Outcome", axis=1)

# Store the target values.
# 0 = negative class; 1 = positive class.
y = df["Outcome"]


# ------------------------------------------------------------
# 3. REPLACE INVALID ZERO VALUES
# ------------------------------------------------------------

# These physiological measurements use zero as a
# missing-value indicator in this dataset.
columns_with_invalid_zero = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Create a separate copy of the feature dataset.
X_clean = X.copy()

# Replace the selected zero values with NaN.
X_clean[columns_with_invalid_zero] = (
    X_clean[columns_with_invalid_zero].replace(0, np.nan)
)


# ------------------------------------------------------------
# 4. CREATE THE IQR CLIPPER
# ------------------------------------------------------------

class IQRClipper(BaseEstimator, TransformerMixin):

    """
    Clips values outside the IQR boundaries.

    The boundaries are learned from the training data
    of each cross-validation fold.

    NaN values are ignored when calculating quartiles.
    """

    def __init__(self):

        # Store learned lower and upper boundaries.
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

        # Store the original feature order.
        self.feature_names_in_ = None


    def fit(self, X, y=None):

        # Convert input into a DataFrame.
        X_df = pd.DataFrame(X)

        # Remember the feature order.
        self.feature_names_in_ = list(X_df.columns)

        # Calculate IQR boundaries for each feature.
        for column in X_df.columns:

            # Calculate the first quartile.
            Q1 = X_df[column].quantile(0.25)

            # Calculate the third quartile.
            Q3 = X_df[column].quantile(0.75)

            # Calculate the interquartile range.
            IQR = Q3 - Q1

            # Calculate and save the lower boundary.
            self.lower_bounds_[column] = Q1 - (1.5 * IQR)

            # Calculate and save the upper boundary.
            self.upper_bounds_[column] = Q3 + (1.5 * IQR)

        # Return the fitted transformer.
        return self


    def transform(self, X):

        # Convert input into a DataFrame using the
        # same feature order learned during fitting.
        X_df = pd.DataFrame(
            X,
            columns=self.feature_names_in_
        ).copy()

        # Apply the learned IQR boundaries.
        for column in X_df.columns:

            X_df[column] = X_df[column].clip(
                lower=self.lower_bounds_[column],
                upper=self.upper_bounds_[column]
            )

        # Return the transformed numerical data.
        return X_df.to_numpy()


# ------------------------------------------------------------
# 5. DEFINE STRATIFIED CROSS-VALIDATION
# ------------------------------------------------------------

# Use five folds and preserve class proportions.
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ------------------------------------------------------------
# 6. CREATE THE MACHINE LEARNING PIPELINE
# ------------------------------------------------------------

# Keep every preprocessing step inside the pipeline.
# GridSearchCV will fit these steps separately
# for every training fold.
pipeline = Pipeline([

    # Step 1: Calculate and apply IQR boundaries.
    ("iqr_clipping", IQRClipper()),

    # Step 2: Replace missing values with training medians.
    ("imputer", SimpleImputer(strategy="median")),

    # Step 3: Standardize the feature values.
    ("scaler", StandardScaler()),

    # Step 4: Logistic Regression classifier.
    ("model", LogisticRegression(
        random_state=42,
        max_iter=2000
    ))
])


# ------------------------------------------------------------
# 7. DEFINE HYPERPARAMETER COMBINATIONS
# ------------------------------------------------------------

# C controls the strength of regularization.
# Smaller C means stronger regularization.
# Larger C means weaker regularization.
#
# L1 and L2 are two regularization techniques.
#
# class_weight=None uses the original class weights.
# class_weight='balanced' adjusts weights according
# to the frequency of each class.
#
# liblinear supports both L1 and L2 penalties.

param_grid = {

    "model__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ],

    "model__penalty": [
        "l1",
        "l2"
    ],

    "model__class_weight": [
        None,
        "balanced"
    ],

    "model__solver": [
        "liblinear"
    ]
}


# ------------------------------------------------------------
# 8. DEFINE EVALUATION METRICS
# ------------------------------------------------------------

scoring = {

    # Overall percentage of correct predictions.
    "accuracy": "accuracy",

    # Correct positive predictions among predicted positives.
    "precision": make_scorer(
        precision_score,
        zero_division=0
    ),

    # Correctly identified positive cases among actual positives.
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
# 9. CONFIGURE GRID SEARCH
# ------------------------------------------------------------

print("\n================================================")
print("LOGISTIC REGRESSION HYPERPARAMETER TUNING")
print("================================================")

print("\nDataset records:", len(X_clean))
print("Input features:", X_clean.shape[1])
print("Cross-validation folds:", skf.n_splits)

# Calculate the total number of parameter combinations.
total_combinations = (
    len(param_grid["model__C"])
    * len(param_grid["model__penalty"])
    * len(param_grid["model__class_weight"])
    * len(param_grid["model__solver"])
)

print("\nParameter combinations:", total_combinations)

print(
    "Total cross-validation fits:",
    total_combinations * skf.n_splits
)


# Create GridSearchCV.
grid_search = GridSearchCV(

    # Complete preprocessing and classification pipeline.
    estimator=pipeline,

    # Parameter combinations to investigate.
    param_grid=param_grid,

    # Use the same five stratified folds.
    cv=skf,

    # Evaluate every candidate using all five metrics.
    scoring=scoring,

    # Select the best candidate using mean recall.
    # Other metrics will still be recorded for comparison.
    refit="recall",

    # Display progress during model tuning.
    verbose=1,

    # Use one CPU worker for predictable execution.
    n_jobs=1,

    # Raise errors rather than silently replacing failed scores.
    error_score="raise"
)


# ------------------------------------------------------------
# 10. RUN HYPERPARAMETER TUNING
# ------------------------------------------------------------

print("\nStarting GridSearchCV...")

# Search for the best hyperparameter combination.
grid_search.fit(X_clean, y)

print("\nGrid search completed successfully.")


# ------------------------------------------------------------
# 11. DISPLAY THE BEST HYPERPARAMETERS
# ------------------------------------------------------------

print("\n================================================")
print("BEST LOGISTIC REGRESSION CONFIGURATION")
print("================================================")

# Display the parameter combination selected by recall.
print("\nBest parameters:")

for parameter, value in grid_search.best_params_.items():

    print(parameter, ":", value)


# Display the best mean recall across five folds.
print(
    "\nBest mean CV Recall:",
    round(grid_search.best_score_, 4)
)


# ------------------------------------------------------------
# 12. DISPLAY OTHER METRICS FOR THE SELECTED MODEL
# ------------------------------------------------------------

# GridSearchCV refits the selected model using all available
# data after completing cross-validation.
#
# The metrics below come from its cross-validation results,
# not from evaluating the fitted model on its training data.

best_index = grid_search.best_index_

cv_results = grid_search.cv_results_

print("\nSelected model's cross-validation metrics:")

print(
    "Accuracy :",
    round(cv_results["mean_test_accuracy"][best_index], 4)
)

print(
    "Precision:",
    round(cv_results["mean_test_precision"][best_index], 4)
)

print(
    "Recall   :",
    round(cv_results["mean_test_recall"][best_index], 4)
)

print(
    "F1-score :",
    round(cv_results["mean_test_f1"][best_index], 4)
)

print(
    "ROC-AUC  :",
    round(cv_results["mean_test_roc_auc"][best_index], 4)
)


# ------------------------------------------------------------
# 13. CREATE THE COMPLETE TUNING RESULTS TABLE
# ------------------------------------------------------------

# Convert all parameter-search results into a DataFrame.
tuning_df = pd.DataFrame(cv_results)

# Keep the most useful columns for research analysis.
selected_columns = [

    "param_model__C",
    "param_model__penalty",
    "param_model__class_weight",

    "mean_test_accuracy",
    "std_test_accuracy",

    "mean_test_precision",
    "std_test_precision",

    "mean_test_recall",
    "std_test_recall",

    "mean_test_f1",
    "std_test_f1",

    "mean_test_roc_auc",
    "std_test_roc_auc",

    "rank_test_recall"
]

# Select only the required columns.
tuning_df = tuning_df[selected_columns].copy()

# Rename columns to make the research table easier to read.
tuning_df.columns = [

    "C",
    "Penalty",
    "Class Weight",

    "Accuracy Mean",
    "Accuracy Std",

    "Precision Mean",
    "Precision Std",

    "Recall Mean",
    "Recall Std",

    "F1 Mean",
    "F1 Std",

    "ROC-AUC Mean",
    "ROC-AUC Std",

    "Recall Rank"
]

# Sort configurations by their mean recall.
tuning_df = tuning_df.sort_values(
    by=["Recall Mean", "ROC-AUC Mean"],
    ascending=[False, False]
)

# Round numerical values for readability.
tuning_df = tuning_df.round(4)


# ------------------------------------------------------------
# 14. DISPLAY THE TOP CONFIGURATIONS
# ------------------------------------------------------------

print("\n================================================")
print("TOP 10 HYPERPARAMETER CONFIGURATIONS")
print("================================================")

print(
    tuning_df.head(10).to_string(index=False)
)


# ------------------------------------------------------------
# 15. SAVE THE RESULTS
# ------------------------------------------------------------

# Ensure the results directory exists.
Path("results").mkdir(parents=True, exist_ok=True)

# Save every tested configuration and its metrics.
tuning_df.to_csv(
    "results/logistic_regression_tuning.csv",
    index=False
)

print("\nResults saved to:")
print("results/logistic_regression_tuning.csv")


# ------------------------------------------------------------
# 16. COMPLETION MESSAGE
# ------------------------------------------------------------

print("\n================================================")
print("LOGISTIC REGRESSION TUNING COMPLETED")
print("================================================")

print(
    "\nNote: The existing saved model was not modified."
)
