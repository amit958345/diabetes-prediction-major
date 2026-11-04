
# ============================================================
# DIABETES PREDICTION USING MACHINE LEARNING
# PREPROCESSING STRATEGY COMPARISON
# ============================================================
#
# Research objective:
# Compare three preprocessing strategies to investigate
# whether IQR outlier clipping after median imputation
# reduces useful information in the Insulin feature.
#
# Experiment A: Imputation -> IQR Clipping -> Scaling
# Experiment B: IQR Clipping -> Imputation -> Scaling
# Experiment C: Imputation -> Scaling (No IQR Clipping)
#
# All experiments use:
# - The same Logistic Regression classifier.
# - The same 5-fold Stratified Cross-Validation.
# - The same evaluation metrics.
#
# This is an experimental script. It does not overwrite
# the existing trained model or prediction API.
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

# Pandas is used to load and manipulate the dataset.
import pandas as pd

# NumPy is used for numerical calculations.
import numpy as np

# Path helps create the results directory if necessary.
from pathlib import Path

# Pipeline connects preprocessing steps and the ML model.
from sklearn.pipeline import Pipeline

# StratifiedKFold creates balanced cross-validation folds.
# cross_validate evaluates the complete pipeline.
from sklearn.model_selection import StratifiedKFold, cross_validate

# BaseEstimator and TransformerMixin help us create
# a custom preprocessing transformer compatible with sklearn.
from sklearn.base import BaseEstimator, TransformerMixin

# SimpleImputer replaces missing values.
from sklearn.impute import SimpleImputer

# StandardScaler standardizes feature values.
from sklearn.preprocessing import StandardScaler

# Logistic Regression is our existing baseline classifier.
from sklearn.linear_model import LogisticRegression

# These functions calculate classification evaluation metrics.
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

# Separate the eight input features from the target column.
X = df.drop("Outcome", axis=1)

# Store the target values (0 = negative class, 1 = positive class).
y = df["Outcome"]


# ------------------------------------------------------------
# 3. HANDLE INVALID ZERO VALUES
# ------------------------------------------------------------

# These measurements should not normally be zero
# in the dataset's intended physiological context.
columns_with_invalid_zero = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Create a separate copy so the original dataset remains unchanged.
X_clean = X.copy()

# Replace zero values in the selected columns with NaN.
# NaN represents a missing value.
X_clean[columns_with_invalid_zero] = (
    X_clean[columns_with_invalid_zero].replace(0, np.nan)
)


# ------------------------------------------------------------
# 4. CREATE A CROSS-VALIDATION-SAFE IQR CLIPPER
# ------------------------------------------------------------

class IQRClipper(BaseEstimator, TransformerMixin):

    """
    Custom transformer that clips values outside IQR boundaries.

    The boundaries are learned during fit() using only the
    training data supplied to that cross-validation fold.

    Pandas quantile() ignores NaN values. Therefore, when this
    transformer is used before imputation, IQR boundaries are
    calculated from observed values rather than median-filled values.
    """

    def __init__(self):

        # Store the lower and upper clipping boundaries.
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

        # Store the feature names and their original order.
        self.feature_names_in_ = None


    def fit(self, X, y=None):

        # Convert input into a DataFrame for feature-wise calculations.
        X_df = pd.DataFrame(X)

        # Remember the feature names/order used during fitting.
        self.feature_names_in_ = list(X_df.columns)

        # Calculate IQR boundaries separately for every feature.
        for column in X_df.columns:

            # First quartile: 25th percentile.
            Q1 = X_df[column].quantile(0.25)

            # Third quartile: 75th percentile.
            Q3 = X_df[column].quantile(0.75)

            # Interquartile range.
            IQR = Q3 - Q1

            # Calculate the lower clipping boundary.
            lower_bound = Q1 - (1.5 * IQR)

            # Calculate the upper clipping boundary.
            upper_bound = Q3 + (1.5 * IQR)

            # Save the boundaries learned from this training fold.
            self.lower_bounds_[column] = lower_bound
            self.upper_bounds_[column] = upper_bound

        # Return the fitted transformer.
        return self


    def transform(self, X):

        # Convert input into a DataFrame using the same feature order.
        X_df = pd.DataFrame(
            X,
            columns=self.feature_names_in_
        ).copy()

        # Apply the boundaries learned during fit().
        for column in X_df.columns:

            X_df[column] = X_df[column].clip(
                lower=self.lower_bounds_[column],
                upper=self.upper_bounds_[column]
            )

        # Return a NumPy array for compatibility with later steps.
        return X_df.to_numpy()


# ------------------------------------------------------------
# 5. DEFINE STRATIFIED 5-FOLD CROSS-VALIDATION
# ------------------------------------------------------------

# Stratification keeps the class proportions approximately
# equal across all five folds.
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ------------------------------------------------------------
# 6. DEFINE EVALUATION METRICS
# ------------------------------------------------------------

# All three experiments will use exactly the same metrics.
scoring = {

    # Overall percentage of correct predictions.
    "accuracy": "accuracy",

    # Percentage of predicted positive cases that are positive.
    "precision": make_scorer(
        precision_score,
        zero_division=0
    ),

    # Percentage of actual positive cases correctly identified.
    "recall": make_scorer(
        recall_score,
        zero_division=0
    ),

    # Harmonic mean of precision and recall.
    "f1": make_scorer(
        f1_score,
        zero_division=0
    ),

    # Measures the model's ability to rank positive cases
    # above negative cases across classification thresholds.
    "roc_auc": "roc_auc"
}


# ------------------------------------------------------------
# 7. DEFINE THE THREE PREPROCESSING EXPERIMENTS
# ------------------------------------------------------------

experiments = {

    # Experiment A:
    # This reproduces the preprocessing order in the
    # existing cross-validation baseline.
    "A - Impute -> IQR -> Scale": Pipeline([

        ("imputer", SimpleImputer(strategy="median")),

        ("iqr_clipping", IQRClipper()),

        ("scaler", StandardScaler()),

        ("model", LogisticRegression(random_state=42))
    ]),


    # Experiment B:
    # Calculate IQR boundaries from observed values first.
    # Missing values remain NaN during clipping and are
    # filled afterward using the median.
    "B - IQR -> Impute -> Scale": Pipeline([

        ("iqr_clipping", IQRClipper()),

        ("imputer", SimpleImputer(strategy="median")),

        ("scaler", StandardScaler()),

        ("model", LogisticRegression(random_state=42))
    ]),


    # Experiment C:
    # Do not perform IQR clipping.
    # This tests whether clipping is necessary at all.
    "C - Impute -> Scale": Pipeline([

        ("imputer", SimpleImputer(strategy="median")),

        ("scaler", StandardScaler()),

        ("model", LogisticRegression(random_state=42))
    ])
}


# ------------------------------------------------------------
# 8. RUN THE PREPROCESSING EXPERIMENTS
# ------------------------------------------------------------

# This list will hold the evaluation results for each experiment.
results = []

print("\n================================================")
print("PREPROCESSING STRATEGY COMPARISON")
print("================================================")

print("\nDataset records:", len(X_clean))
print("Input features:", X_clean.shape[1])
print("Cross-validation folds:", skf.n_splits)


# Run every preprocessing experiment using identical folds.
for experiment_name, pipeline in experiments.items():

    print("\n-----------------------------------------------")
    print("Experiment:", experiment_name)
    print("-----------------------------------------------")

    # Evaluate the complete pipeline using 5-fold CV.
    cv_results = cross_validate(
        estimator=pipeline,
        X=X_clean,
        y=y,
        cv=skf,
        scoring=scoring,
        return_train_score=False,
        error_score="raise"
    )

    # Calculate the mean and sample standard deviation
    # for each metric across the five validation folds.
    experiment_result = {
        "Experiment": experiment_name,

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

    # Add this experiment's results to the final results list.
    results.append(experiment_result)

    # Display the average evaluation metrics.
    print(f"Mean Accuracy : {experiment_result['Accuracy Mean']:.4f}")
    print(f"Mean Precision: {experiment_result['Precision Mean']:.4f}")
    print(f"Mean Recall   : {experiment_result['Recall Mean']:.4f}")
    print(f"Mean F1-Score : {experiment_result['F1 Mean']:.4f}")
    print(f"Mean ROC-AUC  : {experiment_result['ROC-AUC Mean']:.4f}")


# ------------------------------------------------------------
# 9. CREATE THE FINAL COMPARISON TABLE
# ------------------------------------------------------------

# Convert the results list into a Pandas DataFrame.
comparison_df = pd.DataFrame(results)

# Round numerical values to four decimal places.
comparison_df = comparison_df.round(4)


# ------------------------------------------------------------
# 10. DISPLAY AND SAVE THE RESULTS
# ------------------------------------------------------------

print("\n================================================")
print("FINAL PREPROCESSING COMPARISON")
print("================================================")

# Display the complete comparison table.
print(comparison_df.to_string(index=False))

# Ensure the results directory exists.
Path("results").mkdir(parents=True, exist_ok=True)

# Save the comparison results as a CSV file.
comparison_df.to_csv(
    "results/preprocessing_comparison.csv",
    index=False
)

print("\nResults saved to:")
print("results/preprocessing_comparison.csv")

print("\n================================================")
print("PREPROCESSING COMPARISON COMPLETED")
print("================================================")
