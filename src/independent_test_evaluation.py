
# ============================================================
# PHASE 10.7: INDEPENDENT TEST-SET EVALUATION
# ============================================================
# Purpose:
# Prepare an independent training and testing dataset.
#
# Training data:
# Used for model development and selection.
#
# Testing data:
# Reserved for final evaluation on unseen records.
#
# IMPORTANT:
# This script is separate from the original training script.
# It does not overwrite any existing model files.
# ============================================================


# Clone creates a fresh, unfitted copy of the selected pipeline.
from sklearn.base import clone

# Generate out-of-fold probabilities using training data only.
from sklearn.model_selection import cross_val_predict

# Import metrics for threshold comparison.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Import GridSearchCV for hyperparameter optimization.
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Import the preprocessing and classification components.
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Import our custom IQR clipping transformer.
from iqr_clipper import IQRClipper

# Import Path to manage project file locations.
from pathlib import Path

# Import NumPy and Pandas for dataset processing.
import numpy as np
import pandas as pd

# Import train_test_split for separating training and testing data.
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# STEP 1: DEFINE THE PROJECT DIRECTORY
# ------------------------------------------------------------

# Find the main project directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Define the folder where all research result CSV files will be saved.
# PROJECT_ROOT points to the main project directory.
RESULTS_DIR = PROJECT_ROOT / "results"


# Define the dataset location.
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"


# ------------------------------------------------------------
# STEP 2: LOAD THE DATASET
# ------------------------------------------------------------

# Read the diabetes dataset.
df = pd.read_csv(DATA_PATH)

# Display the original dataset dimensions.
print("\nINDEPENDENT TEST-SET EVALUATION")
print("=" * 65)

print("\nOriginal dataset shape:", df.shape)


# ------------------------------------------------------------
# STEP 3: SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------

# Select all input columns except the Outcome column.
X = df.drop(columns=["Outcome"]).copy()

# Store the actual outcome labels.
y = df["Outcome"].copy()

# Replace invalid zero measurements with missing values.
# These replacements are made before splitting because they
# only identify missing-value markers; no statistics are learned.
columns_with_invalid_zero = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

X[columns_with_invalid_zero] = (
    X[columns_with_invalid_zero].replace(0, np.nan)
)


# ------------------------------------------------------------
# STEP 4: CREATE A STRATIFIED TRAIN-TEST SPLIT
# ------------------------------------------------------------

# Reserve 20% of the records for independent testing.
# Use stratification to preserve the class proportions.
# A fixed random state makes the split reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# STEP 5: DISPLAY THE SPLIT INFORMATION
# ------------------------------------------------------------

# Display the number of records in each partition.
print("\nDATASET SPLIT")
print("-" * 65)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Display the number of input features.
print("Number of features:", X_train.shape[1])

# Display class distributions in the training partition.
print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

# Display class distributions in the testing partition.
print("\nTesting class distribution:")
print(y_test.value_counts().sort_index())

# Calculate the positive-class percentages.
print("\nPositive class percentage:")

print(
    "Training:",
    round((y_train == 1).mean() * 100, 2),
    "%"
)

print(
    "Testing:",
    round((y_test == 1).mean() * 100, 2),
    "%"
)


# ------------------------------------------------------------
# STEP 6: VERIFY THE SPLIT
# ------------------------------------------------------------

# Confirm that the training and testing indexes do not overlap.
overlap = set(X_train.index).intersection(set(X_test.index))

print("\nSPLIT VALIDATION")
print("-" * 65)

print("Overlapping records:", len(overlap))

# Stop execution if the split contains overlapping records.
assert len(overlap) == 0, "Training and testing data overlap!"

# Confirm that all records have been assigned.
assert len(X_train) + len(X_test) == len(df)

print("Dataset split validation passed.")

print("\nNo model has been trained or overwritten.")


# ------------------------------------------------------------
# STEP 7: BUILD THE TRAINING-ONLY MODEL PIPELINE
# ------------------------------------------------------------

# Create a pipeline that performs all learned preprocessing
# separately inside each cross-validation training fold.
#
# Order:
# IQR clipping -> Median imputation -> Standard scaling
# -> Logistic Regression.
#
# This order was investigated in our previous experiments.
training_pipeline = Pipeline([
    ("iqr_clipping", IQRClipper()),
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        solver="liblinear",
        random_state=42,
        max_iter=2000
    ))
])


# ------------------------------------------------------------
# STEP 8: DEFINE THE HYPERPARAMETER SEARCH
# ------------------------------------------------------------

# Define the values that GridSearchCV will evaluate.
#
# l1_ratio = 0.0 represents L2 regularization.
# l1_ratio = 1.0 represents L1 regularization.
#
# This uses the newer scikit-learn regularization interface.
parameter_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__l1_ratio": [0.0, 1.0],
    "model__class_weight": [None, "balanced"]
}


# ------------------------------------------------------------
# STEP 9: CONFIGURE CROSS-VALIDATION
# ------------------------------------------------------------

# Use five stratified folds.
# Each fold preserves the class proportions.
training_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ------------------------------------------------------------
# STEP 10: RUN HYPERPARAMETER TUNING
# ------------------------------------------------------------

# Evaluate every parameter combination using training data only.
#
# Recall is the selection metric because our current research
# prioritizes identifying positive cases.
#
# The independent test partition is not passed to this search.
grid_search = GridSearchCV(
    estimator=training_pipeline,
    param_grid=parameter_grid,
    scoring={
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    },
    refit="recall",
    cv=training_cv,
    n_jobs=1,
    verbose=1,
    error_score="raise",
    return_train_score=False
)


# ------------------------------------------------------------
# STEP 11: FIT THE SEARCH ON TRAINING DATA
# ------------------------------------------------------------

print("\nTRAINING-ONLY HYPERPARAMETER TUNING")
print("=" * 65)

# Perform hyperparameter selection using only X_train and y_train.
grid_search.fit(X_train, y_train)


# ------------------------------------------------------------
# STEP 12: DISPLAY THE BEST CONFIGURATION
# ------------------------------------------------------------

# Display the parameter combination with the highest mean CV recall.
print("\nBest hyperparameters:")
print(grid_search.best_params_)

# Display the best mean cross-validation recall.
print(
    "\nBest training CV recall:",
    round(grid_search.best_score_, 4)
)

# Retrieve the cross-validation results of the selected model.
best_index = grid_search.best_index_

# Display all evaluation metrics for the selected configuration.
print("\nSelected model cross-validation metrics:")

for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:

    mean_value = grid_search.cv_results_[
        f"mean_test_{metric}"
    ][best_index]

    std_value = grid_search.cv_results_[
        f"std_test_{metric}"
    ][best_index]

    print(
        f"{metric.upper():10}: "
        f"{mean_value:.4f} (+/- {std_value:.4f})"
    )


# ------------------------------------------------------------
# STEP 13: SAVE TRAINING-ONLY TUNING RESULTS
# ------------------------------------------------------------

# Convert all hyperparameter search results into a DataFrame.
tuning_results = pd.DataFrame(grid_search.cv_results_)

# Keep useful parameter and evaluation columns.
selected_columns = [
    "param_model__C",
    "param_model__l1_ratio",
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
    "std_test_roc_auc"
]

# Select and rename columns for easier interpretation.
tuning_results = tuning_results[selected_columns].rename(columns={
    "param_model__C": "C",
    "param_model__l1_ratio": "L1_Ratio",
    "param_model__class_weight": "Class_Weight",
    "mean_test_accuracy": "Accuracy_Mean",
    "std_test_accuracy": "Accuracy_Std",
    "mean_test_precision": "Precision_Mean",
    "std_test_precision": "Precision_Std",
    "mean_test_recall": "Recall_Mean",
    "std_test_recall": "Recall_Std",
    "mean_test_f1": "F1_Mean",
    "std_test_f1": "F1_Std",
    "mean_test_roc_auc": "ROC_AUC_Mean",
    "std_test_roc_auc": "ROC_AUC_Std"
})

# Create the results directory if necessary.
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Save the training-only tuning results separately.
tuning_output = RESULTS_DIR / "independent_training_tuning.csv"

tuning_results.to_csv(
    tuning_output,
    index=False
)

print("\nTraining-only tuning results saved to:")
print(tuning_output)

print("\nIndependent test data has not been evaluated.")



# ============================================================
# STEP 14: GENERATE OUT-OF-FOLD PROBABILITIES
# ============================================================

# Create a fresh, unfitted copy of the original training pipeline.
# Apply the best hyperparameters found by GridSearchCV.
# This prevents using predictions from a model trained on the
# same records that it is predicting.
best_pipeline_for_oof = clone(training_pipeline).set_params(
    **grid_search.best_params_
)

# Generate predictions for training records using cross-validation.
# Each record receives a probability from a model that did not
# train on that particular record.
oof_probabilities = cross_val_predict(
    best_pipeline_for_oof,
    X_train,
    y_train,
    cv=training_cv,
    method="predict_proba",
    n_jobs=1
)

# Extract the probability of the positive class (Outcome = 1).
positive_probabilities = oof_probabilities[:, 1]

# ============================================================
# STEP 15: COMPARE DIFFERENT DECISION THRESHOLDS
# ============================================================

# Test several possible thresholds.
# We will compare their performance rather than choosing one blindly.
threshold_values = [
    0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70
]

# Store the evaluation results for every threshold.
threshold_results = []

for threshold in threshold_values:

    # Convert probabilities into predicted classes.
    # Probability >= threshold means positive prediction (Outcome = 1).
    threshold_predictions = (
        positive_probabilities >= threshold
    ).astype(int)

    # Calculate confusion matrix values.
    tn, fp, fn, tp = confusion_matrix(
        y_train,
        threshold_predictions,
        labels=[0, 1]
    ).ravel()

    # Calculate evaluation metrics.
    threshold_results.append({
        "threshold": threshold,
        "accuracy": accuracy_score(
            y_train, threshold_predictions
        ),
        "precision": precision_score(
            y_train, threshold_predictions, zero_division=0
        ),
        "recall": recall_score(
            y_train, threshold_predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_train, threshold_predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_train, positive_probabilities
        ),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp
    })

# Convert the results into a Pandas DataFrame.
threshold_df = pd.DataFrame(threshold_results)

# Display the threshold comparison.
print("\nTRAINING-ONLY DECISION THRESHOLD ANALYSIS")
print("=" * 65)
print(threshold_df.round(4).to_string(index=False))

# ============================================================
# STEP 16: SAVE THRESHOLD ANALYSIS RESULTS
# ============================================================

# Save the threshold comparison separately from the test results.
threshold_output = (
    RESULTS_DIR / "independent_training_threshold_analysis.csv"
)

threshold_df.to_csv(threshold_output, index=False)

print("\nTraining-only threshold results saved to:")
print(threshold_output)

print("\nIndependent test data has still not been evaluated.")



# ============================================================
# STEP 17: EVALUATE THE SELECTED MODEL ON INDEPENDENT TEST DATA
# ============================================================

# Use the best estimator selected by GridSearchCV.
# GridSearchCV has already refitted this estimator on all
# 614 training records. It has never used the test records.
final_model = grid_search.best_estimator_

# Generate positive-class probabilities for the independent test set.
test_probabilities = final_model.predict_proba(X_test)[:, 1]

# Use the threshold selected using training-only predictions.
selected_threshold = 0.45

# Convert test probabilities into final predicted classes.
test_predictions = (
    test_probabilities >= selected_threshold
).astype(int)

# ============================================================
# STEP 18: CALCULATE FINAL TEST METRICS
# ============================================================

# Calculate confusion matrix values.
tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
).ravel()

# Calculate final performance metrics.
test_accuracy = accuracy_score(y_test, test_predictions)
test_precision = precision_score(
    y_test, test_predictions, zero_division=0
)
test_recall = recall_score(
    y_test, test_predictions, zero_division=0
)
test_f1 = f1_score(
    y_test, test_predictions, zero_division=0
)

# ROC-AUC uses probabilities, not thresholded predictions.
test_roc_auc = roc_auc_score(y_test, test_probabilities)

# Specificity measures how many actual negative cases
# were correctly identified.
test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# ============================================================
# STEP 19: DISPLAY FINAL TEST RESULTS
# ============================================================

print("\nFINAL INDEPENDENT TEST-SET RESULTS")
print("=" * 65)

print(f"Selected threshold : {selected_threshold:.2f}")
print(f"Test records       : {len(X_test)}")

print("\nEvaluation metrics:")
print(f"Accuracy           : {test_accuracy:.4f}")
print(f"Precision          : {test_precision:.4f}")
print(f"Recall (Sensitivity): {test_recall:.4f}")
print(f"Specificity        : {test_specificity:.4f}")
print(f"F1-score           : {test_f1:.4f}")
print(f"ROC-AUC            : {test_roc_auc:.4f}")

print("\nConfusion matrix:")
print(f"True Negatives  (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives  (TP): {tp}")

# ============================================================
# STEP 20: SAVE FINAL TEST RESULTS
# ============================================================

# Store the final independent test evaluation in one row.
test_results = pd.DataFrame([{
    "selected_threshold": selected_threshold,
    "test_records": len(X_test),
    "accuracy": test_accuracy,
    "precision": test_precision,
    "recall": test_recall,
    "specificity": test_specificity,
    "f1_score": test_f1,
    "roc_auc": test_roc_auc,
    "true_negative": tn,
    "false_positive": fp,
    "false_negative": fn,
    "true_positive": tp
}])

# Save the final evaluation separately from training experiments.
test_output = RESULTS_DIR / "independent_test_evaluation.csv"
test_results.to_csv(test_output, index=False)

print("\nFinal test evaluation saved to:")
print(test_output)

print("\nIndependent test evaluation completed.")
