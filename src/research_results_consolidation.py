
# ============================================================
# PHASE 10.6: RESEARCH RESULTS CONSOLIDATION
# ============================================================
# Purpose:
# Combine the results of five ML experiments into one
# standardized research dataset.
#
# Input files:
# 1. cross_validation_comparison.csv
# 2. preprocessing_comparison.csv
# 3. model_comparison_iqr_first.csv
# 4. logistic_regression_tuning.csv
# 5. threshold_analysis.csv
#
# Output:
# results/research_results_consolidated.csv
#
# IMPORTANT:
# This script only reads existing results and creates
# a new consolidated CSV. It does not modify model files.
# ============================================================


# Import Path for handling project file locations.
from pathlib import Path

# Import Pandas for reading, processing and combining CSV files.
import pandas as pd


# ------------------------------------------------------------
# STEP 1: DEFINE PROJECT PATHS
# ------------------------------------------------------------

# Find the main project directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define the results directory.
RESULTS_DIR = PROJECT_ROOT / "results"

# Define the final consolidated output file.
OUTPUT_PATH = RESULTS_DIR / "research_results_consolidated.csv"


# ------------------------------------------------------------
# STEP 2: DEFINE A COMMON RESULT STRUCTURE
# ------------------------------------------------------------

# Every experiment will be converted into these common columns.
# Missing values are kept empty rather than being invented.
COMMON_COLUMNS = [
    "Experiment",
    "Configuration",
    "Accuracy",
    "Accuracy_Std",
    "Precision",
    "Precision_Std",
    "Recall",
    "Recall_Std",
    "F1_Score",
    "F1_Std",
    "ROC_AUC",
    "ROC_AUC_Std",
    "Threshold",
    "True_Negative",
    "False_Positive",
    "False_Negative",
    "True_Positive"
]


# Store the standardized results from each experiment.
all_results = []


# ------------------------------------------------------------
# STEP 3: CONSOLIDATE ORIGINAL MODEL COMPARISON
# ------------------------------------------------------------

# Read the original cross-validation comparison.
original = pd.read_csv(
    RESULTS_DIR / "cross_validation_comparison.csv"
)

# Rename its columns to the common structure.
original = original.rename(columns={
    "Model": "Configuration",
    "CV Accuracy": "Accuracy",
    "Accuracy Std": "Accuracy_Std",
    "CV Precision": "Precision",
    "CV Recall": "Recall",
    "CV F1": "F1_Score",
    "CV ROC-AUC": "ROC_AUC"
})

# Identify the source experiment.
original["Experiment"] = "Original Model Comparison"

# Add the standardized results.
all_results.append(original)


# ------------------------------------------------------------
# STEP 4: CONSOLIDATE PREPROCESSING COMPARISON
# ------------------------------------------------------------

# Read the preprocessing experiment results.
preprocessing = pd.read_csv(
    RESULTS_DIR / "preprocessing_comparison.csv"
)

# Rename the columns.
preprocessing = preprocessing.rename(columns={
    "Experiment": "Configuration",
    "Accuracy Mean": "Accuracy",
    "Accuracy Std": "Accuracy_Std",
    "Precision Mean": "Precision",
    "Precision Std": "Precision_Std",
    "Recall Mean": "Recall",
    "Recall Std": "Recall_Std",
    "F1 Mean": "F1_Score",
    "F1 Std": "F1_Std",
    "ROC-AUC Mean": "ROC_AUC",
    "ROC-AUC Std": "ROC_AUC_Std"
})

# Identify the source experiment.
preprocessing["Experiment"] = "Preprocessing Comparison"

# Add the standardized results.
all_results.append(preprocessing)


# ------------------------------------------------------------
# STEP 5: CONSOLIDATE IQR-FIRST MODEL COMPARISON
# ------------------------------------------------------------

# Read the IQR-first model comparison.
iqr_models = pd.read_csv(
    RESULTS_DIR / "model_comparison_iqr_first.csv"
)

# Rename the columns.
iqr_models = iqr_models.rename(columns={
    "Model": "Configuration",
    "Accuracy Mean": "Accuracy",
    "Accuracy Std": "Accuracy_Std",
    "Precision Mean": "Precision",
    "Precision Std": "Precision_Std",
    "Recall Mean": "Recall",
    "Recall Std": "Recall_Std",
    "F1 Mean": "F1_Score",
    "F1 Std": "F1_Std",
    "ROC-AUC Mean": "ROC_AUC",
    "ROC-AUC Std": "ROC_AUC_Std"
})

# Identify the source experiment.
iqr_models["Experiment"] = "IQR-First Model Comparison"

# Add the standardized results.
all_results.append(iqr_models)


# ------------------------------------------------------------
# STEP 6: CONSOLIDATE LOGISTIC REGRESSION TUNING
# ------------------------------------------------------------

# Read all tested Logistic Regression configurations.
tuning = pd.read_csv(
    RESULTS_DIR / "logistic_regression_tuning.csv"
)

# Rename the columns.
tuning = tuning.rename(columns={
    "Accuracy Mean": "Accuracy",
    "Accuracy Std": "Accuracy_Std",
    "Precision Mean": "Precision",
    "Precision Std": "Precision_Std",
    "Recall Mean": "Recall",
    "Recall Std": "Recall_Std",
    "F1 Mean": "F1_Score",
    "F1 Std": "F1_Std",
    "ROC-AUC Mean": "ROC_AUC",
    "ROC-AUC Std": "ROC_AUC_Std"
})

# Combine the hyperparameters into one readable configuration.
tuning["Configuration"] = (
    "C=" + tuning["C"].astype(str)
    + ", Penalty=" + tuning["Penalty"].astype(str)
    + ", Class Weight=" + tuning["Class Weight"].fillna("None").astype(str)
)

# Identify the source experiment.
tuning["Experiment"] = "Logistic Regression Tuning"

# Add the standardized results.
all_results.append(tuning)


# ------------------------------------------------------------
# STEP 7: CONSOLIDATE DECISION THRESHOLD ANALYSIS
# ------------------------------------------------------------

# Read the threshold analysis results.
thresholds = pd.read_csv(
    RESULTS_DIR / "threshold_analysis.csv"
)

# Rename the columns.
thresholds = thresholds.rename(columns={
    "Threshold": "Threshold",
    "Accuracy": "Accuracy",
    "Precision": "Precision",
    "Recall": "Recall",
    "F1_Score": "F1_Score",
    "ROC_AUC": "ROC_AUC",
    "True_Negative": "True_Negative",
    "False_Positive": "False_Positive",
    "False_Negative": "False_Negative",
    "True_Positive": "True_Positive"
})

# Record the threshold as the configuration.
thresholds["Configuration"] = (
    "Threshold=" + thresholds["Threshold"].astype(str)
)

# Identify the source experiment.
thresholds["Experiment"] = "Decision Threshold Analysis"

# Add the standardized results.
all_results.append(thresholds)


# ------------------------------------------------------------
# STEP 8: COMBINE ALL EXPERIMENTS
# ------------------------------------------------------------

# Combine all five experiment DataFrames.
# reindex ensures that every experiment has the same columns.
consolidated = pd.concat(
    all_results,
    ignore_index=True
).reindex(columns=COMMON_COLUMNS)


# ------------------------------------------------------------
# STEP 9: SAVE THE CONSOLIDATED RESULTS
# ------------------------------------------------------------

# Save the combined research results.
consolidated.to_csv(
    OUTPUT_PATH,
    index=False
)

# Display the number of consolidated experiment records.
print("\nRESEARCH RESULTS CONSOLIDATION")
print("=" * 70)

print(f"Total experiment records: {len(consolidated)}")

# Display the number of records from each experiment.
print("\nRecords by experiment:")

print(
    consolidated["Experiment"].value_counts().to_string()
)

# Display a preview of the consolidated dataset.
print("\nConsolidated results preview:")

print(
    consolidated.head(10).to_string(index=False)
)

# Confirm the output location.
print("\nConsolidation completed successfully.")
print(f"Saved to: {OUTPUT_PATH}")
