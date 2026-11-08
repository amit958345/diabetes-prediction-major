from pathlib import Path
import pandas as pd
import numpy as np
from fastapi import HTTPException, status

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
ZERO_AS_MISSING = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
]

def _read_csv(filename: str) -> pd.DataFrame:
    path = RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Required research result not found: {filename}")
    return pd.read_csv(path)

def _rounded(value, digits=4):
    if pd.isna(value):
        return None
    return round(float(value), digits)

def get_model_insights() -> dict:
    """Build authenticated, read-only insights from the project's real artifacts."""
    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset not found at data/diabetes.csv")

    dataset = pd.read_csv(DATA_PATH)
    missing_columns = [column for column in FEATURES + ["Outcome"] if column not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")

    raw_features = dataset[FEATURES].copy()
    analysis_features = raw_features.replace({column: 0 for column in ZERO_AS_MISSING}, np.nan)

    feature_summary = []
    for feature in FEATURES:
        series = analysis_features[feature].dropna()
        feature_summary.append({
            "feature": feature,
            "count": int(series.count()),
            "missing_or_zero_count": int(analysis_features[feature].isna().sum()),
            "mean": _rounded(series.mean(), 3),
            "median": _rounded(series.median(), 3),
            "std": _rounded(series.std(), 3),
            "min": _rounded(series.min(), 3),
            "max": _rounded(series.max(), 3),
            "q1": _rounded(series.quantile(0.25), 3),
            "q3": _rounded(series.quantile(0.75), 3),
        })

    # Pairwise-complete Pearson correlations after treating medically invalid
    # zero measurements as missing for the five affected variables.
    correlation = analysis_features.corr(method="pearson").replace({np.nan: None})
    correlation_matrix = [
        {
            "feature": feature,
            "values": [
                _rounded(correlation.loc[feature, other], 3)
                for other in FEATURES
            ]
        }
        for feature in FEATURES
    ]

    outcome_counts = dataset["Outcome"].value_counts().reindex([0, 1], fill_value=0)
    total = int(len(dataset))
    outcome_distribution = [
        {
            "label": "Non-diabetic (0)",
            "class": 0,
            "count": int(outcome_counts.loc[0]),
            "percentage": _rounded(outcome_counts.loc[0] / total * 100, 2)
        },
        {
            "label": "Diabetic (1)",
            "class": 1,
            "count": int(outcome_counts.loc[1]),
            "percentage": _rounded(outcome_counts.loc[1] / total * 100, 2)
        }
    ]

    model_comparison_df = _read_csv("model_comparison.csv")
    model_comparison = []
    for _, row in model_comparison_df.iterrows():
        model_comparison.append({
            "model": str(row["Model"]),
            "accuracy": _rounded(row["Accuracy"]),
            "precision": _rounded(row["Precision"]),
            "recall": _rounded(row["Recall"]),
            "f1_score": _rounded(row["F1 Score"]),
            "roc_auc": _rounded(row["ROC-AUC"])
        })

    cv_df = _read_csv("cross_validation_comparison.csv")
    cross_validation = []
    for _, row in cv_df.iterrows():
        cross_validation.append({
            "model": str(row["Model"]),
            "accuracy": _rounded(row["CV Accuracy"]),
            "precision": _rounded(row["CV Precision"]),
            "recall": _rounded(row["CV Recall"]),
            "f1_score": _rounded(row["CV F1"]),
            "roc_auc": _rounded(row["CV ROC-AUC"]),
            "accuracy_std": _rounded(row["Accuracy Std"])
        })

    independent_df = _read_csv("independent_test_evaluation.csv")
    independent = independent_df.iloc[0].to_dict()

    preprocessing_df = _read_csv("preprocessing_comparison.csv")
    preprocessing = []
    for _, row in preprocessing_df.iterrows():
        preprocessing.append({
            "experiment": str(row["Experiment"]),
            "accuracy_mean": _rounded(row["Accuracy Mean"]),
            "accuracy_std": _rounded(row["Accuracy Std"]),
            "precision_mean": _rounded(row["Precision Mean"]),
            "recall_mean": _rounded(row["Recall Mean"]),
            "f1_mean": _rounded(row["F1 Mean"]),
            "roc_auc_mean": _rounded(row["ROC-AUC Mean"])
        })

    return {
        "project": {
            "name": "DiaPredict",
            "dataset_name": "Pima Indians Diabetes Dataset",
            "dataset_records": total,
            "feature_count": len(FEATURES),
            "target": "Outcome",
            "analysis_note": (
                "For Glucose, BloodPressure, SkinThickness, Insulin and BMI, "
                "zero values are treated as missing measurements in feature "
                "statistics and pairwise correlations. The source CSV itself "
                "contains no explicit NaN values."
            )
        },
        "dataset": {
            "outcome_distribution": outcome_distribution,
            "explicit_missing_cells": int(dataset.isna().sum().sum()),
            "zero_as_missing_counts": {
                feature: int((raw_features[feature] == 0).sum())
                for feature in ZERO_AS_MISSING
            },
            "feature_summary": feature_summary,
            "correlation_matrix": correlation_matrix
        },
        "model": {
            "selected_algorithm": "Logistic Regression",
            "independent_test": {
                "threshold": _rounded(independent.get("selected_threshold")),
                "records": int(independent.get("test_records")),
                "accuracy": _rounded(independent.get("accuracy")),
                "precision": _rounded(independent.get("precision")),
                "recall": _rounded(independent.get("recall")),
                "specificity": _rounded(independent.get("specificity")),
                "f1_score": _rounded(independent.get("f1_score")),
                "roc_auc": _rounded(independent.get("roc_auc")),
                "confusion_matrix": {
                    "true_negative": int(independent.get("true_negative")),
                    "false_positive": int(independent.get("false_positive")),
                    "false_negative": int(independent.get("false_negative")),
                    "true_positive": int(independent.get("true_positive"))
                }
            },
            "algorithm_comparison": model_comparison,
            "cross_validation": cross_validation,
            "preprocessing_comparison": preprocessing
        },
        "provenance": {
            "dataset_file": "data/diabetes.csv",
            "model_comparison_file": "results/model_comparison.csv",
            "cross_validation_file": "results/cross_validation_comparison.csv",
            "independent_test_file": "results/independent_test_evaluation.csv",
            "preprocessing_file": "results/preprocessing_comparison.csv"
        }
    }
