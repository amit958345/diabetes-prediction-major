# ==========================================
# DIABETES PREDICTION USING MACHINE LEARNING
# MULTI-METRIC 5-FOLD STRATIFIED
# CROSS-VALIDATION
# ==========================================


# ------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# ------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------

df = pd.read_csv("data/diabetes.csv")


# Separate features and target.
X = df.drop("Outcome", axis=1)

y = df["Outcome"]


# ------------------------------------------
# 3. REPLACE INVALID ZERO VALUES
# ------------------------------------------

# Zero is treated as invalid/missing for
# these physiological measurements.

columns_with_invalid_zero = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]


# Create a copy.
X_clean = X.copy()


# Replace invalid zeros with NaN.
X_clean[columns_with_invalid_zero] = (
    X_clean[columns_with_invalid_zero]
    .replace(0, np.nan)
)


# ------------------------------------------
# 4. CUSTOM IQR CLIPPER
# ------------------------------------------

class IQRClipper(BaseEstimator, TransformerMixin):

    """
    Performs IQR-based outlier clipping.

    The important point is that the IQR boundaries
    are learned only from the training portion
    of each cross-validation fold.
    """

    def __init__(self):

        # Store lower boundaries.
        self.lower_bounds_ = {}

        # Store upper boundaries.
        self.upper_bounds_ = {}


    def fit(self, X, y=None):

        # Convert input into DataFrame.
        X = pd.DataFrame(X)


        # Calculate boundaries for each feature.
        for column in X.columns:

            # First quartile.
            Q1 = X[column].quantile(0.25)

            # Third quartile.
            Q3 = X[column].quantile(0.75)

            # Interquartile range.
            IQR = Q3 - Q1

            # Calculate lower boundary.
            lower_bound = Q1 - 1.5 * IQR

            # Calculate upper boundary.
            upper_bound = Q3 + 1.5 * IQR


            # Save the learned boundaries.
            self.lower_bounds_[column] = lower_bound
            self.upper_bounds_[column] = upper_bound


        # Return fitted transformer.
        return self


    def transform(self, X):

        # Convert input into DataFrame.
        X = pd.DataFrame(X).copy()


        # Apply previously learned boundaries.
        for column in X.columns:

            X[column] = X[column].clip(
                self.lower_bounds_[column],
                self.upper_bounds_[column]
            )


        # Return transformed data.
        return X.values


# ------------------------------------------
# 5. STRATIFIED 5-FOLD CROSS-VALIDATION
# ------------------------------------------

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ------------------------------------------
# 6. DEFINE MODELS
# ------------------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=5
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        metric="euclidean"
    ),

    "SVM": SVC(
        kernel="rbf",
        C=1.0,
        random_state=42
    )
}


# ------------------------------------------
# 7. DEFINE EVALUATION METRICS
# ------------------------------------------

scoring = {

    "accuracy": "accuracy",

    "precision": make_scorer(
        precision_score,
        zero_division=0
    ),

    "recall": make_scorer(
        recall_score,
        zero_division=0
    ),

    "f1": make_scorer(
        f1_score,
        zero_division=0
    ),

    "roc_auc": "roc_auc"
}


# ------------------------------------------
# 8. STORE FINAL CV RESULTS
# ------------------------------------------

results = []


# ------------------------------------------
# 9. START CROSS-VALIDATION
# ------------------------------------------

print("\n==========================================")
print("MULTI-METRIC 5-FOLD STRATIFIED")
print("CROSS-VALIDATION")
print("==========================================")


# Loop through all models.
for model_name, model in models.items():


    # --------------------------------------
    # CREATE PIPELINE
    # --------------------------------------

    pipeline = Pipeline([

        # Missing-value handling.
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        # IQR outlier clipping.
        (
            "iqr_clipping",
            IQRClipper()
        ),

        # Feature scaling.
        (
            "scaler",
            StandardScaler()
        ),

        # Machine learning model.
        (
            "model",
            model
        )
    ])


    # --------------------------------------
    # PERFORM CROSS-VALIDATION
    # --------------------------------------

    cv_results = cross_validate(

        pipeline,

        X_clean,

        y,

        cv=skf,

        scoring=scoring,

        return_train_score=False
    )


    # --------------------------------------
    # CALCULATE MEAN METRICS
    # --------------------------------------

    mean_accuracy = cv_results[
        "test_accuracy"
    ].mean()

    mean_precision = cv_results[
        "test_precision"
    ].mean()

    mean_recall = cv_results[
        "test_recall"
    ].mean()

    mean_f1 = cv_results[
        "test_f1"
    ].mean()

    mean_roc_auc = cv_results[
        "test_roc_auc"
    ].mean()


    # Calculate standard deviation
    # of accuracy across folds.
    accuracy_std = cv_results[
        "test_accuracy"
    ].std()


    # --------------------------------------
    # DISPLAY MODEL RESULTS
    # --------------------------------------

    print("\n------------------------------------------")

    print("Model:", model_name)

    print(
        f"Mean Accuracy : {mean_accuracy:.4f}"
    )

    print(
        f"Mean Precision: {mean_precision:.4f}"
    )

    print(
        f"Mean Recall   : {mean_recall:.4f}"
    )

    print(
        f"Mean F1-Score : {mean_f1:.4f}"
    )

    print(
        f"Mean ROC-AUC  : {mean_roc_auc:.4f}"
    )

    print(
        f"Accuracy Std  : {accuracy_std:.4f}"
    )


    # --------------------------------------
    # SAVE RESULTS
    # --------------------------------------

    results.append({

        "Model": model_name,

        "CV Accuracy": mean_accuracy,

        "CV Precision": mean_precision,

        "CV Recall": mean_recall,

        "CV F1": mean_f1,

        "CV ROC-AUC": mean_roc_auc,

        "Accuracy Std": accuracy_std
    })


# ------------------------------------------
# 10. CREATE RESULTS TABLE
# ------------------------------------------

comparison_df = pd.DataFrame(results)


# Round numerical values.
comparison_df = comparison_df.round(4)


# ------------------------------------------
# 11. DISPLAY FINAL TABLE
# ------------------------------------------

print("\n==========================================")
print("CROSS-VALIDATION COMPARISON")
print("==========================================")

print(comparison_df.to_string(index=False))


# ------------------------------------------
# 12. SAVE RESULTS TO CSV
# ------------------------------------------

comparison_df.to_csv(
    "results/cross_validation_comparison.csv",
    index=False
)


# ------------------------------------------
# 13. COMPLETION MESSAGE
# ------------------------------------------

print("\n==========================================")
print("MULTI-METRIC CROSS-VALIDATION COMPLETED")
print("==========================================")

print(
    "\nResults saved to:"
    " results/cross_validation_comparison.csv"
)