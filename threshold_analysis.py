
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict
)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from src.iqr_clipper import IQRClipper


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/diabetes.csv")

X = df.drop("Outcome", axis=1).copy()
y = df["Outcome"]

missing_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

X[missing_columns] = X[missing_columns].replace(
    0, float("nan")
)


# ==========================================
# 2. CREATE TRAINING AND TESTING SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== DATASET SPLIT ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 3. BUILD PRODUCTION TRAINING PIPELINE
# ==========================================

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("iqr", IQRClipper()),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(random_state=42))
])


# ==========================================
# 4. GENERATE OUT-OF-FOLD PROBABILITIES
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nGenerating training-only out-of-fold predictions...")

oof_probabilities = cross_val_predict(
    model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba"
)[:, 1]

print("Out-of-fold predictions generated successfully.")


# ==========================================
# 5. DEFINE THRESHOLDS
# ==========================================

thresholds = [
    0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70
]


# ==========================================
# 6. COMPARE THRESHOLD PERFORMANCE
# ==========================================

results = []

for threshold in thresholds:

    # Convert probabilities into class predictions.
    y_pred = (
        oof_probabilities >= threshold
    ).astype(int)

    # Calculate confusion matrix.
    tn, fp, fn, tp = confusion_matrix(
        y_train,
        y_pred,
        labels=[0, 1]
    ).ravel()

    # Calculate evaluation metrics.
    accuracy = accuracy_score(y_train, y_pred)

    precision = precision_score(
        y_train,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_train,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_train,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_train,
        oof_probabilities
    )

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })


# ==========================================
# 7. DISPLAY COMPARISON TABLE
# ==========================================

results_df = pd.DataFrame(results)

print("\n========== TRAINING THRESHOLD ANALYSIS ==========")

print(
    results_df.round(4).to_string(index=False)
)


# ==========================================
# 8. SAVE RESULTS
# ==========================================

results_df.to_csv(
    "results/threshold_analysis_training.csv",
    index=False
)

print("\nResults saved to:")
print("results/threshold_analysis_training.csv")

print("\nTHRESHOLD ANALYSIS COMPLETED SUCCESSFULLY")
