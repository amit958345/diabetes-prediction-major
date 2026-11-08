
import pandas as pd

from sklearn.model_selection import train_test_split
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
# 2. TRAIN-TEST SPLIT
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
# 3. BUILD PRODUCTION-STYLE PIPELINE
# ==========================================

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("iqr", IQRClipper()),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(random_state=42))
])


# ==========================================
# 4. TRAIN TEMPORARY MODEL
# ==========================================

print("\nTraining temporary evaluation model...")

model.fit(X_train, y_train)

print("Model training completed.")


# ==========================================
# 5. GENERATE TEST PROBABILITIES
# ==========================================

test_probabilities = model.predict_proba(X_test)[:, 1]

print("\nTest probabilities generated successfully.")


# ==========================================
# 6. EVALUATE TWO THRESHOLDS
# ==========================================

thresholds = [0.35, 0.50]

results = []

for threshold in thresholds:

    y_pred = (
        test_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    roc_auc = roc_auc_score(
        y_test,
        test_probabilities
    )

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Specificity": specificity,
        "ROC-AUC": roc_auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })


# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

results_df = pd.DataFrame(results)

print("\n========== HELD-OUT TEST RESULTS ==========")

print(
    results_df.round(4).to_string(index=False)
)


# ==========================================
# 8. SAVE RESULTS
# ==========================================

results_df.to_csv(
    "results/selected_threshold_test_evaluation.csv",
    index=False
)

print("\nResults saved to:")
print("results/selected_threshold_test_evaluation.csv")

print("\nTHRESHOLD TEST EVALUATION COMPLETED")
