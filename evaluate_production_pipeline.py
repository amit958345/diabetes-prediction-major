
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
    confusion_matrix,
    classification_report
)

from src.iqr_clipper import IQRClipper


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/diabetes.csv")

X = df.drop("Outcome", axis=1).copy()
y = df["Outcome"]

# Replace zero values representing missing measurements.
missing_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

X[missing_columns] = X[missing_columns].replace(0, float("nan"))


# ==========================================
# 2. SPLIT DATASET
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
# 3. BUILD THE SAME MODEL PIPELINE
# ==========================================

evaluation_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("iqr", IQRClipper()),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(random_state=42))
])


# ==========================================
# 4. TRAIN TEMPORARY MODEL
# ==========================================

evaluation_model.fit(X_train, y_train)

print("\nTemporary evaluation model trained successfully.")


# ==========================================
# 5. GENERATE TEST PREDICTIONS
# ==========================================

y_pred = evaluation_model.predict(X_test)

y_probability = evaluation_model.predict_proba(X_test)[:, 1]


# ==========================================
# 6. CALCULATE PERFORMANCE METRICS
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

roc_auc = roc_auc_score(y_test, y_probability)


# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print("\n========== HELD-OUT TEST RESULTS ==========")

print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))


# ==========================================
# 8. SAVE RESULTS
# ==========================================

results = pd.DataFrame([{
    "Model": "Logistic Regression",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "ROC-AUC": roc_auc,
    "Training Records": len(X_train),
    "Testing Records": len(X_test)
}])

results.to_csv(
    "results/production_pipeline_evaluation.csv",
    index=False
)

print("\nResults saved to:")
print("results/production_pipeline_evaluation.csv")

print("\nEVALUATION COMPLETED SUCCESSFULLY")
