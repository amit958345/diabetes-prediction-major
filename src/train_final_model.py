# ==========================================
# DIABETES PREDICTION USING MACHINE LEARNING
# FINAL MODEL TRAINING
# ==========================================

import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

# Import our custom IQR transformer.
from iqr_clipper import IQRClipper


# ------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------

df = pd.read_csv("data/diabetes.csv")


# ------------------------------------------
# 2. SEPARATE FEATURES AND TARGET
# ------------------------------------------

X = df.drop("Outcome", axis=1)

y = df["Outcome"]


# ------------------------------------------
# 3. REPLACE INVALID ZERO VALUES
# ------------------------------------------

columns_with_invalid_zero = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]


X[columns_with_invalid_zero] = (
    X[columns_with_invalid_zero]
    .replace(0, np.nan)
)


# ------------------------------------------
# 4. CREATE FINAL PIPELINE
# ------------------------------------------

pipeline = Pipeline([

    # Step 1: Median imputation.
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),

    # Step 2: IQR outlier clipping.
    (
        "iqr_clipping",
        IQRClipper()
    ),

    # Step 3: Standard scaling.
    (
        "scaler",
        StandardScaler()
    ),

    # Step 4: Selected classification model.
    (
        "model",
        LogisticRegression(
            random_state=42
        )
    )
])


# ------------------------------------------
# 5. TRAIN FINAL MODEL
# ------------------------------------------

print("\n==========================================")
print("FINAL MODEL TRAINING")
print("==========================================")


print("\nSelected Model: Logistic Regression")

print("\nTraining on complete dataset...")

pipeline.fit(X, y)


# ------------------------------------------
# 6. SAVE COMPLETE PIPELINE
# ------------------------------------------

joblib.dump(
    pipeline,
    "models/diabetes_model.pkl"
)


# ------------------------------------------
# 7. SAVE FEATURE NAMES
# ------------------------------------------

feature_names = list(X.columns)

joblib.dump(
    feature_names,
    "models/feature_names.pkl"
)


# ------------------------------------------
# 8. DISPLAY INFORMATION
# ------------------------------------------

print("\nModel trained successfully.")

print(
    "\nNumber of training records:",
    len(X)
)

print(
    "Number of input features:",
    X.shape[1]
)

print(
    "\nFeatures:"
)

for feature in feature_names:

    print(
        " -",
        feature
    )


print("\nModel saved to:")

print(
    "models/diabetes_model.pkl"
)


print("\nFeature names saved to:")

print(
    "models/feature_names.pkl"
)


print("\n==========================================")
print("FINAL MODEL TRAINING COMPLETED")
print("==========================================")