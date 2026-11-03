from pathlib import Path
import sys

import joblib
import pandas as pd


# Find the main project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Add the src directory to Python's module search path
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))


# Load the saved ML model and feature names
MODEL_PATH = PROJECT_ROOT / "models" / "diabetes_model.pkl"
FEATURE_PATH = PROJECT_ROOT / "models" / "feature_names.pkl"

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_PATH)


# Create sample patient data
sample_data = pd.DataFrame(
    [[2, 120, 70, 25, 100, 28.5, 0.45, 30]],
    columns=feature_names
)


# Replace invalid zero measurements with missing values
invalid_zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

sample_data[invalid_zero_columns] = sample_data[
    invalid_zero_columns
].replace(0, float("nan"))


# Generate the predicted class
prediction = model.predict(sample_data)[0]


# Generate class probabilities
probabilities = model.predict_proba(sample_data)[0]


# Display prediction results
print("ML PREDICTION TEST SUCCESSFUL")

print("Predicted Class:", prediction)

print("Probability of Class 0:", probabilities[0])
print("Probability of Class 1:", probabilities[1])

print("Predicted Class Probability:", probabilities[prediction])