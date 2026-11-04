from pathlib import Path
import sys

import joblib
import pandas as pd

from backend.app.schemas.prediction import PredictionInput


# Find the main project directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]


# Add the src directory to Python's module search path
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))


# Define the saved model file paths
MODEL_PATH = PROJECT_ROOT / "models" / "diabetes_model.pkl"
FEATURE_PATH = PROJECT_ROOT / "models" / "feature_names.pkl"


# Load the trained ML pipeline and expected feature names
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_PATH)


# Prepare the original, untransformed patient data.
def prepare_raw_patient_data(patient_data: PredictionInput) -> pd.DataFrame:

    # Map API field names to the original feature names
    # expected by the trained machine learning model.
    feature_values = {
        "Pregnancies": patient_data.pregnancies,
        "Glucose": patient_data.glucose,
        "BloodPressure": patient_data.blood_pressure,
        "SkinThickness": patient_data.skin_thickness,
        "Insulin": patient_data.insulin,
        "BMI": patient_data.bmi,
        "DiabetesPedigreeFunction": patient_data.diabetes_pedigree_function,
        "Age": patient_data.age
    }

    # Arrange patient values in the exact feature order
    # used when the original model was trained.
    raw_input = pd.DataFrame(
        [[feature_values[name] for name in feature_names]],
        columns=feature_names
    )

    # Return raw values without applying preprocessing.
    return raw_input


# Prepare transformed patient data for SHAP explanations.
def prepare_patient_data(patient_data: PredictionInput) -> pd.DataFrame:

    # First create the original, untransformed patient DataFrame.
    raw_input = prepare_raw_patient_data(patient_data)

    # Apply the preprocessing steps exactly once.
    # model[:-1] excludes the final Logistic Regression classifier.
    transformed_data = model[:-1].transform(raw_input)

    # Preserve the original feature names for SHAP processing.
    return pd.DataFrame(
        transformed_data,
        columns=feature_names
    )


# Generate a diabetes prediction from validated patient data.
def predict_diabetes(patient_data: PredictionInput) -> dict:

    # Prepare the original patient input without preprocessing it manually.
    raw_input = prepare_raw_patient_data(patient_data)

    # Let the complete trained pipeline preprocess the raw input
    # and generate the predicted class exactly once.
    predicted_class = int(model.predict(raw_input)[0])

    # Let the same complete pipeline calculate class probabilities.
    probabilities = model.predict_proba(raw_input)[0]

    # Associate each probability with its corresponding class label.
    class_probabilities = {
        int(class_label): float(probability)
        for class_label, probability in zip(model.classes_, probabilities)
    }

    # Return the prediction results and the probability of the predicted class.
    return {
        "predicted_class": predicted_class,
        "probability_class_0": class_probabilities[0],
        "probability_class_1": class_probabilities[1],
        "predicted_class_probability": class_probabilities[predicted_class]
    }
