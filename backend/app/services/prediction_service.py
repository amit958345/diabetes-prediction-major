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

# Prepare patient data for model prediction and SHAP explanation.
def prepare_patient_data(patient_data: PredictionInput) -> pd.DataFrame:

    # Map API field names to the original ML feature names.
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

    # Arrange values in the exact feature order expected by the model.
    input_data = pd.DataFrame(
        [[feature_values[name] for name in feature_names]],
        columns=feature_names
    )

    # Apply preprocessing without running the Logistic Regression model.
    transformed_data = model[:-1].transform(input_data)

    # Return the transformed data with the original feature names.
    return pd.DataFrame(
        transformed_data,
        columns=feature_names
    )


# Generate a diabetes prediction from validated patient data.
def predict_diabetes(patient_data: PredictionInput) -> dict:

    # Prepare the patient data using the shared preprocessing helper.
    input_data = prepare_patient_data(patient_data)

    # Generate the predicted class using the complete ML pipeline.
    predicted_class = int(model.predict(input_data)[0])

    # Generate probabilities for each class.
    probabilities = model.predict_proba(input_data)[0]

    # Associate each probability with its corresponding class.
    class_probabilities = {
        int(class_label): float(probability)
        for class_label, probability in zip(model.classes_, probabilities)
    }

    # Return the prediction results.
    return {
        "predicted_class": predicted_class,
        "probability_class_0": class_probabilities[0],
        "probability_class_1": class_probabilities[1],
        "predicted_class_probability": class_probabilities[predicted_class]
    }