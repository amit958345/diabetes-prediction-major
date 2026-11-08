
from backend.app.services.prediction_service import (
    model,
    prepare_patient_data,
    predict_diabetes,
    CLASSIFICATION_THRESHOLD
)

from backend.app.services.shap_service import explain_prediction
from backend.app.schemas.prediction import PredictionInput

import numpy as np


# Create sample patient data
patient = PredictionInput(
    pregnancies=2,
    glucose=120,
    blood_pressure=70,
    skin_thickness=25,
    insulin=80,
    bmi=28.5,
    diabetes_pedigree_function=0.5,
    age=35
)


# Generate prediction
result = predict_diabetes(patient)


# Generate SHAP explanation
shap_result = explain_prediction(patient)


# Transform patient data using the model pipeline
transformed = prepare_patient_data(patient)


# Extract Logistic Regression classifier
classifier = model.named_steps["model"]


# Calculate the model's decision score
decision_score = float(
    classifier.decision_function(transformed)[0]
)


# Get the SHAP reconstructed score
shap_reconstructed = shap_result["model_output"]


# Calculate difference
difference = abs(decision_score - shap_reconstructed)


# Verify threshold classification
expected_class = (
    1
    if result["probability_class_1"] >= CLASSIFICATION_THRESHOLD
    else 0
)

threshold_check = result["predicted_class"] == expected_class


# Display results
print("\n========== MODEL VALIDATION TEST ==========")

print("\n--- Prediction Results ---")
print("Predicted class:", result["predicted_class"])
print("Class 1 probability:", result["probability_class_1"])
print("Classification threshold:", CLASSIFICATION_THRESHOLD)
print("Expected class:", expected_class)
print("Threshold verification:", threshold_check)


print("\n--- SHAP Consistency Results ---")
print("Decision score:", decision_score)
print("SHAP reconstructed score:", shap_reconstructed)
print("Difference:", difference)

shap_check = np.isclose(
    decision_score,
    shap_reconstructed,
    atol=1e-5
)

print("SHAP verification:", shap_check)


print("\n--- Final Test Status ---")

if threshold_check and shap_check:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")

print("===========================================\n")
