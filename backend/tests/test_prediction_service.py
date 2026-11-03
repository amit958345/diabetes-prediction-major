# Import NumPy for checking numerical values and detecting
# invalid values such as NaN and infinity.
import numpy as np

# Import pytest so this file can be executed using the Pytest framework.
import pytest

# Import the Pydantic schema used to validate patient input.
from backend.app.schemas.prediction import PredictionInput

# Import prediction functions and the trained ML pipeline.
from backend.app.services.prediction_service import (
    predict_diabetes,          # Generates the final diabetes prediction.
    prepare_patient_data,      # Applies the model's preprocessing steps.
    model                      # Provides access to the trained ML pipeline.
)

# Import the function that generates individual SHAP explanations.
from backend.app.services.shap_service import explain_prediction

# Import the function that calculates global SHAP feature importance.
from backend.app.services.shap_service import explain_global_features


# Store sample patient data containing all eight required input features.
sample_data = {
    "pregnancies": 2,
    "glucose": 120,
    "blood_pressure": 70,
    "skin_thickness": 25,
    "insulin": 100,
    "bmi": 28.5,
    "diabetes_pedigree_function": 0.45,
    "age": 30
}


# Define an automated test for the complete ML prediction service.
def test_predict_diabetes_returns_valid_result():

    # Validate the sample dictionary using the Pydantic input schema.
    patient = PredictionInput.model_validate(sample_data)

    # Send the validated patient data to the trained ML model.
    result = predict_diabetes(patient)

    # Confirm that the prediction result is a Python dictionary.
    assert isinstance(result, dict)

    # Confirm that the model predicts either class 0 or class 1.
    assert result["predicted_class"] in [0, 1]

    # Confirm that the probability of class 0 is between 0 and 1.
    assert 0 <= result["probability_class_0"] <= 1

    # Confirm that the probability of class 1 is between 0 and 1.
    assert 0 <= result["probability_class_1"] <= 1

    # Confirm that the two class probabilities add up approximately to 1.
    assert abs(
        result["probability_class_0"]
        + result["probability_class_1"] - 1
    ) < 0.000001

    # Confirm that the predicted class probability is between 0 and 1.
    assert 0 <= result["predicted_class_probability"] <= 1

    # Confirm that the predicted class probability matches its corresponding class.
    if result["predicted_class"] == 0:
        assert result["predicted_class_probability"] == pytest.approx(
            result["probability_class_0"]
        )
    else:
        assert result["predicted_class_probability"] == pytest.approx(
            result["probability_class_1"]
        )


# ============================================================
# SHAP EXPLANATION CONSISTENCY TEST
# ============================================================

# Verify that SHAP contributions reconstruct the actual
# Logistic Regression model output.
def test_shap_explanation_matches_model_output():

    # Validate the sample patient data using the Pydantic schema.
    patient = PredictionInput.model_validate(sample_data)

    # Apply the exact preprocessing steps used by the trained model.
    transformed_input = prepare_patient_data(patient)

    # Access the Logistic Regression classifier from the pipeline.
    classifier = model.named_steps["model"]

    # Calculate the actual model output in log-odds.
    # This calculation is independent of the SHAP explanation.
    actual_model_output = float(
        classifier.decision_function(transformed_input)[0]
    )

    # Generate the SHAP explanation for the same patient.
    explanation = explain_prediction(patient)

    # Reconstruct the model output using the SHAP baseline
    # and the sum of all individual feature contributions.
    reconstructed_output = (
        explanation["base_value"]
        + sum(explanation["feature_contributions"].values())
    )

    # Verify that the SHAP-reconstructed output matches
    # the actual Logistic Regression output.
    assert reconstructed_output == pytest.approx(
        actual_model_output,
        abs=0.000001
    )

    # Verify that the model output returned by the SHAP service
    # also matches the independently calculated model output.
    assert explanation["model_output"] == pytest.approx(
        actual_model_output,
        abs=0.000001
    )


# ============================================================
# SHAP FEATURE CONTRIBUTION VALIDATION TEST
# ============================================================

# Verify that the SHAP explanation contains valid contributions
# for all eight input features.
def test_shap_explanation_contains_valid_feature_contributions():

    # Validate the sample patient data.
    patient = PredictionInput.model_validate(sample_data)

    # Generate the individual SHAP explanation.
    explanation = explain_prediction(patient)

    # Define the exact feature names expected by the ML model.
    expected_features = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ]

    # Extract the feature contributions from the explanation.
    contributions = explanation["feature_contributions"]

    # Verify that all eight expected features are present.
    assert set(contributions.keys()) == set(expected_features)

    # Verify that exactly eight feature contributions exist.
    assert len(contributions) == 8

    # Verify that every feature contribution is a finite number.
    assert all(
        np.isfinite(value)
        for value in contributions.values()
    )

    # Verify that the baseline value is finite.
    assert np.isfinite(explanation["base_value"])

    # Verify that the reconstructed model output is finite.
    assert np.isfinite(explanation["model_output"])


# ============================================================
# GLOBAL SHAP FEATURE IMPORTANCE VALIDATION TEST
# ============================================================

# Verify that global SHAP importance contains valid,
# non-negative and correctly sorted feature values.
def test_global_shap_importance_contains_valid_results():

    # Generate global SHAP feature importance.
    global_explanation = explain_global_features()

    # Define the eight expected ML input features.
    expected_features = {
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    }

    # Extract the feature importance dictionary.
    importance = global_explanation["feature_importance"]

    # Verify that all 768 dataset records were analysed.
    assert global_explanation["total_records_analysed"] == 768

    # Verify that all eight expected features are present.
    assert set(importance.keys()) == expected_features

    # Verify that all importance values are finite numbers.
    assert all(
        np.isfinite(value)
        for value in importance.values()
    )

    # Verify that no feature has negative importance.
    assert all(
        value >= 0
        for value in importance.values()
    )

    # Extract importance values in their returned order.
    importance_values = list(importance.values())

    # Verify that importance is sorted from highest to lowest.
    assert importance_values == sorted(
        importance_values,
        reverse=True
    )