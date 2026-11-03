
# Import pytest to write and execute automated tests.
import pytest

# Import Pydantic's ValidationError to check invalid input.
from pydantic import ValidationError

# Import the schema that we want to test.
from backend.app.schemas.prediction import PredictionInput


# ============================================================
# COMMON TEST DATA
# ============================================================
# This dictionary contains valid patient input.
# We reuse it across multiple test functions.

valid_data = {
    "pregnancies": 2,
    "glucose": 120,
    "blood_pressure": 70,
    "skin_thickness": 25,
    "insulin": 100,
    "bmi": 28.5,
    "diabetes_pedigree_function": 0.45,
    "age": 30
}


# ============================================================
# TEST 1: VALID PREDICTION INPUT
# ============================================================
# Verify that valid patient data is accepted by Pydantic.

def test_valid_prediction_input():

    # Validate the dictionary using the PredictionInput schema.
    patient = PredictionInput.model_validate(valid_data)

    # Confirm that the glucose value was accepted correctly.
    assert patient.glucose == 120

    # Confirm that the BMI value was accepted correctly.
    assert patient.bmi == 28.5


# ============================================================
# TEST 2: NEGATIVE GLUCOSE VALUE
# ============================================================
# Verify that negative glucose values are rejected.

def test_negative_glucose_rejected():

    # Create a copy so the original valid data remains unchanged.
    invalid_data = valid_data.copy()

    # Replace the glucose value with an invalid negative value.
    invalid_data["glucose"] = -10

    # Confirm that Pydantic raises a validation error.
    with pytest.raises(ValidationError):
        PredictionInput.model_validate(invalid_data)


# ============================================================
# TEST 3: ZERO BMI
# ============================================================
# Verify that zero BMI is rejected by the schema.

def test_zero_bmi_rejected():

    # Create a copy of the valid patient data.
    invalid_data = valid_data.copy()

    # Replace BMI with zero.
    invalid_data["bmi"] = 0

    # Confirm that Pydantic rejects the invalid value.
    with pytest.raises(ValidationError):
        PredictionInput.model_validate(invalid_data)
