
# Import pytest to create and run automated tests.
import pytest

# Import ValidationError to identify invalid Pydantic input.
from pydantic import ValidationError

# Import the PredictionInput schema that validates patient data.
from backend.app.schemas.prediction import PredictionInput


# Store common valid patient data.
# Each test will use this data as its starting point.
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


# Run this same test separately for each supplied value.
@pytest.mark.parametrize(
    "invalid_glucose",
    [
        float("nan"),   # Not a Number
        float("inf"),   # Positive infinity
        float("-inf")   # Negative infinity
    ]
)
def test_non_finite_glucose_rejected(invalid_glucose):

    # Make a separate copy of the valid patient data.
    invalid_data = valid_data.copy()

    # Replace the valid glucose value with the invalid value.
    invalid_data["glucose"] = invalid_glucose

    # Confirm that Pydantic rejects this invalid input.
    with pytest.raises(ValidationError):

        # Attempt to validate the patient data.
        # A ValidationError is expected.
        PredictionInput.model_validate(invalid_data)
