
# Import pytest so this file can be executed using the Pytest framework.
import pytest

# Import the Pydantic schema used to validate patient input.
from backend.app.schemas.prediction import PredictionInput

# Import the function that generates predictions using our trained ML model.
from backend.app.services.prediction_service import predict_diabetes


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
