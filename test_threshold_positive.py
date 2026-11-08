
from backend.app.services.prediction_service import (
    predict_diabetes,
    CLASSIFICATION_THRESHOLD
)

from backend.app.schemas.prediction import PredictionInput


# Candidate test inputs (synthetic values)
test_patients = [
    PredictionInput(
        pregnancies=6, glucose=148, blood_pressure=72,
        skin_thickness=35, insulin=120, bmi=33.6,
        diabetes_pedigree_function=0.627, age=50
    ),
    PredictionInput(
        pregnancies=8,
        glucose=183,
        blood_pressure=64,
        skin_thickness=20,
        insulin=100,
        bmi=23.3,
        diabetes_pedigree_function=0.672,
        age=32
),
    PredictionInput(
        pregnancies=1, glucose=189, blood_pressure=60,
        skin_thickness=23, insulin=846, bmi=30.1,
        diabetes_pedigree_function=0.398, age=59
    ),
    PredictionInput(
        pregnancies=5, glucose=166, blood_pressure=72,
        skin_thickness=19, insulin=175, bmi=25.8,
        diabetes_pedigree_function=0.587, age=51
    ),
    PredictionInput(
        pregnancies=10, glucose=168, blood_pressure=74,
        skin_thickness=25, insulin=150, bmi=38.0,
        diabetes_pedigree_function=0.537, age=34
    ),
]


print("\n========== THRESHOLD POSITIVE TEST ==========")

found_positive = False

for index, patient in enumerate(test_patients, start=1):

    result = predict_diabetes(patient)

    probability = result["probability_class_1"]
    predicted_class = result["predicted_class"]

    print(f"\nTest patient {index}")
    print(f"Class 1 probability: {probability:.4f}")
    print(f"Threshold: {CLASSIFICATION_THRESHOLD}")
    print(f"Predicted class: {predicted_class}")

    if probability >= CLASSIFICATION_THRESHOLD:
        found_positive = True

        if predicted_class == 1:
            print("Threshold verification: PASSED")
        else:
            print("Threshold verification: FAILED")

        break


print("\n---------------------------------------------")

if found_positive:
    print("POSITIVE THRESHOLD TEST PASSED")
else:
    print("No candidate crossed the threshold.")
    print("We will generate additional test inputs.")

print("=============================================\n")
