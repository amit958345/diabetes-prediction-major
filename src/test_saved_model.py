# ==========================================
# TEST SAVED DIABETES PREDICTION MODEL
# ==========================================

import pandas as pd
import joblib


# ------------------------------------------
# 1. LOAD THE SAVED MODEL
# ------------------------------------------

model = joblib.load(
    "models/diabetes_model.pkl"
)


# ------------------------------------------
# 2. CREATE SAMPLE PATIENT DATA
# ------------------------------------------

# The values correspond to the 8 features
# used during model training.

sample_patient = pd.DataFrame({

    "Pregnancies": [2],

    "Glucose": [120],

    "BloodPressure": [70],

    "SkinThickness": [25],

    "Insulin": [80],

    "BMI": [30.5],

    "DiabetesPedigreeFunction": [0.45],

    "Age": [35]
})


# ------------------------------------------
# 3. DISPLAY INPUT
# ------------------------------------------

print("\n==========================================")
print("SAMPLE PATIENT INPUT")
print("==========================================")

print(sample_patient)


# ------------------------------------------
# 4. MAKE PREDICTION
# ------------------------------------------

prediction = model.predict(
    sample_patient
)


# ------------------------------------------
# 5. GET PREDICTION PROBABILITY
# ------------------------------------------

probability = model.predict_proba(
    sample_patient
)


# Probability of class 0.
probability_no_diabetes = probability[0][0]


# Probability of class 1.
probability_diabetes = probability[0][1]


# ------------------------------------------
# 6. DISPLAY RESULT
# ------------------------------------------

print("\n==========================================")
print("PREDICTION RESULT")
print("==========================================")


if prediction[0] == 1:

    print(
        "Prediction: Diabetes Risk Detected"
    )

else:

    print(
        "Prediction: Lower Diabetes Risk"
    )


print(
    f"Probability of Class 0: "
    f"{probability_no_diabetes:.2%}"
)


print(
    f"Probability of Class 1: "
    f"{probability_diabetes:.2%}"
)


# ------------------------------------------
# 7. COMPLETION MESSAGE
# ------------------------------------------

print("\n==========================================")
print("SAVED MODEL TEST COMPLETED")
print("==========================================")