# Import NumPy for numerical calculations.
import numpy as np

# Import Pandas for creating structured datasets.
import pandas as pd

# Import SHAP for explainable AI calculations.
import shap

# Import the existing prediction model, feature names,
# and shared preprocessing helper from the prediction service.
from backend.app.services.prediction_service import (
    model,                   # Complete trained ML pipeline.
    feature_names,            # Original eight feature names in the expected order.
    prepare_patient_data     # Preprocesses validated patient data for SHAP.
)

# Import the validated patient input schema.
from backend.app.schemas.prediction import PredictionInput

# ============================================================
# PREPARE SHAP BACKGROUND DATASET
# ============================================================

# Load the original diabetes dataset.
df = pd.read_csv("data/diabetes.csv")

# Select the eight input features.
X = df[feature_names].copy()

# Identify columns where zero represents missing data.
missing_value_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Replace zero values with NaN.
X[missing_value_columns] = X[missing_value_columns].replace(
    0,
    np.nan
)

# Apply the existing preprocessing pipeline.
transformed_data = model[:-1].transform(X)

# Create a DataFrame with the original feature names.
background_data = pd.DataFrame(
    transformed_data,
    columns=feature_names
)

# Select 100 reproducible background records.
background_data = background_data.sample(
    n=100,
    random_state=42
)

# Get the Logistic Regression classifier from the trained pipeline.
classifier = model.named_steps["model"]

# Initialize the SHAP LinearExplainer.
explainer = shap.LinearExplainer(
    classifier,
    background_data
)

# ============================================================
# GENERATE INDIVIDUAL PATIENT SHAP EXPLANATION
# ============================================================

def explain_prediction(patient_data: PredictionInput) -> dict:
    """
    Generate SHAP feature contributions for one patient.

    Parameters:
        patient_data: Validated patient information received
                      through the PredictionInput schema.

    Returns:
        Dictionary containing:
        - SHAP baseline value.
        - Model output (log-odds score).
        - Individual feature contributions.
    """

    # Apply the same preprocessing used by the trained ML pipeline.
    # This ensures SHAP receives exactly the same transformed features.
    transformed_input = prepare_patient_data(patient_data)

    # Generate SHAP values for the transformed patient data.
    explanation = explainer(transformed_input)

    # Extract the SHAP contributions for the first patient.
    shap_values = explanation.values[0]

    # Extract the baseline model output calculated by SHAP.
    base_value = float(explanation.base_values[0])

    # Reconstruct the model output using the SHAP additive property.
    # Model output is a log-odds score, not a probability.
    model_output = float(
        base_value + np.sum(shap_values)
    )

    # Associate each original feature name with its SHAP contribution.
    # Convert NumPy values into Python floats for JSON compatibility.
    feature_contributions = {
        feature: float(value)
        for feature, value in zip(feature_names, shap_values)
    }

    # Return the complete explanation for the patient.
    return {
        "base_value": base_value,
        "model_output": model_output,
        "feature_contributions": feature_contributions
    }

# ============================================================
# GENERATE GLOBAL SHAP FEATURE IMPORTANCE
# ============================================================

def explain_global_features() -> dict:
    """
    Calculate global SHAP feature importance across the dataset.

    Global importance represents the average absolute SHAP
    contribution of each feature across all analysed records.

    Returns:
        Dictionary containing:
        - Total number of analysed records.
        - Mean absolute SHAP importance for each feature.
        - Features sorted from highest to lowest importance.

    Note:
        The current dataset is the same dataset used to train
        the model. Therefore, these results describe the trained
        model's behaviour on its training dataset, not independent
        test-set performance.
    """

    # Prepare the complete transformed dataset using the same
    # preprocessing pipeline used during model training.
    global_data = pd.DataFrame(
        transformed_data,
        columns=feature_names
    )

    # Generate SHAP explanations for all dataset records.
    global_explanations = explainer(global_data)

    # Extract the SHAP contribution matrix.
    # Rows represent records and columns represent features.
    shap_values = global_explanations.values

    # Calculate the average absolute SHAP contribution
    # separately for each feature.
    mean_absolute_shap = np.mean(
        np.abs(shap_values),
        axis=0
    )

    # Associate each feature name with its importance value.
    feature_importance = {
        feature: float(importance)
        for feature, importance in zip(
            feature_names,
            mean_absolute_shap
        )
    }

    # Sort features from highest to lowest importance.
    sorted_importance = dict(
        sorted(
            feature_importance.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )

    # Return the global explanation results.
    return {
        "total_records_analysed": len(global_data),
        "feature_importance": sorted_importance
    }