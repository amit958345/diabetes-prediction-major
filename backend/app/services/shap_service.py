
# Import NumPy for numerical calculations and array operations.
import numpy as np

# Import Pandas for loading and processing structured datasets.
import pandas as pd

# Import SHAP for explainable AI calculations.
import shap

# Import lru_cache to avoid repeating expensive SHAP calculations.
from functools import lru_cache

# Import the existing prediction model, feature names,
# and shared preprocessing helper from the prediction service.
from backend.app.services.prediction_service import (
    model,
    feature_names,
    prepare_patient_data
)

# Import the validated patient input schema.
from backend.app.schemas.prediction import PredictionInput


# ============================================================
# PREPARE SHAP BACKGROUND DATASET
# ============================================================

# Load the original diabetes dataset.
df = pd.read_csv("data/diabetes.csv")

# Select the eight input features without modifying the original dataset.
X = df[feature_names].copy()

# Identify columns where zero represents missing data.
missing_value_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Replace zero values with NaN so the preprocessing pipeline
# can handle them as missing observations.
X[missing_value_columns] = X[missing_value_columns].replace(
    0,
    np.nan
)

# Apply the preprocessing steps from the trained ML pipeline.
# This ensures SHAP receives data in the same format as the model.
transformed_data = model[:-1].transform(X)

# Create a DataFrame using the original feature names.
background_data = pd.DataFrame(
    transformed_data,
    columns=feature_names
)

# Select 100 reproducible background records for SHAP.
background_data = background_data.sample(
    n=100,
    random_state=42
)

# Extract the Logistic Regression classifier from the pipeline.
classifier = model.named_steps["model"]

# Initialize the SHAP LinearExplainer using the classifier
# and the selected background dataset.
explainer = shap.LinearExplainer(
    classifier,
    background_data
)


# ============================================================
# CACHE FULL-DATASET SHAP CALCULATIONS
# ============================================================

@lru_cache(maxsize=1)
def get_global_shap_values() -> np.ndarray:
    """
    Calculate and cache SHAP values for the complete dataset.

    The first request performs the expensive SHAP calculation.
    Subsequent requests reuse the cached NumPy array.

    Both global feature importance and beeswarm data use
    this shared result.

    Returns:
        Two-dimensional NumPy array containing SHAP values.
        Rows represent records and columns represent features.
    """

    # Prepare the complete transformed dataset.
    global_data = pd.DataFrame(
        transformed_data,
        columns=feature_names
    )

    # Calculate SHAP explanations for the complete dataset.
    global_explanations = explainer(global_data)

    # Extract the SHAP contribution matrix.
    shap_matrix = np.asarray(
        global_explanations.values
    )

    # Verify that SHAP returned a two-dimensional matrix.
    if shap_matrix.ndim != 2:
        raise ValueError(
            "Expected a two-dimensional SHAP matrix, "
            f"but received shape {shap_matrix.shape}."
        )

    # Verify that the number of SHAP columns matches
    # the number of model input features.
    if shap_matrix.shape[1] != len(feature_names):
        raise ValueError(
            "SHAP feature count does not match model feature count."
        )

    # Return the calculated SHAP values.
    # lru_cache stores this result for subsequent requests.
    return shap_matrix


# ============================================================
# GENERATE INDIVIDUAL PATIENT SHAP EXPLANATION
# ============================================================

def explain_prediction(patient_data: PredictionInput) -> dict:
    """
    Generate SHAP feature contributions for one patient.

    Parameters:
        patient_data: Validated patient information.

    Returns:
        Dictionary containing:
        - SHAP baseline value.
        - Model output in log-odds.
        - Individual feature contributions.
    """

    # Apply the same preprocessing used by the trained pipeline.
    transformed_input = prepare_patient_data(patient_data)

    # Generate the SHAP explanation for the patient.
    explanation = explainer(transformed_input)

    # Extract SHAP contributions for the first patient.
    shap_values = explanation.values[0]

    # Extract the baseline model output.
    base_value = float(explanation.base_values[0])

    # Reconstruct the model output using SHAP additivity.
    # This is a log-odds score, not a probability.
    model_output = float(
        base_value + np.sum(shap_values)
    )

    # Associate every feature name with its contribution.
    feature_contributions = {
        feature: float(value)
        for feature, value in zip(feature_names, shap_values)
    }

    # Return the individual explanation.
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
    Calculate global SHAP feature importance.

    Global importance is the mean absolute SHAP contribution
    of each feature across all analysed dataset records.

    The full-dataset SHAP values are retrieved from the cache.

    Important:
        This analysis uses the same dataset used to train
        the model, not an independent test dataset.
    """

    # Retrieve cached SHAP values instead of recalculating them.
    shap_values = get_global_shap_values()

    # Calculate mean absolute SHAP contribution per feature.
    mean_absolute_shap = np.mean(
        np.abs(shap_values),
        axis=0
    )

    # Associate feature names with their importance values.
    feature_importance = {
        feature: float(importance)
        for feature, importance in zip(
            feature_names,
            mean_absolute_shap
        )
    }

    # Sort the features from highest to lowest importance.
    sorted_importance = dict(
        sorted(
            feature_importance.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )

    # Return the global feature importance results.
    return {
        "total_records_analysed": len(shap_values),
        "feature_importance": sorted_importance
    }


# ============================================================
# GENERATE DATA FOR SHAP BEESWARM VISUALIZATION
# ============================================================

def explain_beeswarm_data() -> dict:
    """
    Generate per-record SHAP contributions for a beeswarm plot.

    Each record contains:
        - SHAP values for all eight features.
        - Original feature values for all eight features.

    SHAP values are retrieved from the shared cache.

    Original feature values are returned in their original
    measurement units so the visualization can use meaningful
    feature values for its low-to-high colour scale.

    Missing original measurements are returned as None.

    Important:
        The current analysis uses the training dataset.
        It describes model behaviour on that dataset and should
        not be interpreted as independent test-set explanation
        or clinical validation.
    """

    # Retrieve the shared cached SHAP contribution matrix.
    shap_matrix = get_global_shap_values()

    # Convert the original feature measurements into a list.
    # These values remain in their original measurement units.
    # Missing values are preserved as None instead of zero.
    original_feature_values = [
        [
            None if pd.isna(value) else float(value)
            for value in row
        ]
        for row in X.to_numpy()
    ]

    # Convert the SHAP matrix into ordinary Python lists.
    # Python floats are JSON-compatible and can be sent by FastAPI.
    shap_value_records = [
        [
            float(value)
            for value in row
        ]
        for row in shap_matrix
    ]

    # Calculate mean absolute SHAP importance for each feature.
    mean_absolute_shap = np.mean(
        np.abs(shap_matrix),
        axis=0
    )

    # Associate every feature with its mean absolute SHAP value.
    feature_importance = {
        feature: float(importance)
        for feature, importance in zip(
            feature_names,
            mean_absolute_shap
        )
    }

    # Sort the feature importance values in descending order.
    sorted_importance = dict(
        sorted(
            feature_importance.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )

    # Return all information required by the beeswarm frontend.
    return {
        "total_records_analysed": len(shap_value_records),

        # Provide the original feature names in model input order.
        "feature_names": list(feature_names),

        # Provide the per-record SHAP contribution matrix.
        "shap_values": shap_value_records,

        # Provide the corresponding original feature measurements.
        "feature_values": original_feature_values,

        # Provide the global importance ranking.
        "feature_importance": sorted_importance,

        # Explain where these analysed records came from.
        "data_source": "Training dataset",

        # Clarify the interpretation and limitation of the results.
        "note": (
            "SHAP values describe the trained model's behaviour "
            "on the training dataset. They do not establish "
            "causation or clinical validity."
        )
    }
