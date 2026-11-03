# Import FastAPI components for creating API routes,
# dependency injection, HTTP errors and status codes.
from fastapi import APIRouter, Depends, HTTPException, status

# Import SQLAlchemy functions used to construct database queries.
from sqlalchemy import (
    select,  # Builds SQL SELECT queries.
    func,    # Provides SQL aggregate functions such as COUNT, SUM and MAX.
    case     # Provides conditional logic similar to SQL CASE WHEN.
)


# Import Pydantic schemas for validating API inputs
# and defining structured API responses.
from backend.app.schemas.prediction import (
    PredictionInput,                    # Validates patient data submitted for prediction.
    PredictionResponse,                 # Defines the immediate ML prediction response.
    PredictionExplanationResponse,      # Defines individual SHAP explanation responses.
    PredictionHistoryResponse,          # Defines individual prediction history records.
    PredictionDashboardResponse,         # Defines summary statistics for the dashboard.
    GlobalSHAPResponse,                 # Defines global SHAP feature importance responses.
)


# Import the ML prediction function.
from backend.app.services.prediction_service import predict_diabetes

# Import SHAP explanation functions.
# explain_prediction() explains one patient's prediction.
# explain_global_features() calculates overall feature importance.
from backend.app.services.shap_service import (
    explain_prediction,
    explain_global_features
)

# Import the function that identifies the logged-in user.
from backend.app.core.dependencies import get_current_user

# Import the database session dependency.
from backend.app.db.database import get_db

# Import the database models used by these endpoints.
from backend.app.models import (
    User,                  # Represents registered application users.
    Prediction,            # Stores individual diabetes predictions.
    ModelMetadata,         # Stores information about trained ML models.
    PredictionExplanation # Stores SHAP explanations linked to predictions.
)

# ============================================================
# CREATE PREDICTION API ROUTER
# ============================================================

# Create a router to group all prediction-related endpoints.
router = APIRouter(
    prefix="/predictions",
    tags=["Predictions"]
)


# ============================================================
# 1. GET PREDICTION HISTORY
# ============================================================

# Register a GET endpoint at /predictions/history.
@router.get(
    "/history",
    response_model=list[PredictionHistoryResponse]
)
def get_prediction_history(
    current_user: User = Depends(get_current_user),
    db=Depends(get_db)
):

    # Retrieve only predictions belonging to the logged-in user.
    prediction_records = db.scalars(
        select(Prediction)
        .where(Prediction.user_id == current_user.id)
        .order_by(Prediction.created_at.desc())
    ).all()

    # Return the retrieved prediction records.
    return prediction_records


# ============================================================
# 2. GET INDIVIDUAL PREDICTION SHAP EXPLANATION
# ============================================================

# Register a GET endpoint to retrieve one prediction's SHAP explanation.
@router.get(
    "/{prediction_id}/explanation",
    response_model=PredictionExplanationResponse
)
def get_prediction_explanation(

    # Receive the unique ID of the requested prediction.
    prediction_id: int,

    # Identify the currently authenticated user.
    current_user: User = Depends(get_current_user),

    # Obtain a database session for retrieving the explanation.
    db=Depends(get_db)
):

    # Find the requested prediction only if it belongs to this user.
    prediction_record = db.scalars(
        select(Prediction).where(
            Prediction.id == prediction_id,
            Prediction.user_id == current_user.id
        )
    ).first()

    # Return HTTP 404 if the prediction does not exist
    # or belongs to another user.
    if prediction_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found."
        )

    # Retrieve the SHAP explanation linked to this prediction.
    explanation_record = db.scalars(
        select(PredictionExplanation).where(
            PredictionExplanation.prediction_id == prediction_record.id
        )
    ).first()

    # Return HTTP 404 if no SHAP explanation exists.
    if explanation_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="SHAP explanation not found."
        )

    # Return the stored SHAP explanation.
    return {
        "base_value": explanation_record.base_value,
        "model_output": explanation_record.model_output,
        "feature_contributions": explanation_record.feature_contributions
    }



# ============================================================
# 3. GET PREDICTION DASHBOARD STATISTICS
# ============================================================

# Register a GET endpoint at /predictions/dashboard.
# This endpoint returns summary statistics for the logged-in user.
@router.get(
    "/dashboard",
    response_model=PredictionDashboardResponse
)
def get_prediction_dashboard(
    # Identify the currently authenticated user using their JWT token.
    current_user: User = Depends(get_current_user),

    # Obtain a database session for executing SQL queries.
    db=Depends(get_db)
):

    # Execute an aggregate database query to calculate dashboard statistics.
    dashboard_data = db.execute(
        select(

            # Count the total number of predictions.
            func.count(Prediction.id).label("total_predictions"),

            # Count predictions classified as class 0.
            func.sum(
                case(
                    (Prediction.predicted_class == 0, 1),
                    else_=0
                )
            ).label("class_0_predictions"),

            # Count predictions classified as class 1.
            func.sum(
                case(
                    (Prediction.predicted_class == 1, 1),
                    else_=0
                )
            ).label("class_1_predictions"),

            # Find the date and time of the latest prediction.
            func.max(Prediction.created_at).label("latest_prediction")
        )

        # Restrict the query to the currently authenticated user's records.
        .where(Prediction.user_id == current_user.id)

    ).mappings().one()

    # Retrieve the total prediction count.
    total_predictions = dashboard_data["total_predictions"]

    # Retrieve the class 0 count.
    # Use zero if the database returns NULL for an empty history.
    class_0_predictions = dashboard_data["class_0_predictions"] or 0

    # Retrieve the class 1 count.
    # Use zero if the database returns NULL for an empty history.
    class_1_predictions = dashboard_data["class_1_predictions"] or 0

    # Calculate class 0 percentage.
    # Avoid division by zero when the user has no predictions.
    class_0_percentage = (
        (class_0_predictions / total_predictions) * 100
        if total_predictions > 0
        else 0.0
    )

    # Calculate class 1 percentage.
    # Return zero when the user has no predictions.
    class_1_percentage = (
        (class_1_predictions / total_predictions) * 100
        if total_predictions > 0
        else 0.0
    )

    # Return the calculated dashboard statistics.
    return {
        "total_predictions": total_predictions,
        "class_0_predictions": class_0_predictions,
        "class_1_predictions": class_1_predictions,
        "class_0_percentage": round(class_0_percentage, 2),
        "class_1_percentage": round(class_1_percentage, 2),
        "latest_prediction": dashboard_data["latest_prediction"]
    }


# ============================================================
# 4.GET GLOBAL SHAP FEATURE IMPORTANCE
# ============================================================

# Register an authenticated GET endpoint for global SHAP feature importance.
@router.get(
    "/explanations/global",
    response_model=GlobalSHAPResponse
)
def get_global_shap_explanation(
    # Require a logged-in user before providing model explanations.
    current_user: User = Depends(get_current_user)
):
    # Calculate the average absolute SHAP importance for all input features.
    global_explanation = explain_global_features()

    # Return the global feature importance and analysed record count.
    return global_explanation


# ============================================================
# 5. POST GENERATE DIABETES PREDICTION WITH SHAP EXPLANATION
# ============================================================

# Register a POST endpoint at /predictions/predict.
# This endpoint generates a prediction, explains it using SHAP,
# and saves both records in a single database transaction.
@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK
)
def create_prediction(
    # Receive validated patient information from the API request.
    patient_data: PredictionInput,

    # Identify the currently authenticated user.
    current_user: User = Depends(get_current_user),

    # Obtain a database session for storing prediction records.
    db=Depends(get_db)
):

    try:
        # ----------------------------------------------------
        # STEP 1: VERIFY THE ACTIVE ML MODEL
        # ----------------------------------------------------

        # Find all ML models currently marked as active.
        active_models = db.scalars(
            select(ModelMetadata).where(
                ModelMetadata.is_active.is_(True)
            )
        ).all()

        # Ensure exactly one active model is configured.
        # Prediction generation must stop if zero or multiple
        # active models are found.
        if len(active_models) != 1:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Exactly one active ML model must be configured."
            )

        # Select the single active model.
        active_model = active_models[0]

        # ----------------------------------------------------
        # STEP 2: GENERATE THE ML PREDICTION
        # ----------------------------------------------------

        # Pass validated patient information to the prediction service.
        # The service uses the complete saved ML pipeline.
        result = predict_diabetes(patient_data)

        # ----------------------------------------------------
        # STEP 3: GENERATE THE SHAP EXPLANATION
        # ----------------------------------------------------

        # Generate individual feature contributions for this patient.
        # The SHAP service applies the shared preprocessing helper.
        shap_result = explain_prediction(patient_data)

        # ----------------------------------------------------
        # STEP 4: CREATE THE PREDICTION DATABASE RECORD
        # ----------------------------------------------------

        # Prepare a database record containing patient features
        # and the prediction generated by the ML model.
        prediction_record = Prediction(
            user_id=current_user.id,
            model_id=active_model.id,
            pregnancies=patient_data.pregnancies,
            glucose=patient_data.glucose,
            blood_pressure=patient_data.blood_pressure,
            skin_thickness=patient_data.skin_thickness,
            insulin=patient_data.insulin,
            bmi=patient_data.bmi,
            diabetes_pedigree_function=patient_data.diabetes_pedigree_function,
            age=patient_data.age,
            predicted_class=result["predicted_class"],
            predicted_probability=result["predicted_class_probability"]
        )

        # Add the prediction record to the current database session.
        # It is not permanently saved until db.commit() executes.
        db.add(prediction_record)

        # Send the pending INSERT operation to MySQL.
        # This generates prediction_record.id without committing.
        # We need this ID to connect the SHAP explanation to its prediction.
        db.flush()

        # ----------------------------------------------------
        # STEP 5: CREATE THE SHAP EXPLANATION DATABASE RECORD
        # ----------------------------------------------------

        # Prepare the explanation record using the generated prediction ID.
        explanation_record = PredictionExplanation(
            prediction_id=prediction_record.id,
            explanation_method="SHAP",
            base_value=shap_result["base_value"],
            model_output=shap_result["model_output"],
            feature_contributions=shap_result["feature_contributions"]
        )

        # Add the explanation record to the same database session.
        db.add(explanation_record)

        # ----------------------------------------------------
        # STEP 6: SAVE BOTH RECORDS IN ONE TRANSACTION
        # ----------------------------------------------------

        # Commit the prediction and its SHAP explanation together.
        # If either operation fails, the transaction can be rolled back.
        db.commit()

        # ----------------------------------------------------
        # STEP 7: PREPARE THE API RESPONSE
        # ----------------------------------------------------

        # Combine the original prediction results with the SHAP explanation.
        # The response structure is validated by PredictionResponse.
        return {
            **result,
            "explanation": shap_result
        }

    except HTTPException:
        # Undo pending database operations if an HTTP error occurs.
        db.rollback()

        # Preserve and return the original HTTP error.
        raise

    except Exception:
        # Undo both prediction and explanation operations if
        # any unexpected error occurs.
        db.rollback()

        # Return a generic server error without exposing internal details.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating or saving the prediction."
        )