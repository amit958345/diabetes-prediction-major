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
    PredictionInput,              # Validates patient data submitted for prediction.
    PredictionResponse,           # Defines the immediate ML prediction response.
    PredictionHistoryResponse,    # Defines individual prediction history records.
    PredictionDashboardResponse   # Defines summary statistics for the dashboard.
)

# Import the ML prediction function.
from backend.app.services.prediction_service import predict_diabetes

# Import the function that identifies the logged-in user.
from backend.app.core.dependencies import get_current_user

# Import the database session dependency.
from backend.app.db.database import get_db

# Import the database models used by these endpoints.
from backend.app.models import User, Prediction, ModelMetadata


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
# 2. GET PREDICTION DASHBOARD STATISTICS
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
# 3. POST GENERATE DIABETES PREDICTION
# ============================================================

# Register a POST endpoint at /predictions/predict.
@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK
)
def create_prediction(
    patient_data: PredictionInput,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db)
):

    try:
        # Find all active ML models in the database.
        active_models = db.scalars(
            select(ModelMetadata).where(
                ModelMetadata.is_active.is_(True)
            )
        ).all()

        # Ensure exactly one active ML model exists.
        if len(active_models) != 1:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Exactly one active ML model must be configured."
            )

        # Select the active model.
        active_model = active_models[0]

        # Generate the ML prediction.
        result = predict_diabetes(patient_data)

        # Create a prediction database record.
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

        # Add the prediction record to the database session.
        db.add(prediction_record)

        # Save the prediction record in MySQL.
        db.commit()

        # Return the generated prediction result.
        return result

    except HTTPException:
        # Roll back the database transaction if an HTTP error occurs.
        db.rollback()
        raise

    except Exception:
        # Roll back the database transaction if another error occurs.
        db.rollback()

        # Return an internal server error.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while generating or saving the prediction."
        )