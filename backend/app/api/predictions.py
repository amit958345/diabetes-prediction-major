# Import FastAPI components for creating API routes,
# dependency injection, HTTP errors and status codes.
import csv
from io import StringIO

from fastapi import APIRouter, Depends, HTTPException, Response, status

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
from backend.app.services.prediction_report_service import (
    generate_prediction_report_pdf,
    generate_research_report_pdf,
)

# Import SHAP explanation functions.
# explain_prediction() explains one patient's prediction.
# explain_global_features() calculates overall feature importance.
from backend.app.services.shap_service import (
    explain_prediction,          # Generates an explanation for one patient.
    explain_global_features,     # Calculates global feature importance.
    explain_beeswarm_data        # Generates per-record SHAP data for the beeswarm plot.
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
# 2. EXPORT AUTHENTICATED PREDICTION HISTORY AS CSV
# ============================================================

@router.get(
    "/history/export",
    tags=["Prediction Reports"]
)
def export_prediction_history_csv(
    search: str | None = None,
    predicted_class: int | None = None,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db)
):
    """
    Export the current user's prediction history as a CSV file.

    Optional filters:
    - search: matches prediction ID, age, glucose, BMI, or creation date.
    - predicted_class: 0 or 1; omit it to export all classes.
    """

    if predicted_class not in (None, 0, 1):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="predicted_class must be 0, 1, or omitted."
        )

    # Always restrict the query to the authenticated user's records.
    records = db.scalars(
        select(Prediction)
        .where(Prediction.user_id == current_user.id)
        .order_by(Prediction.created_at.desc())
    ).all()

    # Apply the same practical search fields used by Prediction History.
    search_text = (search or "").strip().casefold()
    if search_text:
        def matches_search(record):
            created_text = (
                record.created_at.isoformat()
                if record.created_at is not None
                else ""
            )
            searchable_values = (
                str(record.id),
                str(record.age),
                str(record.glucose),
                str(record.bmi),
                created_text,
            )
            return any(
                search_text in value.casefold()
                for value in searchable_values
            )

        records = [record for record in records if matches_search(record)]

    if predicted_class is not None:
        records = [
            record for record in records
            if record.predicted_class == predicted_class
        ]

    output = StringIO(newline="")
    writer = csv.writer(output)

    writer.writerow([
        "Prediction ID",
        "Created At",
        "Pregnancies",
        "Glucose",
        "Blood Pressure",
        "Skin Thickness",
        "Insulin",
        "BMI",
        "Diabetes Pedigree Function",
        "Age",
        "Predicted Class",
        "Predicted Class Probability",
        "Class 1 Probability",
        "Class 0 Probability",
    ])

    for record in records:
        predicted_class_value = int(record.predicted_class)
        predicted_probability = float(record.predicted_probability)

        # The stored probability is the probability of the predicted class.
        # Convert it into explicit Class 1 and Class 0 probabilities.
        probability_class_1 = (
            predicted_probability
            if predicted_class_value == 1
            else 1.0 - predicted_probability
        )
        probability_class_0 = 1.0 - probability_class_1

        created_text = (
            record.created_at.isoformat(sep=" ")
            if record.created_at is not None
            else ""
        )

        writer.writerow([
            record.id,
            created_text,
            record.pregnancies,
            record.glucose,
            record.blood_pressure,
            record.skin_thickness,
            record.insulin,
            record.bmi,
            record.diabetes_pedigree_function,
            record.age,
            predicted_class_value,
            round(predicted_probability, 6),
            round(probability_class_1, 6),
            round(probability_class_0, 6),
        ])

    csv_content = "\ufeff" + output.getvalue()
    filename = "DiaPredict_Prediction_History.csv"

    return Response(
        content=csv_content,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


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
# 3. DOWNLOAD THE COMPLETE RESEARCH REPORT AS PDF
# ============================================================

@router.get(
    "/research-report",
    tags=["Prediction Reports"]
)
def download_research_report(
    current_user: User = Depends(get_current_user)
):
    """Generate the complete DiaPredict research evaluation PDF."""
    try:
        pdf_content = generate_research_report_pdf()
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to generate the Research Report PDF."
        ) from error

    return Response(
        content=pdf_content,
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'attachment; filename="DiaPredict_Research_Report.pdf"'
        }
    )


# ============================================================
# 3. DOWNLOAD AN INDIVIDUAL PREDICTION PDF REPORT
# ============================================================

@router.get(
    "/{prediction_id}/report",
    tags=["Prediction Reports"]
)
def download_prediction_report(
    prediction_id: int,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db)
):
    """Generate a PDF report for a prediction owned by the current user."""

    # The ownership condition prevents users from downloading another
    # user's prediction report, even if they know its numeric ID.
    prediction_record = db.scalars(
        select(Prediction).where(
            Prediction.id == prediction_id,
            Prediction.user_id == current_user.id
        )
    ).first()

    if prediction_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found."
        )

    # Older prediction records may not have a stored SHAP explanation.
    # The report service handles that case and clearly labels it.
    explanation_record = db.scalars(
        select(PredictionExplanation).where(
            PredictionExplanation.prediction_id == prediction_record.id
        )
    ).first()

    try:
        pdf_content = generate_prediction_report_pdf(
            prediction=prediction_record,
            explanation=explanation_record
        )
    except Exception as error:
        # Keep internal exception details out of the API response.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to generate the prediction PDF report."
        ) from error

    filename = f"DiaPredict_Prediction_{prediction_record.id}.pdf"

    return Response(
        content=pdf_content,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


# ============================================================
# 4. GET PREDICTION DASHBOARD STATISTICS
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
# 5. GET ADVANCED ANALYTICS FOR THE AUTHENTICATED USER
# ============================================================

@router.get(
    "/analytics",
    tags=["Advanced Analytics"]
)
def get_prediction_analytics(
    current_user: User = Depends(get_current_user),
    db=Depends(get_db)
):
    """
    Return advanced analytics calculated from the logged-in
    user's prediction history.

    Analytics include:
    - Daily prediction trends.
    - Class 0 and Class 1 trends.
    - Prediction confidence distribution.
    - Average glucose, BMI and age.
    """

    # Prediction probability already represents confidence
    # in the predicted class.
    confidence_expression = Prediction.predicted_probability

    # --------------------------------------------------------
    # STEP 1: CALCULATE OVERALL ANALYTICS
    # --------------------------------------------------------

    analytics_data = db.execute(
        select(
            func.count(Prediction.id).label("total_predictions"),

            func.sum(
                case(
                    (Prediction.predicted_class == 0, 1),
                    else_=0
                )
            ).label("class_0_predictions"),

            func.sum(
                case(
                    (Prediction.predicted_class == 1, 1),
                    else_=0
                )
            ).label("class_1_predictions"),

            func.avg(
                confidence_expression
            ).label("average_confidence"),

            func.avg(
                Prediction.glucose
            ).label("average_glucose"),

            func.avg(
                Prediction.bmi
            ).label("average_bmi"),

            func.avg(
                Prediction.age
            ).label("average_age"),

            func.min(
                Prediction.glucose
            ).label("minimum_glucose"),

            func.max(
                Prediction.glucose
            ).label("maximum_glucose"),

            func.min(
                Prediction.bmi
            ).label("minimum_bmi"),

            func.max(
                Prediction.bmi
            ).label("maximum_bmi")
        )
        .where(
            Prediction.user_id == current_user.id
        )
    ).mappings().one()

    total_predictions = (
        analytics_data["total_predictions"] or 0
    )

    class_0_predictions = (
        analytics_data["class_0_predictions"] or 0
    )

    class_1_predictions = (
        analytics_data["class_1_predictions"] or 0
    )

    # --------------------------------------------------------
    # STEP 2: CALCULATE DAILY PREDICTION TRENDS
    # --------------------------------------------------------

    daily_data = db.execute(
        select(
            func.date(
                Prediction.created_at
            ).label("prediction_date"),

            func.count(
                Prediction.id
            ).label("total"),

            func.sum(
                case(
                    (Prediction.predicted_class == 0, 1),
                    else_=0
                )
            ).label("class_0"),

            func.sum(
                case(
                    (Prediction.predicted_class == 1, 1),
                    else_=0
                )
            ).label("class_1")
        )
        .where(
            Prediction.user_id == current_user.id
        )
        .group_by(
            func.date(Prediction.created_at)
        )
        .order_by(
            func.date(Prediction.created_at).desc()
        )
        .limit(30)
    ).mappings().all()

    # Reverse the records so the chart displays
    # dates from oldest to newest.
    daily_trends = [
        {
            "date": (
                str(record["prediction_date"])
                if record["prediction_date"] is not None
                else None
            ),
            "total": record["total"] or 0,
            "class_0": record["class_0"] or 0,
            "class_1": record["class_1"] or 0
        }
        for record in reversed(daily_data)
    ]

    # --------------------------------------------------------
    # STEP 3: CALCULATE PREDICTION CONFIDENCE DISTRIBUTION
    # --------------------------------------------------------

    confidence_data = db.execute(
        select(
            func.sum(
                case(
                    (
                        confidence_expression < 0.60,
                        1
                    ),
                    else_=0
                )
            ).label("range_below_60"),

            func.sum(
                case(
                    (
                        (confidence_expression >= 0.60) &
                        (confidence_expression < 0.70),
                        1
                    ),
                    else_=0
                )
            ).label("range_60_69"),

            func.sum(
                case(
                    (
                        (confidence_expression >= 0.70) &
                        (confidence_expression < 0.80),
                        1
                    ),
                    else_=0
                )
            ).label("range_70_79"),

            func.sum(
                case(
                    (
                        (confidence_expression >= 0.80) &
                        (confidence_expression < 0.90),
                        1
                    ),
                    else_=0
                )
            ).label("range_80_89"),

            func.sum(
                case(
                    (
                        confidence_expression >= 0.90,
                        1
                    ),
                    else_=0
                )
            ).label("range_90_100")
        )
        .where(
            Prediction.user_id == current_user.id
        )
    ).mappings().one()

    probability_distribution = [
        {
            "range": "Below 60%",
            "count": confidence_data["range_below_60"] or 0
        },
        {
            "range": "60–69%",
            "count": confidence_data["range_60_69"] or 0
        },
        {
            "range": "70–79%",
            "count": confidence_data["range_70_79"] or 0
        },
        {
            "range": "80–89%",
            "count": confidence_data["range_80_89"] or 0
        },
        {
            "range": "90–100%",
            "count": confidence_data["range_90_100"] or 0
        }
    ]
    # --------------------------------------------------------
    # STEP 4: CALCULATE OUTCOME DISTRIBUTION
    # --------------------------------------------------------

    class_0_percentage = (
        (class_0_predictions / total_predictions) * 100
        if total_predictions > 0
        else 0
    )

    class_1_percentage = (
        (class_1_predictions / total_predictions) * 100
        if total_predictions > 0
        else 0
    )

    outcome_distribution = [
        {
            "label": "Class 0",
            "count": class_0_predictions,
            "percentage": round(class_0_percentage, 2)
        },
        {
            "label": "Class 1",
            "count": class_1_predictions,
            "percentage": round(class_1_percentage, 2)
        }
    ]

    # --------------------------------------------------------
    # STEP 5: CALCULATE INPUT FEATURE AVERAGES
    # --------------------------------------------------------

    input_averages = {
        "glucose": round(
            float(analytics_data["average_glucose"] or 0),
            2
        ),
        "bmi": round(
            float(analytics_data["average_bmi"] or 0),
            2
        ),
        "age": round(
            float(analytics_data["average_age"] or 0),
            2
        )
    }

    input_ranges = {
        "glucose": {
            "minimum": analytics_data["minimum_glucose"],
            "maximum": analytics_data["maximum_glucose"]
        },
        "bmi": {
            "minimum": analytics_data["minimum_bmi"],
            "maximum": analytics_data["maximum_bmi"]
        }
    }

    # --------------------------------------------------------
    # STEP 6: PREPARE ANALYTICS SUMMARY
    # --------------------------------------------------------

    average_confidence = (
        float(analytics_data["average_confidence"] or 0)
        * 100
    )

    analytics_summary = {
        "total_predictions": total_predictions,
        "class_0_predictions": class_0_predictions,
        "class_1_predictions": class_1_predictions,
        "average_confidence": round(
            average_confidence,
            2
        ),
        "average_glucose": input_averages["glucose"],
        "average_bmi": input_averages["bmi"],
        "average_age": input_averages["age"]
    }

    # --------------------------------------------------------
    # STEP 7: RETURN THE COMPLETE ANALYTICS RESPONSE
    # --------------------------------------------------------

    return {
        "total_predictions": total_predictions,
        "outcome_distribution": outcome_distribution,
        "daily_trends": daily_trends,
        "probability_distribution": probability_distribution,
        "input_averages": input_averages,
        "input_ranges": input_ranges,
        "summary": analytics_summary
    }



# ============================================================
# 6.GET GLOBAL SHAP FEATURE IMPORTANCE
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
# 7. GET DATA FOR THE SHAP BEESWARM VISUALIZATION
# ============================================================

# Register an authenticated GET endpoint for the SHAP beeswarm plot.
# The full URL is /predictions/explanations/beeswarm.
@router.get(
    "/explanations/beeswarm",
    tags=["Explainable AI"]
)
def get_beeswarm_shap_explanation(
    # Require a valid logged-in user before returning explanation data.
    # This follows the same authentication pattern as the global SHAP endpoint.
    current_user: User = Depends(get_current_user)
):
    """
    Return per-record SHAP values and original feature values.

    The response is used by the React beeswarm visualization.
    It contains the training dataset explanations, not user-specific
    prediction history or independent test-set explanations.
    """

    # Calculate actual SHAP contributions for every dataset record.
    # The service returns ordinary Python lists and numbers,
    # which FastAPI can serialize into JSON.
    beeswarm_data = explain_beeswarm_data()

    # Return the complete dataset required by the frontend chart.
    return beeswarm_data


# ============================================================
# 8. POST GENERATE DIABETES PREDICTION WITH SHAP EXPLANATION
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
