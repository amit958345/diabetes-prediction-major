
"""
Administration API routes.

This module contains endpoints accessible only to administrators.

Available functionality:
1. System-wide dashboard statistics.
2. Paginated user listing.
3. Individual user details.
4. Paginated prediction monitoring.
5. Individual prediction monitoring.

Sensitive information such as password hashes and patient
medical input features is excluded from administrative responses.
"""

# APIRouter creates the administration API router.
# Depends injects dependencies into endpoint functions.
# HTTPException handles API errors.
# Query validates query parameters.
from fastapi import APIRouter, Depends, HTTPException, Query

# Import SQL aggregate functions such as COUNT().
from sqlalchemy import func

# Import IntegrityError to handle duplicate model registrations.
from sqlalchemy.exc import IntegrityError

# Session represents a SQLAlchemy database session.
from sqlalchemy.orm import Session

# Import the database dependency.
from backend.app.db.database import get_db

# Import admin permission dependency.
from backend.app.core.permissions import require_admin

# Import database models.
from backend.app.models.user import User
from backend.app.models.prediction import Prediction
from backend.app.models.model_metadata import ModelMetadata

# Import the safe user response schema.
from backend.app.schemas.user import UserResponse

# Import schemas used for model management.
from backend.app.schemas.model import (
    ModelCreate,
    ModelResponse,
    ModelDetailResponse
)

# Import the safe admin prediction response schema.
from backend.app.schemas.prediction import AdminPredictionResponse


# Create the administration API router.
router = APIRouter(
    prefix="/admin",
    tags=["Administration"]
)


# ---------------------------------------------------------
# ADMIN DASHBOARD
# ---------------------------------------------------------

# Admin-only dashboard endpoint.
@router.get("/dashboard")
def admin_dashboard(
    # Retrieve the currently authenticated administrator.
    current_admin=Depends(require_admin),

    # Inject a database session.
    db: Session = Depends(get_db)
):
    """
    Return administrator information and system-wide statistics.

    Only aggregate prediction information is returned.
    Individual patient inputs are not exposed.
    """

    # Count all registered users.
    total_users = (
        db.query(func.count(User.id)).scalar() or 0
    )

    # Count all prediction records.
    total_predictions = (
        db.query(func.count(Prediction.id)).scalar() or 0
    )

    # Count class 0 predictions.
    class_0_predictions = (
        db.query(func.count(Prediction.id))
        .filter(Prediction.predicted_class == 0)
        .scalar() or 0
    )

    # Count class 1 predictions.
    class_1_predictions = (
        db.query(func.count(Prediction.id))
        .filter(Prediction.predicted_class == 1)
        .scalar() or 0
    )

    # Count active model metadata records.
    active_models = (
        db.query(func.count(ModelMetadata.id))
        .filter(ModelMetadata.is_active.is_(True))
        .scalar() or 0
    )

    # Return administrator details and aggregate statistics.
    return {
        "message": "Welcome to the Admin Dashboard!",
        "admin_name": current_admin.name,
        "admin_email": current_admin.email,

        "statistics": {
            "total_users": total_users,
            "total_predictions": total_predictions,
            "class_0_predictions": class_0_predictions,
            "class_1_predictions": class_1_predictions,
            "active_models": active_models
        }
    }


# ---------------------------------------------------------
# LIST ALL USERS
# ---------------------------------------------------------

# Return a paginated list of registered users.
@router.get("/users", response_model=list[UserResponse])
def get_all_users(
    # Skip this many users before returning results.
    skip: int = Query(default=0, ge=0),

    # Return at most 100 users in one request.
    limit: int = Query(default=20, ge=1, le=100),

    # Require administrator authentication.
    current_admin=Depends(require_admin),

    # Inject the database session.
    db: Session = Depends(get_db)
):
    """
    Retrieve registered users with pagination.

    The UserResponse schema ensures password hashes
    and other unlisted database fields are not exposed.
    """

    # Retrieve users ordered by their database ID.
    users = (
        db.query(User)
        .order_by(User.id.asc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    # FastAPI converts each SQLAlchemy User object
    # into the safe UserResponse schema.
    return users


# ---------------------------------------------------------
# GET USER BY ID
# ---------------------------------------------------------

# Retrieve one registered user using their database ID.
@router.get("/users/{user_id}", response_model=UserResponse)
def get_user_by_id(
    # User ID supplied through the URL.
    user_id: int,

    # Require administrator authentication.
    current_admin=Depends(require_admin),

    # Inject the database session.
    db: Session = Depends(get_db)
):
    """
    Retrieve safe information about one registered user.

    Return HTTP 404 if the requested user does not exist.
    """

    # Search for the user using their primary key.
    user = (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )

    # Return HTTP 404 when no matching user exists.
    if user is None:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    # Return the SQLAlchemy user object.
    # FastAPI validates it against UserResponse.
    return user


# ---------------------------------------------------------
# ADMIN PREDICTION MONITORING
# ---------------------------------------------------------

# Return a paginated list of system-wide predictions.
@router.get(
    "/predictions",
    response_model=list[AdminPredictionResponse]
)
def get_all_predictions(

    # Skip this many prediction records.
    skip: int = Query(default=0, ge=0),

    # Return at most 100 records in one request.
    limit: int = Query(default=20, ge=1, le=100),

    # Optionally filter by prediction class.
    # Only 0 and 1 are valid prediction classes.
    predicted_class: int | None = Query(
        default=None,
        ge=0,
        le=1
    ),

    # Optionally filter by the user who generated predictions.
    # User IDs must be positive integers.
    user_id: int | None = Query(
        default=None,
        ge=1
    ),

    # Only administrators can access system-wide predictions.
    current_admin=Depends(require_admin),

    # Inject the database session.
    db: Session = Depends(get_db)
):
    """
    Retrieve system-wide prediction records.

    Supports:
    - Pagination using skip and limit.
    - Filtering by predicted class.
    - Filtering by user ID.

    Patient medical input features are excluded by
    AdminPredictionResponse.
    """

    # Start a query for all prediction records.
    query = db.query(Prediction)

    # Apply the prediction-class filter when provided.
    if predicted_class is not None:
        query = query.filter(
            Prediction.predicted_class == predicted_class
        )

    # Apply the user-ID filter when provided.
    if user_id is not None:
        query = query.filter(
            Prediction.user_id == user_id
        )

    # Retrieve the filtered records.
    # Newest predictions appear first.
    # ID provides a deterministic secondary sort order.
    predictions = (
        query
        .order_by(
            Prediction.created_at.desc(),
            Prediction.id.desc()
        )
        .offset(skip)
        .limit(limit)
        .all()
    )

    # Return SQLAlchemy objects.
    # FastAPI validates each object against the safe schema.
    return predictions


# ---------------------------------------------------------
# GET PREDICTION BY ID
# ---------------------------------------------------------

# Retrieve one prediction record using its database ID.
@router.get(
    "/predictions/{prediction_id}",
    response_model=AdminPredictionResponse
)
def get_prediction_by_id(

    # Prediction ID supplied through the URL.
    prediction_id: int,

    # Require administrator authentication.
    current_admin=Depends(require_admin),

    # Inject the database session.
    db: Session = Depends(get_db)
):
    """
    Retrieve one prediction record for administrative monitoring.

    Patient medical input features are not returned.

    Return HTTP 404 if the prediction does not exist.
    """

    # Search for the prediction using its primary key.
    prediction = (
        db.query(Prediction)
        .filter(Prediction.id == prediction_id)
        .first()
    )

    # Return HTTP 404 when no matching prediction exists.
    if prediction is None:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found"
        )

    # Return the prediction object.
    # FastAPI filters its fields using AdminPredictionResponse.
    return prediction


# ---------------------------------------------------------
# MODEL MANAGEMENT - LIST ALL REGISTERED MODELS
# ---------------------------------------------------------

# Return a list of registered machine learning models.
# Only administrators are allowed to access this endpoint.
@router.get(
    "/models",
    response_model=list[ModelResponse]
)
def get_all_models(

    # Skip this many model records before returning results.
    skip: int = Query(default=0, ge=0),

    # Return at most 100 model records in one request.
    limit: int = Query(default=20, ge=1, le=100),

    # Verify that the current user has administrator permission.
    current_admin=Depends(require_admin),

    # Inject a database session for querying model metadata.
    db: Session = Depends(get_db)
):
    """
    Retrieve registered machine learning model metadata.

    Supports pagination through skip and limit.

    This endpoint returns metadata only.
    It does not load or switch the model used for predictions.
    """

    # Start a query for all registered model metadata records.
    query = db.query(ModelMetadata)

    # Retrieve model records in newest-first order.
    # The model ID provides a secondary deterministic sort order.
    models = (
        query
        .order_by(
            ModelMetadata.created_at.desc(),
            ModelMetadata.id.desc()
        )
        .offset(skip)
        .limit(limit)
        .all()
    )

    # Return SQLAlchemy objects.
    # FastAPI validates each object against ModelResponse.
    return models


# ---------------------------------------------------------
# MODEL MANAGEMENT - GET MODEL BY ID
# ---------------------------------------------------------

# Retrieve detailed information about one registered ML model.
# Only administrators can access this endpoint.
@router.get(
    "/models/{model_id}",
    response_model=ModelDetailResponse
)
def get_model_by_id(

    # Model ID supplied through the URL.
    model_id: int,

    # Verify administrator permissions.
    current_admin=Depends(require_admin),

    # Inject the database session.
    db: Session = Depends(get_db)
):
    """
    Retrieve detailed metadata for one registered ML model.

    The response includes the model name, version, algorithm,
    active status, creation timestamp, and evaluation metrics.

    This endpoint retrieves database metadata only.
    It does not load the model artifact into memory.
    """

    # Search for the model using its primary key.
    model_record = (
        db.query(ModelMetadata)
        .filter(ModelMetadata.id == model_id)
        .first()
    )

    # Return HTTP 404 if the requested model does not exist.
    if model_record is None:
        raise HTTPException(
            status_code=404,
            detail="Model not found"
        )

    # Return the SQLAlchemy model metadata object.
    # FastAPI validates it against ModelDetailResponse.
    return model_record



# ---------------------------------------------------------
# MODEL MANAGEMENT - REGISTER NEW MODEL METADATA
# ---------------------------------------------------------

# Register metadata for a newly trained machine learning model.
# Only administrators can register model metadata.
@router.post(
    "/models",
    response_model=ModelDetailResponse,
    status_code=201
)
def register_model(

    # Validate incoming registration data using ModelCreate.
    model_data: ModelCreate,

    # Verify that the current user has administrator permissions.
    current_admin=Depends(require_admin),

    # Inject a database session for saving model metadata.
    db: Session = Depends(get_db)
):
    """
    Register metadata for a new machine learning model.

    The endpoint stores:
    - Model name
    - Version
    - Algorithm
    - Evaluation metrics

    New model records are inactive by default.

    This endpoint does not upload model files, replace the
    currently loaded model, or activate a different model.
    """

    # Create a new SQLAlchemy model metadata object.
    # is_active is intentionally omitted so the database default
    # keeps newly registered models inactive.
    new_model = ModelMetadata(
        model_name=model_data.model_name,
        version=model_data.version,
        algorithm=model_data.algorithm,
        metrics=model_data.metrics
    )

    try:

        # Add the new model metadata object to the database session.
        db.add(new_model)

        # Commit the transaction to permanently save the record.
        db.commit()

        # Refresh the object to retrieve database-generated values,
        # such as its ID and creation timestamp.
        db.refresh(new_model)

    except IntegrityError:

        # Roll back the failed transaction to keep the session usable.
        db.rollback()

        # Return HTTP 409 when the model name/version combination
        # already exists or another integrity constraint is violated.
        raise HTTPException(
            status_code=409,
            detail="A model with this name and version already exists"
        )

    except Exception:

        # Roll back any other failed database transaction.
        db.rollback()

        # Return a generic server error without exposing internal details.
        raise HTTPException(
            status_code=500,
            detail="Unable to register model metadata"
        )

    # Return the newly registered model metadata.
    # FastAPI validates it using ModelDetailResponse.
    return new_model


# ---------------------------------------------------------
# MODEL MANAGEMENT - ACTIVATE A REGISTERED MODEL
# ---------------------------------------------------------

@router.post(
    "/models/{model_id}/activate",
    response_model=ModelDetailResponse
)
def activate_model(
    model_id: int,
    current_admin=Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Activate one registered model metadata record.

    All other registered models are deactivated.
    This endpoint changes database metadata only.
    It does not load or replace model files.
    """

    # Find the requested model.
    model_record = (
        db.query(ModelMetadata)
        .filter(ModelMetadata.id == model_id)
        .first()
    )

    # Do not change anything if the model does not exist.
    if model_record is None:
        raise HTTPException(
            status_code=404,
            detail="Model not found"
        )

    try:
        # Deactivate all currently active models.
        db.query(ModelMetadata).filter(
            ModelMetadata.is_active.is_(True)
        ).update(
            {ModelMetadata.is_active: False},
            synchronize_session=False
        )

        # Activate the selected model.
        model_record.is_active = True

        # Save both changes in one transaction.
        db.commit()
        db.refresh(model_record)

    except Exception:
        # Undo changes if anything goes wrong.
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail="Unable to activate model"
        )

    return model_record
