from fastapi import APIRouter, Depends, HTTPException, status

from backend.app.core.dependencies import get_current_user
from backend.app.models.user import User
from backend.app.services.insights_service import get_model_insights

router = APIRouter(
    prefix="/insights",
    tags=["Model Insights"]
)

@router.get("/overview")
def read_model_insights(
    current_user: User = Depends(get_current_user)
):
    """
    Return dataset intelligence and offline research metrics.

    Authentication is required. This endpoint is read-only and does
    not expose administrator functions or individual patient records.
    """
    try:
        return get_model_insights()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error)
        )
    except (ValueError, KeyError) as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to prepare model insights: {error}"
        )
