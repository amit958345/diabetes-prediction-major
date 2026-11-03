
from fastapi import APIRouter, Depends

from backend.app.core.permissions import require_admin


# Create an administration API router
router = APIRouter(
    prefix="/admin",
    tags=["Administration"]
)


# Admin-only dashboard endpoint
@router.get("/dashboard")
def admin_dashboard(
    current_admin = Depends(require_admin)
):
    return {
        "message": "Welcome to the Admin Dashboard!",
        "admin_name": current_admin.name,
        "admin_email": current_admin.email
    }
