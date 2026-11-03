
from fastapi import Depends, HTTPException, status

from backend.app.models import User
from backend.app.core.dependencies import get_current_user


# Check whether the authenticated user is an administrator
def require_admin(
    current_user: User = Depends(get_current_user)
) -> User:

    # Reject users who do not have the admin role
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required."
        )

    # Return the administrator if permission is granted
    return current_user
