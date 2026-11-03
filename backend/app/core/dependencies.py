
from fastapi import Depends,HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import jwt

from backend.app.db.database import get_db
from backend.app.models import User
from backend.app.core.security import decode_access_token


# Create the Bearer authentication scheme
bearer_scheme = HTTPBearer(auto_error=False)


# Identify the currently authenticated user
def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
    db: Session = Depends(get_db)
) -> User:

    # Check whether authentication credentials were provided
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials are missing.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Extract the JWT from the credentials
    token = credentials.credentials

    # Decode and validate the JWT
    try:
        payload = decode_access_token(token)

    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Extract the user ID from the JWT subject
    user_id = payload.get("sub")

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Validate and convert the user ID
    try:
        user_id = int(user_id)

        if user_id <= 0:
            raise ValueError

    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Find the user in MySQL using the primary key
    user = db.get(User, user_id)

    # Check whether the user account still exists
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account not found.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Return the authenticated user
    return user
