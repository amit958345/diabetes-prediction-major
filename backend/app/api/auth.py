
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from backend.app.models import User
from backend.app.core.dependencies import get_current_user

from backend.app.db.database import get_db
from backend.app.schemas.user import (
    UserCreate,
    UserResponse,
    UserLogin,
    TokenResponse
)
from backend.app.services.auth_service import (
    create_user,
    get_user_by_email
)
from backend.app.core.security import (
    verify_password,
    create_access_token
)


# Create an authentication API router
router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


# User registration endpoint
@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED
)
def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    try:
        new_user = create_user(
            db=db,
            name=user_data.name,
            email=str(user_data.email),
            password=user_data.password
        )

        return new_user

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error)
        )

    except IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This email is already registered."
        )


# User login endpoint
@router.post(
    "/login",
    response_model=TokenResponse
)
def login_user(
    user_data: UserLogin,
    db: Session = Depends(get_db)
):
    # Find the user using their email address
    user = get_user_by_email(
        db,
        str(user_data.email)
    )

    # Verify the submitted password
    if user is None or not verify_password(
        user_data.password,
        user.password_hash
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Generate a JWT access token
    access_token = create_access_token(
        data={"sub": str(user.id)}
    )

    # Return the token and safe user information
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }

# Get the currently authenticated user's information
@router.get(
    "/me",
    response_model=UserResponse
)
def get_my_profile(
    current_user: User = Depends(get_current_user)
):
    return current_user