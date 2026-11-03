
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import User
from backend.app.core.security import hash_password

# Find an existing user using their email address
def get_user_by_email(db: Session, email: str):
    statement = select(User).where(User.email == email)

    return db.execute(statement).scalar_one_or_none()


# Create a new user in the database
def create_user(
    db: Session,
    name: str,
    email: str,
    password: str
):
    # Check whether the email is already registered
    existing_user = get_user_by_email(db, email)

    if existing_user is not None:
        raise ValueError("Email is already registered.")

    # Hash the user's password
    hashed_password = hash_password(password)

    # Create a new User object
    new_user = User(
        name=name,
        email=email,
        password_hash=hashed_password,
        role="user"
    )

    try:
        # Add the new user to the database session
        db.add(new_user)

        # Save the new user in MySQL
        db.commit()

        # Retrieve database-generated values
        db.refresh(new_user)

        return new_user

    except Exception:
        # Cancel the transaction if an error occurs
        db.rollback()
        raise
