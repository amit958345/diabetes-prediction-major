
from backend.app.db.database import SessionLocal
from backend.app.services.auth_service import get_user_by_email


# Create a database session
db = SessionLocal()

try:
    # Search for an email that should not exist
    existing_user = get_user_by_email(
        db,
        "notregistered@example.com"
    )

    # Check the search result
    if existing_user is None:
        print("TEST PASSED: No matching user found.")
    else:
        print("User found:")
        print("Name:", existing_user.name)
        print("Email:", existing_user.email)

finally:
    # Close the database session
    db.close()
