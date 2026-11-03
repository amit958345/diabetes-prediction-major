
from backend.app.db.database import SessionLocal
from backend.app.services.auth_service import create_user
from backend.app.core.security import verify_password
from backend.app.models import User


# Sample registration information for testing
test_name = "Registration Test"
test_email = "registration.test@example.com"
test_password = "StrongPassword123"

db = SessionLocal()
new_user = None

try:
    # TEST 1: Register a new user
    new_user = create_user(
        db,
        test_name,
        test_email,
        test_password
    )

    print("TEST 1: User registered successfully")
    print("Generated ID:", new_user.id)
    print("Name:", new_user.name)
    print("Email:", new_user.email)
    print("Role:", new_user.role)

    # TEST 2: Check whether the password is hashed
    if new_user.password_hash.startswith("$argon2id$"):
        print("TEST 2: Argon2id hash confirmed")
    else:
        print("TEST 2: Argon2id hash NOT confirmed")

    # TEST 3: Verify the original password
    if verify_password(test_password, new_user.password_hash):
        print("TEST 3: Password verification successful")
    else:
        print("TEST 3: Password verification failed")

    # TEST 4: Try registering the same email again
    try:
        create_user(
            db,
            test_name,
            test_email,
            test_password
        )

        print("TEST 4 FAILED: Duplicate email was accepted")

    except ValueError as error:
        print("TEST 4: Duplicate email rejected")
        print("Reason:", error)

finally:
    # Delete the temporary test user
    if new_user is not None:
        db.delete(new_user)
        db.commit()
        print("Temporary test user deleted.")

    # Close the database session
    db.close()
