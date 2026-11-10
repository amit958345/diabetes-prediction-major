
# ==========================================================
# DIAPREDICT - ONE-TIME ADMIN ACCOUNT CREATION
# ==========================================================

from getpass import getpass

from sqlalchemy.exc import IntegrityError

from backend.app.db.database import SessionLocal
from backend.app.models import User
from backend.app.core.security import hash_password
from backend.app.services.auth_service import get_user_by_email


def create_admin():

    # Create a database session.
    db = SessionLocal()

    try:

        print("\n================================")
        print(" DIAPREDICT ADMIN ACCOUNT SETUP")
        print("================================\n")

        # Collect administrator details.
        name = input("Enter admin name: ").strip()
        email = input("Enter admin email: ").strip().lower()

        # Check that required fields are not empty.
        if not name or not email:
            print("Error: Name and email are required.")
            return

        # Check whether the email already exists.
        existing_user = get_user_by_email(db, email)

        if existing_user is not None:
            print("Error: This email is already registered.")
            return

        # Request the password without displaying it.
        password = getpass("Enter admin password: ")
        confirm_password = getpass("Confirm admin password: ")

        # Validate password confirmation.
        if password != confirm_password:
            print("Error: Passwords do not match.")
            return

        # Require a sufficiently strong password.
        if len(password) < 12:
            print("Error: Password must contain at least 12 characters.")
            return

        # Hash the password using the existing security function.
        hashed_password = hash_password(password)

        # Create a separate administrator account.
        new_admin = User(
            name=name,
            email=email,
            password_hash=hashed_password,
            role="admin"
        )

        # Save the administrator account in TiDB Cloud.
        db.add(new_admin)
        db.commit()
        db.refresh(new_admin)

        print("\nAdmin account created successfully!")
        print("Admin ID:", new_admin.id)
        print("Admin Name:", new_admin.name)
        print("Admin Email:", new_admin.email)
        print("Admin Role:", new_admin.role)

    except IntegrityError:
        db.rollback()
        print("Error: This email is already registered.")

    except Exception as error:
        db.rollback()
        print("Admin creation failed:", str(error))

    finally:
        # Always close the database session.
        db.close()


if __name__ == "__main__":
    create_admin()
