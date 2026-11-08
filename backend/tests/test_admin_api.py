
"""
Admin API tests.

These tests verify that:
1. An authenticated admin can access the admin dashboard.
2. An authenticated ordinary user cannot access the admin dashboard.
3. An unauthenticated visitor cannot access the admin dashboard.

The tests create temporary users and remove them after execution.
Existing users and model metadata records are not modified.
"""

# Import SQLAlchemy's func module.
# It provides database aggregate functions such as COUNT(),
# which we use to calculate expected dashboard statistics.
from sqlalchemy import func

# Import the Prediction database model.
# It represents prediction records stored in the predictions table.
from backend.app.models.prediction import Prediction

# Import the ModelMetadata database model.
# It represents registered ML model versions and their active status.
from backend.app.models.model_metadata import ModelMetadata

# Generate unique email addresses so test accounts do not conflict
# with real users or accounts created by other test runs.
from uuid import uuid4

import pytest

# TestClient sends HTTP requests to the FastAPI application
# without requiring a separately running server.
from fastapi.testclient import TestClient

# Import the FastAPI application.
from backend.app.main import app

# Import the database session factory.
from backend.app.db.database import SessionLocal

# Import the User database model.
from backend.app.models.user import User

# Import password hashing and JWT creation functions.
from backend.app.core.security import hash_password, create_access_token


@pytest.fixture
def admin_test_clients():
    """
    Create temporary admin and ordinary-user accounts.

    Yield:
        client: FastAPI test client.
        admin_headers: Authentication headers for the admin.
        user_headers: Authentication headers for the ordinary user.

    After the test, delete only the temporary users created here.
    """

    # Open a database session to create the temporary test accounts.
    db = SessionLocal()

    # Generate a unique identifier for this test execution.
    unique_id = uuid4().hex

    # Keep track of the temporary user IDs for safe cleanup.
    created_user_ids = []

    try:
        # Create a temporary administrator account.
        admin_user = User(
            name="Temporary Admin",
            email=f"admin-{unique_id}@example.com",
            password_hash=hash_password("AdminTest@123"),
            role="admin"
        )

        # Create a temporary ordinary-user account.
        normal_user = User(
            name="Temporary User",
            email=f"user-{unique_id}@example.com",
            password_hash=hash_password("UserTest@123"),
            role="user"
        )

        # Add both temporary accounts to the database session.
        db.add_all([admin_user, normal_user])

        # Save both accounts to the database.
        db.commit()

        # Refresh the objects to retrieve their database-generated IDs.
        db.refresh(admin_user)
        db.refresh(normal_user)

        # Save the IDs so cleanup targets only these temporary accounts.
        created_user_ids = [admin_user.id, normal_user.id]

        # Create a JWT containing the admin's database user ID.
        admin_token = create_access_token(
            data={"sub": str(admin_user.id)}
        )

        # Create a JWT containing the ordinary user's database user ID.
        user_token = create_access_token(
            data={"sub": str(normal_user.id)}
        )

        # Build the HTTP Authorization header for the admin.
        admin_headers = {
            "Authorization": f"Bearer {admin_token}"
        }

        # Build the HTTP Authorization header for the ordinary user.
        user_headers = {
            "Authorization": f"Bearer {user_token}"
        }

    except Exception:
        # Undo any incomplete database changes if account creation fails.
        db.rollback()
        raise

    finally:
        # Close the database session used to create the accounts.
        db.close()

    # Create a test client and keep it available during the test.
    try:
        with TestClient(app) as client:

            # Provide the client and both authentication headers to the test.
            yield client, admin_headers, user_headers

    finally:
        # Open a fresh database session for test-account cleanup.
        cleanup_db = SessionLocal()

        try:
            # Retrieve and delete only the temporary users created above.
            for user_id in created_user_ids:
                temporary_user = cleanup_db.get(User, user_id)

                if temporary_user is not None:
                    cleanup_db.delete(temporary_user)

            # Save the cleanup changes.
            cleanup_db.commit()

        except Exception:
            # Roll back cleanup changes if deletion fails.
            cleanup_db.rollback()
            raise

        finally:
            # Always close the cleanup database session.
            cleanup_db.close()


def test_admin_can_access_dashboard(admin_test_clients):
    """
    Verify that an authenticated administrator can access the dashboard.
    """

    # Retrieve the test client and admin authentication headers.
    client, admin_headers, user_headers = admin_test_clients

    # Send an authenticated GET request to the admin dashboard.
    response = client.get(
        "/admin/dashboard",
        headers=admin_headers
    )

    # HTTP 200 means the administrator was allowed to access the endpoint.
    assert response.status_code == 200

    # Convert the JSON response into a Python dictionary.
    response_data = response.json()

    # Verify the dashboard welcome message.
    assert response_data["message"] == "Welcome to the Admin Dashboard!"

    # Verify that the endpoint returned the correct administrator name.
    assert response_data["admin_name"] == "Temporary Admin"

    # Verify that the endpoint returned the administrator's email.
    assert response_data["admin_email"].endswith("@example.com")


def test_normal_user_cannot_access_admin_dashboard(admin_test_clients):
    """
    Verify that an ordinary authenticated user receives HTTP 403.
    """

    # Retrieve the test client and ordinary-user authentication headers.
    client, admin_headers, user_headers = admin_test_clients

    # Send a GET request using the ordinary user's JWT.
    response = client.get(
        "/admin/dashboard",
        headers=user_headers
    )

    # HTTP 403 means the user is authenticated but lacks admin permission.
    assert response.status_code == 403


def test_unauthenticated_user_cannot_access_admin_dashboard(admin_test_clients):
    """
    Verify that a visitor without a JWT receives HTTP 401.
    """

    # Retrieve the test client.
    client, admin_headers, user_headers = admin_test_clients

    # Send a GET request without any Authorization header.
    response = client.get("/admin/dashboard")

    # HTTP 401 means authentication is required.
    assert response.status_code == 401



def test_admin_dashboard_returns_correct_statistics(admin_test_clients):
    """
    Verify that the admin dashboard returns statistics
    matching the actual database records.
    """

    # Retrieve the test client and admin authentication headers.
    client, admin_headers, user_headers = admin_test_clients

    # Open a database session to calculate expected statistics.
    db = SessionLocal()

    try:
        # Count all registered users, including the temporary test accounts.
        expected_users = (
            db.query(func.count(User.id)).scalar() or 0
        )

        # Count all prediction records.
        expected_predictions = (
            db.query(func.count(Prediction.id)).scalar() or 0
        )

        # Count all class 0 predictions.
        expected_class_0 = (
            db.query(func.count(Prediction.id))
            .filter(Prediction.predicted_class == 0)
            .scalar() or 0
        )

        # Count all class 1 predictions.
        expected_class_1 = (
            db.query(func.count(Prediction.id))
            .filter(Prediction.predicted_class == 1)
            .scalar() or 0
        )

        # Count all active model metadata records.
        expected_active_models = (
            db.query(func.count(ModelMetadata.id))
            .filter(ModelMetadata.is_active.is_(True))
            .scalar() or 0
        )

    finally:
        # Close the database session after calculating expected values.
        db.close()

    # Request the admin dashboard using the administrator's JWT.
    response = client.get(
        "/admin/dashboard",
        headers=admin_headers
    )

    # Verify that the dashboard request succeeds.
    assert response.status_code == 200

    # Extract the statistics dictionary from the JSON response.
    statistics = response.json()["statistics"]

    # Compare every API statistic with its actual database count.
    assert statistics["total_users"] == expected_users
    assert statistics["total_predictions"] == expected_predictions
    assert statistics["class_0_predictions"] == expected_class_0
    assert statistics["class_1_predictions"] == expected_class_1
    assert statistics["active_models"] == expected_active_models

    # Verify that class 0 and class 1 counts cover all predictions.
    assert (
        statistics["class_0_predictions"]
        + statistics["class_1_predictions"]
        == statistics["total_predictions"]
    )


# ---------------------------------------------------------
# ADMIN USER MANAGEMENT TESTS
# ---------------------------------------------------------


def test_admin_can_list_users_without_exposing_password_hash(admin_test_clients):
    """
    Verify that an administrator can retrieve the registered user list
    and that password hashes are never included in the response.
    """

    # Retrieve the test client and admin authentication headers.
    client, admin_headers, user_headers = admin_test_clients

    # Request the first page of registered users.
    response = client.get(
        "/admin/users",
        headers=admin_headers
    )

    # Verify that the request succeeds.
    assert response.status_code == 200

    # Convert the JSON response into a Python list.
    users = response.json()

    # Verify that the endpoint returns a list.
    assert isinstance(users, list)

    # Verify that the temporary test administrator is included.
    assert any(
        user["name"] == "Temporary Admin"
        for user in users
    )

    # Verify that the temporary ordinary user is included.
    assert any(
        user["name"] == "Temporary User"
        for user in users
    )

    # Verify that no returned user contains a password hash.
    for user in users:
        assert "password_hash" not in user

        # Verify that only the intended safe user fields are returned.
        assert set(user.keys()) == {
            "id",
            "name",
            "email",
            "role"
        }


def test_admin_user_list_pagination(admin_test_clients):
    """
    Verify that skip and limit correctly control user-list pagination.
    """

    # Retrieve the test client and admin authentication headers.
    client, admin_headers, user_headers = admin_test_clients

    # Request the first user.
    first_response = client.get(
        "/admin/users?skip=0&limit=1",
        headers=admin_headers
    )

    # Request the second user.
    second_response = client.get(
        "/admin/users?skip=1&limit=1",
        headers=admin_headers
    )

    # Both requests should succeed.
    assert first_response.status_code == 200
    assert second_response.status_code == 200

    # Convert both responses into Python lists.
    first_page = first_response.json()
    second_page = second_response.json()

    # Each request should return exactly one user.
    assert len(first_page) == 1
    assert len(second_page) == 1

    # The two pages should contain different users.
    assert first_page[0]["id"] != second_page[0]["id"]

    # Verify that users are returned in ascending ID order.
    assert first_page[0]["id"] < second_page[0]["id"]


def test_admin_can_view_individual_user_without_password_hash(admin_test_clients):
    """
    Verify that an administrator can retrieve an individual user's details
    without exposing the password hash.
    """

    # Retrieve the test client and admin authentication headers.
    client, admin_headers, user_headers = admin_test_clients

    # Retrieve the user list to locate the temporary test user.
    list_response = client.get(
        "/admin/users",
        headers=admin_headers
    )

    # Verify that the user-list request succeeds.
    assert list_response.status_code == 200

    # Find the temporary ordinary user in the returned list.
    temporary_user = next(
        user for user in list_response.json()
        if user["name"] == "Temporary User"
    )

    # Extract the temporary user's database ID.
    user_id = temporary_user["id"]

    # Request the individual user's details.
    response = client.get(
        f"/admin/users/{user_id}",
        headers=admin_headers
    )

    # Verify that the individual-user request succeeds.
    assert response.status_code == 200

    # Convert the JSON response into a Python dictionary.
    user_data = response.json()

    # Verify that the correct user was returned.
    assert user_data["id"] == user_id
    assert user_data["name"] == "Temporary User"
    assert user_data["role"] == "user"

    # Verify that the password hash is never exposed.
    assert "password_hash" not in user_data

    # Verify that only safe user fields are returned.
    assert set(user_data.keys()) == {
        "id",
        "name",
        "email",
        "role"
    }


def test_admin_get_nonexistent_user_returns_404(admin_test_clients):
    """
    Verify that requesting a nonexistent user returns HTTP 404.
    """

    # Retrieve the test client and admin authentication headers.
    client, admin_headers, user_headers = admin_test_clients

    # A negative ID cannot match a valid auto-increment user ID.
    response = client.get(
        "/admin/users/-1",
        headers=admin_headers
    )

    # Verify that the API returns HTTP 404.
    assert response.status_code == 404

    # Verify the error message.
    assert response.json()["detail"] == "User not found"


def test_normal_user_cannot_access_user_management(admin_test_clients):
    """
    Verify that an ordinary authenticated user cannot list users
    or retrieve individual user details.
    """

    # Retrieve the test client and ordinary-user authentication headers.
    client, admin_headers, user_headers = admin_test_clients

    # Attempt to retrieve the registered user list.
    list_response = client.get(
        "/admin/users",
        headers=user_headers
    )

    # Attempt to retrieve an individual user.
    detail_response = client.get(
        "/admin/users/-1",
        headers=user_headers
    )

    # Both requests must be rejected with HTTP 403.
    assert list_response.status_code == 403
    assert detail_response.status_code == 403



# ---------------------------------------------------------
# ADMIN PREDICTION MONITORING TEST FIXTURE
# ---------------------------------------------------------

@pytest.fixture
def admin_prediction_test_data():
    """
    Create temporary accounts and prediction records for
    testing administrator prediction-monitoring endpoints.

    Only records created by this fixture are removed.
    Existing users, predictions and model metadata remain untouched.
    """

    # Open a database session.
    db = SessionLocal()

    # Generate unique email addresses for this test execution.
    unique_id = uuid4().hex

    # Keep track of records created by this fixture.
    created_user_ids = []
    created_prediction_ids = []

    try:
        # Retrieve the currently active model metadata record.
        # Prediction records require a valid model_id foreign key.
        active_model = (
            db.query(ModelMetadata)
            .filter(ModelMetadata.is_active.is_(True))
            .first()
        )

        # Stop the fixture if no active model exists.
        if active_model is None:
            raise RuntimeError(
                "No active model metadata found for prediction tests."
            )

        # Create a temporary administrator account.
        admin_user = User(
            name="Prediction Test Admin",
            email=f"prediction-admin-{unique_id}@example.com",
            password_hash=hash_password("AdminTest@123"),
            role="admin"
        )

        # Create a temporary ordinary-user account.
        normal_user = User(
            name="Prediction Test User",
            email=f"prediction-user-{unique_id}@example.com",
            password_hash=hash_password("UserTest@123"),
            role="user"
        )

        # Add both temporary accounts to the database session.
        db.add_all([admin_user, normal_user])

        # Save the temporary accounts.
        db.commit()

        # Retrieve their database-generated IDs.
        db.refresh(admin_user)
        db.refresh(normal_user)

        # Store only these temporary user IDs for cleanup.
        created_user_ids = [admin_user.id, normal_user.id]

        # Define three test predictions.
        # All records belong to the temporary ordinary user.
        test_predictions = [
            Prediction(
                user_id=normal_user.id,
                model_id=active_model.id,
                pregnancies=2,
                glucose=110,
                blood_pressure=70,
                skin_thickness=20,
                insulin=80,
                bmi=25.5,
                diabetes_pedigree_function=0.5,
                age=25,
                predicted_class=0,
                predicted_probability=0.25
            ),
            Prediction(
                user_id=normal_user.id,
                model_id=active_model.id,
                pregnancies=3,
                glucose=160,
                blood_pressure=80,
                skin_thickness=30,
                insulin=120,
                bmi=32.0,
                diabetes_pedigree_function=0.7,
                age=40,
                predicted_class=1,
                predicted_probability=0.80
            ),
            Prediction(
                user_id=normal_user.id,
                model_id=active_model.id,
                pregnancies=1,
                glucose=145,
                blood_pressure=75,
                skin_thickness=25,
                insulin=100,
                bmi=29.0,
                diabetes_pedigree_function=0.4,
                age=35,
                predicted_class=1,
                predicted_probability=0.75
            )
        ]

        # Add all three temporary predictions.
        db.add_all(test_predictions)

        # Save the prediction records.
        db.commit()

        # Retrieve their database-generated IDs.
        for prediction in test_predictions:
            db.refresh(prediction)

        # Store only the prediction IDs created by this fixture.
        created_prediction_ids = [
            prediction.id for prediction in test_predictions
        ]

        # Create an administrator JWT.
        admin_token = create_access_token(
            data={"sub": str(admin_user.id)}
        )

        # Create an ordinary-user JWT.
        user_token = create_access_token(
            data={"sub": str(normal_user.id)}
        )

        # Prepare the administrator authorization header.
        admin_headers = {
            "Authorization": f"Bearer {admin_token}"
        }

        # Prepare the ordinary-user authorization header.
        user_headers = {
            "Authorization": f"Bearer {user_token}"
        }

        # Close the setup database session.
        db.close()

        # Provide the test data and authentication details.
        with TestClient(app) as client:
            yield {
                "client": client,
                "admin_headers": admin_headers,
                "user_headers": user_headers,
                "user_id": normal_user.id,
                "prediction_ids": created_prediction_ids
            }

    except Exception:
        # Undo incomplete setup changes if an error occurs.
        db.rollback()
        raise

    finally:
        # Ensure the setup session is closed.
        db.close()

        # Open a fresh session for safe cleanup.
        cleanup_db = SessionLocal()

        try:
            # Delete only the prediction records created here.
            for prediction_id in created_prediction_ids:
                prediction = cleanup_db.get(
                    Prediction,
                    prediction_id
                )

                if prediction is not None:
                    cleanup_db.delete(prediction)

            # Delete only the temporary users created here.
            for user_id in created_user_ids:
                temporary_user = cleanup_db.get(User, user_id)

                if temporary_user is not None:
                    cleanup_db.delete(temporary_user)

            # Save cleanup changes.
            cleanup_db.commit()

        except Exception:
            # Undo cleanup changes if deletion fails.
            cleanup_db.rollback()
            raise

        finally:
            # Always close the cleanup session.
            cleanup_db.close()



# ---------------------------------------------------------
# ADMIN PREDICTION MONITORING TESTS
# ---------------------------------------------------------

def test_admin_can_list_predictions_without_patient_inputs(
    admin_prediction_test_data
):
    """
    Verify that administrators can list predictions
    without receiving patient medical input features.
    """

    # Retrieve the test client and administrator headers.
    client = admin_prediction_test_data["client"]
    admin_headers = admin_prediction_test_data["admin_headers"]
    prediction_ids = admin_prediction_test_data["prediction_ids"]

    # Request the system-wide prediction list.
    response = client.get(
        "/admin/predictions",
        headers=admin_headers
    )

    # Verify that the request succeeds.
    assert response.status_code == 200

    # Convert the response into a Python list.
    predictions = response.json()

    # Verify that all three temporary predictions are present.
    returned_ids = {item["id"] for item in predictions}

    assert set(prediction_ids).issubset(returned_ids)

    # Verify that each prediction contains only safe fields.
    expected_fields = {
        "id",
        "user_id",
        "model_id",
        "predicted_class",
        "predicted_probability",
        "created_at"
    }

    for prediction in predictions:
        assert set(prediction.keys()) == expected_fields

        # Explicitly verify that patient input fields are excluded.
        assert "glucose" not in prediction
        assert "bmi" not in prediction
        assert "insulin" not in prediction
        assert "age" not in prediction


def test_admin_prediction_pagination(admin_prediction_test_data):
    """
    Verify that skip and limit correctly paginate predictions.
    """

    # Retrieve the test client and administrator headers.
    client = admin_prediction_test_data["client"]
    admin_headers = admin_prediction_test_data["admin_headers"]
    prediction_ids = admin_prediction_test_data["prediction_ids"]

    # Request the first two prediction records.
    first_response = client.get(
        "/admin/predictions?skip=0&limit=2",
        headers=admin_headers
    )

    # Request the next two prediction records.
    second_response = client.get(
        "/admin/predictions?skip=2&limit=2",
        headers=admin_headers
    )

    # Both requests should succeed.
    assert first_response.status_code == 200
    assert second_response.status_code == 200

    # Extract the returned prediction lists.
    first_page = first_response.json()
    second_page = second_response.json()

    # Verify pagination limits.
    assert len(first_page) == 2
    assert len(second_page) == 2

    # Verify that the pages do not overlap.
    first_page_ids = {item["id"] for item in first_page}
    second_page_ids = {item["id"] for item in second_page}

    assert first_page_ids.isdisjoint(second_page_ids)

    # Verify that the three temporary predictions appear
    # across the two pages.
    returned_ids = first_page_ids.union(second_page_ids)

    assert set(prediction_ids).issubset(returned_ids)


def test_admin_can_filter_predictions(admin_prediction_test_data):
    """
    Verify filtering by predicted class and user ID.
    """

    # Retrieve the test client and administrator headers.
    client = admin_prediction_test_data["client"]
    admin_headers = admin_prediction_test_data["admin_headers"]
    user_id = admin_prediction_test_data["user_id"]
    prediction_ids = admin_prediction_test_data["prediction_ids"]

    # Retrieve only class 1 predictions.
    class_response = client.get(
        "/admin/predictions?predicted_class=1",
        headers=admin_headers
    )

    # Verify that the class filter succeeds.
    assert class_response.status_code == 200

    # All returned records must belong to class 1.
    class_predictions = class_response.json()

    # Only check predictions created by this test.
    test_class_predictions = [
        item
        for item in class_predictions
        if item["id"] in prediction_ids
    ]

    # Exactly two of the three temporary records should be Class 1.
    assert len(test_class_predictions) == 2

    assert all(
        item["predicted_class"] == 1
        for item in test_class_predictions
    )

    # Retrieve predictions belonging to the temporary user.
    user_response = client.get(
        f"/admin/predictions?user_id={user_id}",
        headers=admin_headers
    )

    # Verify that the user filter succeeds.
    assert user_response.status_code == 200

    user_predictions = user_response.json()

    # Verify that all three temporary records are returned.
    returned_ids = {item["id"] for item in user_predictions}

    assert set(prediction_ids).issubset(returned_ids)

    # Verify that returned records belong to the requested user.
    assert all(
        item["user_id"] == user_id
        for item in user_predictions
    )


def test_admin_can_view_individual_prediction(
    admin_prediction_test_data
):
    """
    Verify that an administrator can retrieve one prediction.
    """

    # Retrieve the test client and administrator headers.
    client = admin_prediction_test_data["client"]
    admin_headers = admin_prediction_test_data["admin_headers"]
    prediction_id = admin_prediction_test_data["prediction_ids"][0]

    # Request one prediction using its database ID.
    response = client.get(
        f"/admin/predictions/{prediction_id}",
        headers=admin_headers
    )

    # Verify that the request succeeds.
    assert response.status_code == 200

    # Extract the prediction response.
    prediction = response.json()

    # Verify the requested prediction ID.
    assert prediction["id"] == prediction_id

    # Verify that medical input features are excluded.
    assert "glucose" not in prediction
    assert "blood_pressure" not in prediction
    assert "bmi" not in prediction
    assert "insulin" not in prediction
    assert "age" not in prediction


def test_admin_get_nonexistent_prediction_returns_404(
    admin_prediction_test_data
):
    """
    Verify that a nonexistent prediction returns HTTP 404.
    """

    # Retrieve the test client and administrator headers.
    client = admin_prediction_test_data["client"]
    admin_headers = admin_prediction_test_data["admin_headers"]

    # Request an invalid prediction ID.
    response = client.get(
        "/admin/predictions/-1",
        headers=admin_headers
    )

    # Verify the HTTP 404 response.
    assert response.status_code == 404

    # Verify the error message.
    assert response.json()["detail"] == "Prediction not found"


def test_normal_user_cannot_access_prediction_monitoring(
    admin_prediction_test_data
):
    """
    Verify that ordinary users cannot access admin prediction APIs.
    """

    # Retrieve the test client and ordinary-user headers.
    client = admin_prediction_test_data["client"]
    user_headers = admin_prediction_test_data["user_headers"]
    prediction_id = admin_prediction_test_data["prediction_ids"][0]

    # Attempt to list all predictions.
    list_response = client.get(
        "/admin/predictions",
        headers=user_headers
    )

    # Attempt to retrieve an individual prediction.
    detail_response = client.get(
        f"/admin/predictions/{prediction_id}",
        headers=user_headers
    )

    # Both requests must be rejected.
    assert list_response.status_code == 403
    assert detail_response.status_code == 403


def test_unauthenticated_user_cannot_access_prediction_monitoring(
    admin_prediction_test_data
):
    """
    Verify that visitors without authentication cannot access
    administrator prediction-monitoring APIs.
    """

    # Retrieve the test client.
    client = admin_prediction_test_data["client"]

    # Request the prediction list without authentication.
    list_response = client.get("/admin/predictions")

    # Request an individual prediction without authentication.
    detail_response = client.get("/admin/predictions/-1")

    # Both requests must require authentication.
    assert list_response.status_code == 401
    assert detail_response.status_code == 401



# ---------------------------------------------------------
# ADMIN MODEL MANAGEMENT TEST FIXTURE
# ---------------------------------------------------------

@pytest.fixture
def admin_model_test_data(admin_test_clients):
    """
    Create temporary model metadata records for testing
    administrator model-management endpoints.

    Reuse the existing admin_test_clients fixture for authentication.

    Only model records created by this fixture are deleted.
    Existing model metadata records remain untouched.
    """

    # Reuse the existing test client and authentication headers.
    client, admin_headers, user_headers = admin_test_clients

    # Open a database session for creating temporary model records.
    db = SessionLocal()

    # Generate a unique identifier for this test execution.
    unique_id = uuid4().hex

    # Keep track of only the model IDs created by this fixture.
    created_model_ids = []

    try:

        # Create three temporary model metadata records.
        # All are inactive so existing active model records remain unchanged.
        test_models = [
            ModelMetadata(
                model_name=f"test-model-{unique_id}-{index}",
                version="1.0",
                algorithm="Logistic Regression",
                metrics={
                    "accuracy": 0.75,
                    "precision": 0.70,
                    "recall": 0.65,
                    "f1_score": 0.67,
                    "roc_auc": 0.80
                },
                is_active=False
            )
            for index in range(1, 4)
        ]

        # Add the temporary records to the database session.
        db.add_all(test_models)

        # Save the temporary records.
        db.commit()

        # Retrieve database-generated IDs for safe cleanup.
        for model_record in test_models:
            db.refresh(model_record)

        # Store only the IDs created by this fixture.
        created_model_ids = [
            model_record.id for model_record in test_models
        ]

    except Exception:

        # Undo incomplete database changes if setup fails.
        db.rollback()
        raise

    finally:

        # Close the database session used for fixture setup.
        db.close()

    try:

        # Provide the test client, authentication headers,
        # generated model IDs, and unique identifier to each test.
        yield {
            "client": client,
            "admin_headers": admin_headers,
            "user_headers": user_headers,
            "model_ids": created_model_ids,
            "unique_id": unique_id
        }

    finally:

        # Open a fresh database session for safe cleanup.
        cleanup_db = SessionLocal()

        try:

            # Delete only the temporary model records created above.
            for model_id in created_model_ids:

                model_record = cleanup_db.get(
                    ModelMetadata,
                    model_id
                )

                if model_record is not None:
                    cleanup_db.delete(model_record)

            # Save the cleanup changes.
            cleanup_db.commit()

        except Exception:

            # Undo cleanup changes if deletion fails.
            cleanup_db.rollback()
            raise

        finally:

            # Always close the cleanup database session.
            cleanup_db.close()



# ---------------------------------------------------------
# ADMIN MODEL LISTING TESTS
# ---------------------------------------------------------

def test_admin_can_list_registered_models(admin_model_test_data):
    """
    Verify that an administrator can list registered models.
    """

    # Retrieve the test client and administrator headers.
    client = admin_model_test_data["client"]
    admin_headers = admin_model_test_data["admin_headers"]
    model_ids = admin_model_test_data["model_ids"]

    # Request the registered model list.
    response = client.get(
        "/admin/models",
        headers=admin_headers
    )

    # Verify that the request succeeds.
    assert response.status_code == 200

    # Extract the returned model records.
    models = response.json()

    # Verify that all three temporary models are present.
    returned_ids = {model["id"] for model in models}
    assert set(model_ids).issubset(returned_ids)

    # Verify that model responses contain only the expected fields.
    expected_fields = {
        "id",
        "model_name",
        "version",
        "algorithm",
        "is_active",
        "created_at"
    }

    for model in models:
        assert set(model.keys()) == expected_fields

        # Verify that temporary models remain inactive.
        if model["id"] in model_ids:
            assert model["is_active"] is False


def test_admin_model_list_pagination(admin_model_test_data):
    """
    Verify that skip and limit correctly paginate model records.
    """

    # Retrieve the test client and administrator headers.
    client = admin_model_test_data["client"]
    admin_headers = admin_model_test_data["admin_headers"]

    # Request the first two model records.
    first_response = client.get(
        "/admin/models?skip=0&limit=2",
        headers=admin_headers
    )

    # Request the next two model records.
    second_response = client.get(
        "/admin/models?skip=2&limit=2",
        headers=admin_headers
    )

    # Both requests should succeed.
    assert first_response.status_code == 200
    assert second_response.status_code == 200

    # Extract both pages.
    first_page = first_response.json()
    second_page = second_response.json()

    # Verify the pagination limits.
    assert len(first_page) == 2
    assert len(second_page) == 2

    # Verify that the pages do not overlap.
    first_page_ids = {model["id"] for model in first_page}
    second_page_ids = {model["id"] for model in second_page}

    assert first_page_ids.isdisjoint(second_page_ids)


# ---------------------------------------------------------
# ADMIN INDIVIDUAL MODEL DETAILS TESTS
# ---------------------------------------------------------

def test_admin_can_view_individual_model_details(admin_model_test_data):
    """
    Verify that an administrator can retrieve model details,
    including its evaluation metrics.
    """

    # Retrieve the test client and administrator headers.
    client = admin_model_test_data["client"]
    admin_headers = admin_model_test_data["admin_headers"]
    model_id = admin_model_test_data["model_ids"][0]

    # Request the individual model details.
    response = client.get(
        f"/admin/models/{model_id}",
        headers=admin_headers
    )

    # Verify that the request succeeds.
    assert response.status_code == 200

    # Extract the model information.
    model = response.json()

    # Verify the model identity and metadata.
    assert model["id"] == model_id
    assert model["version"] == "1.0"
    assert model["algorithm"] == "Logistic Regression"

    # Verify that evaluation metrics are included.
    assert model["metrics"]["accuracy"] == 0.75
    assert model["metrics"]["precision"] == 0.70
    assert model["metrics"]["recall"] == 0.65
    assert model["metrics"]["f1_score"] == 0.67
    assert model["metrics"]["roc_auc"] == 0.80

    # Verify that the model remains inactive.
    assert model["is_active"] is False


def test_admin_get_nonexistent_model_returns_404(admin_model_test_data):
    """
    Verify that requesting a nonexistent model returns HTTP 404.
    """

    # Retrieve the test client and administrator headers.
    client = admin_model_test_data["client"]
    admin_headers = admin_model_test_data["admin_headers"]

    # Request an invalid model ID.
    response = client.get(
        "/admin/models/-1",
        headers=admin_headers
    )

    # Verify the HTTP 404 response.
    assert response.status_code == 404

    # Verify the error message.
    assert response.json()["detail"] == "Model not found"



# ---------------------------------------------------------
# ADMIN MODEL REGISTRATION TESTS
# ---------------------------------------------------------

def test_admin_can_register_model_metadata(admin_model_test_data):
    """
    Verify that an administrator can register new model metadata.

    The registered model must remain inactive by default.
    The fixture will clean up the new record after the test.
    """

    # Retrieve the test client and administrator headers.
    client = admin_model_test_data["client"]
    admin_headers = admin_model_test_data["admin_headers"]

    # Retrieve the mutable list used by fixture cleanup.
    model_ids = admin_model_test_data["model_ids"]

    # Prepare unique model metadata for this test.
    model_data = {
        "model_name": f"registration-test-{admin_model_test_data['unique_id']}",
        "version": "1.0",
        "algorithm": "Logistic Regression",
        "metrics": {
            "accuracy": 0.80,
            "precision": 0.75,
            "recall": 0.70,
            "f1_score": 0.72,
            "roc_auc": 0.84
        }
    }

    # Send a POST request to register the model metadata.
    response = client.post(
        "/admin/models",
        json=model_data,
        headers=admin_headers
    )

    # Verify that the model was created successfully.
    assert response.status_code == 201

    # Extract the registered model information.
    registered_model = response.json()

    # Add its ID to the fixture cleanup list.
    # This prevents test records from remaining in the database.
    model_ids.append(registered_model["id"])

    # Verify the registered model's metadata.
    assert registered_model["model_name"] == model_data["model_name"]
    assert registered_model["version"] == "1.0"
    assert registered_model["algorithm"] == "Logistic Regression"

    # Verify that the evaluation metrics were saved.
    assert registered_model["metrics"] == model_data["metrics"]

    # Verify that newly registered models are inactive.
    assert registered_model["is_active"] is False


def test_duplicate_model_registration_returns_409(admin_model_test_data):
    """
    Verify that duplicate model_name and version combinations
    are rejected by the API.
    """

    # Retrieve the test client and administrator headers.
    client = admin_model_test_data["client"]
    admin_headers = admin_model_test_data["admin_headers"]

    # Prepare metadata matching the first existing test model.
    duplicate_model_data = {
        "model_name": (
            f"test-model-{admin_model_test_data['unique_id']}-1"
        ),
        "version": "1.0",
        "algorithm": "Logistic Regression",
        "metrics": {
            "accuracy": 0.80,
            "precision": 0.75,
            "recall": 0.70,
            "f1_score": 0.72,
            "roc_auc": 0.84
        }
    }

    # Attempt to register the duplicate model.
    response = client.post(
        "/admin/models",
        json=duplicate_model_data,
        headers=admin_headers
    )

    # Verify that the API rejects the duplicate.
    assert response.status_code == 409


def test_invalid_model_registration_data_is_rejected(
    admin_model_test_data
):
    """
    Verify that invalid model metadata is rejected
    by Pydantic request validation.
    """

    # Retrieve the test client and administrator headers.
    client = admin_model_test_data["client"]
    admin_headers = admin_model_test_data["admin_headers"]

    # Prepare invalid data with an empty model name and metrics.
    invalid_model_data = {
        "model_name": "",
        "version": "1.0",
        "algorithm": "Logistic Regression",
        "metrics": {}
    }

    # Attempt to register invalid model metadata.
    response = client.post(
        "/admin/models",
        json=invalid_model_data,
        headers=admin_headers
    )

    # Verify that FastAPI returns a validation error.
    assert response.status_code == 422


# ---------------------------------------------------------
# ADMIN MODEL MANAGEMENT AUTHORIZATION TESTS
# ---------------------------------------------------------

def test_normal_user_cannot_access_model_management(
    admin_model_test_data
):
    """
    Verify that an ordinary authenticated user cannot
    access administrator model-management endpoints.
    """

    # Retrieve the test client and ordinary user headers.
    client = admin_model_test_data["client"]
    user_headers = admin_model_test_data["user_headers"]

    # Attempt to list registered models.
    list_response = client.get(
        "/admin/models",
        headers=user_headers
    )

    # Attempt to view an individual model.
    model_id = admin_model_test_data["model_ids"][0]

    detail_response = client.get(
        f"/admin/models/{model_id}",
        headers=user_headers
    )

    # Attempt to register new model metadata.
    registration_response = client.post(
        "/admin/models",
        json={
            "model_name": "unauthorized-test-model",
            "version": "1.0",
            "algorithm": "Logistic Regression",
            "metrics": {"accuracy": 0.80}
        },
        headers=user_headers
    )

    # All three operations must be forbidden.
    assert list_response.status_code == 403
    assert detail_response.status_code == 403
    assert registration_response.status_code == 403


def test_unauthenticated_user_cannot_access_model_management(
    admin_model_test_data
):
    """
    Verify that visitors without authentication cannot
    access administrator model-management endpoints.
    """

    # Retrieve the test client.
    client = admin_model_test_data["client"]

    # Attempt to list registered models without a token.
    list_response = client.get("/admin/models")

    # Attempt to view model details without a token.
    model_id = admin_model_test_data["model_ids"][0]

    detail_response = client.get(
        f"/admin/models/{model_id}"
    )

    # Attempt to register model metadata without a token.
    registration_response = client.post(
        "/admin/models",
        json={
            "model_name": "unauthenticated-test-model",
            "version": "1.0",
            "algorithm": "Logistic Regression",
            "metrics": {"accuracy": 0.80}
        }
    )

    # All three operations must require authentication.
    assert list_response.status_code == 401
    assert detail_response.status_code == 401
    assert registration_response.status_code == 401
