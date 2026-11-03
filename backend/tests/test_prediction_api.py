# Import uuid4 to generate a unique email address for every test user.
from uuid import uuid4

# Import pytest to create reusable test fixtures.
import pytest

# Import TestClient to send simulated HTTP requests to FastAPI.
from fastapi.testclient import TestClient

# Import delete and select to execute database queries.
from sqlalchemy import delete, select

# Import the FastAPI application being tested.
from backend.app.main import app

# Import the SQLAlchemy session factory.
from backend.app.db.database import SessionLocal

# Import the User and Prediction database models.
from backend.app.models import User, Prediction

# Import password hashing and JWT creation functions.
from backend.app.core.security import hash_password, create_access_token


# Store sample patient data containing all eight required features.
valid_data = {
    "pregnancies": 2,
    "glucose": 120,
    "blood_pressure": 70,
    "skin_thickness": 25,
    "insulin": 100,
    "bmi": 28.5,
    "diabetes_pedigree_function": 0.45,
    "age": 30
}


# Create a reusable fixture for authenticated API tests.
@pytest.fixture
def authenticated_client():

    # Open a database session for creating the temporary test user.
    db = SessionLocal()

    # Generate a unique email so repeated test runs do not conflict.
    unique_email = f"prediction-test-{uuid4().hex}@example.com"

    # Create a temporary user with a securely hashed password.
    test_user = User(
        name="Prediction API Test",
        email=unique_email,
        password_hash=hash_password("TestPassword123!"),
        role="user"
    )

    # Add the temporary user to the database session.
    db.add(test_user)

    # Save the user in MySQL.
    db.commit()

    # Refresh the user object to retrieve its generated database ID.
    db.refresh(test_user)

    # Store the generated user ID for authentication and cleanup.
    test_user_id = test_user.id

    # Close the database session used to create the user.
    db.close()

    # Generate a JWT containing the temporary user's ID.
    access_token = create_access_token(
        data={"sub": str(test_user_id)}
    )

    # Prepare the HTTP Authorization header with the Bearer token.
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    # Open a test client for the FastAPI application.
    try:
        with TestClient(app) as client:

            # Provide the client, authentication headers and user ID to the test.
            yield client, headers, test_user_id

    # Always execute cleanup after the test finishes.
    finally:

        # Open a new database session for cleaning up test records.
        cleanup_db = SessionLocal()

        try:
            # Delete only prediction records belonging to this temporary user.
            cleanup_db.execute(
                delete(Prediction).where(
                    Prediction.user_id == test_user_id
                )
            )

            # Delete only the temporary test user.
            cleanup_db.execute(
                delete(User).where(
                    User.id == test_user_id
                )
            )

            # Save the cleanup operations in MySQL.
            cleanup_db.commit()

        finally:
            # Close the cleanup database session.
            cleanup_db.close()


# TEST 1: Verify that an authenticated user can generate a prediction.
def test_valid_prediction(authenticated_client):

    # Receive the test client, authentication headers and temporary user ID.
    client, headers, test_user_id = authenticated_client

    # Send a POST request with valid patient data and a JWT token.
    response = client.post(
        "/predictions/predict",
        json=valid_data,
        headers=headers
    )

    # Confirm that the API returned HTTP 200.
    assert response.status_code == 200

    # Open a database session to verify that the prediction was saved.
    db = SessionLocal()

    try:
        # Find prediction records belonging to this test user.
        saved_predictions = db.scalars(
            select(Prediction).where(
                Prediction.user_id == test_user_id
            )
        ).all()

        # Confirm that exactly one prediction was saved.
        assert len(saved_predictions) == 1

    finally:
        # Close the database session after verification.
        db.close()


# TEST 2: Verify that the API returns all expected response fields.
def test_prediction_response_fields(authenticated_client):

    # Receive the authenticated test client and its headers.
    client, headers, _ = authenticated_client

    # Send a prediction request using valid patient data.
    response = client.post(
        "/predictions/predict",
        json=valid_data,
        headers=headers
    )

    # Convert the JSON response into a Python dictionary.
    result = response.json()

    # Confirm that the response contains the predicted class.
    assert "predicted_class" in result

    # Confirm that the response contains the class 0 probability.
    assert "probability_class_0" in result

    # Confirm that the response contains the class 1 probability.
    assert "probability_class_1" in result

    # Confirm that the response contains the predicted class probability.
    assert "predicted_class_probability" in result


# TEST 3: Verify that the returned probabilities are valid.
def test_prediction_probabilities(authenticated_client):

    # Receive the authenticated test client and its headers.
    client, headers, _ = authenticated_client

    # Send a prediction request using valid patient data.
    response = client.post(
        "/predictions/predict",
        json=valid_data,
        headers=headers
    )

    # Convert the JSON response into a Python dictionary.
    result = response.json()

    # Extract the probability of class 0.
    probability_0 = result["probability_class_0"]

    # Extract the probability of class 1.
    probability_1 = result["probability_class_1"]

    # Extract the predicted class.
    predicted_class = result["predicted_class"]

    # Confirm that class 0 probability is between 0 and 1.
    assert 0 <= probability_0 <= 1

    # Confirm that class 1 probability is between 0 and 1.
    assert 0 <= probability_1 <= 1

    # Confirm that both class probabilities add up approximately to 1.
    assert abs(probability_0 + probability_1 - 1) < 0.000001

    # Confirm that the predicted class is either 0 or 1.
    assert predicted_class in [0, 1]

    # Confirm that the predicted class probability is valid.
    assert 0 <= result["predicted_class_probability"] <= 1

    # Confirm that the returned predicted probability matches its class.
    if predicted_class == 0:
        assert result["predicted_class_probability"] == pytest.approx(
            probability_0
        )
    else:
        assert result["predicted_class_probability"] == pytest.approx(
            probability_1
        )


# TEST 4: Verify that negative glucose is rejected.
def test_invalid_glucose(authenticated_client):

    # Receive the authenticated test client and its headers.
    client, headers, _ = authenticated_client

    # Create a separate copy of the valid patient data.
    invalid_data = valid_data.copy()

    # Replace the glucose value with a negative number.
    invalid_data["glucose"] = -10

    # Send the invalid data with a valid JWT token.
    response = client.post(
        "/predictions/predict",
        json=invalid_data,
        headers=headers
    )

    # Confirm that Pydantic rejects the invalid request with HTTP 422.
    assert response.status_code == 422


# TEST 5: Verify that requests without a JWT token are rejected.
def test_missing_authentication():

    # Create a test client without an Authorization header.
    with TestClient(app) as client:

        # Send a prediction request without authentication credentials.
        response = client.post(
            "/predictions/predict",
            json=valid_data
        )

        # Confirm that the API returns HTTP 401.
        assert response.status_code == 401


# TEST 6: Verify that an invalid JWT token is rejected.
def test_invalid_authentication_token():

    # Create a test client for the FastAPI application.
    with TestClient(app) as client:

        # Prepare an Authorization header containing an invalid token.
        invalid_headers = {
            "Authorization": "Bearer invalid.test.token"
        }

        # Send a prediction request with the invalid token.
        response = client.post(
            "/predictions/predict",
            json=valid_data,
            headers=invalid_headers
        )

        # Confirm that the API returns HTTP 401.
        assert response.status_code == 401


# TEST 7: Verify that a new user receives an empty prediction history.
def test_empty_prediction_history(authenticated_client):

    # Receive the authenticated test client and its headers.
    client, headers, _ = authenticated_client

    # Request the prediction history of the authenticated user.
    response = client.get(
        "/predictions/history",
        headers=headers
    )

    # Confirm that the API returned HTTP 200.
    assert response.status_code == 200

    # Confirm that the prediction history is empty.
    assert response.json() == []


# TEST 8: Verify that a user can see their own prediction history.
def test_user_can_see_own_prediction_history(authenticated_client):

    # Receive the authenticated test client and its headers.
    client, headers, test_user_id = authenticated_client

    # Generate a prediction for the authenticated user.
    prediction_response = client.post(
        "/predictions/predict",
        json=valid_data,
        headers=headers
    )

    # Confirm that the prediction was generated successfully.
    assert prediction_response.status_code == 200

    # Request the prediction history of the same user.
    history_response = client.get(
        "/predictions/history",
        headers=headers
    )

    # Confirm that the history request was successful.
    assert history_response.status_code == 200

    # Convert the history response into a Python list.
    history = history_response.json()

    # Confirm that exactly one prediction exists.
    assert len(history) == 1

    # Confirm that the returned prediction belongs to this user.
    db = SessionLocal()

    try:
        saved_prediction = db.scalar(
            select(Prediction).where(
                Prediction.user_id == test_user_id
            )
        )

        assert saved_prediction is not None
        assert history[0]["id"] == saved_prediction.id

    finally:
        db.close()


# TEST 9: Verify that users cannot access another user's predictions.
def test_user_cannot_see_another_users_predictions():

    db = SessionLocal()

    # Generate unique email addresses for both temporary users.
    user_a_email = f"user-a-{uuid4().hex}@example.com"
    user_b_email = f"user-b-{uuid4().hex}@example.com"

    user_a_id = None
    user_b_id = None

    try:
        # Create User A.
        user_a = User(
            name="Test User A",
            email=user_a_email,
            password_hash=hash_password("TestPassword123!"),
            role="user"
        )

        # Create User B.
        user_b = User(
            name="Test User B",
            email=user_b_email,
            password_hash=hash_password("TestPassword123!"),
            role="user"
        )

        # Add both users to the database.
        db.add_all([user_a, user_b])
        db.commit()

        # Retrieve their generated database IDs.
        db.refresh(user_a)
        db.refresh(user_b)

        user_a_id = user_a.id
        user_b_id = user_b.id

        # Create a JWT token for User A.
        token_a = create_access_token(
            data={"sub": str(user_a_id)}
        )

        # Create a JWT token for User B.
        token_b = create_access_token(
            data={"sub": str(user_b_id)}
        )

        # Prepare separate authentication headers.
        headers_a = {
            "Authorization": f"Bearer {token_a}"
        }

        headers_b = {
            "Authorization": f"Bearer {token_b}"
        }

        # Create a prediction belonging to User A.
        with TestClient(app) as client:

            prediction_response = client.post(
                "/predictions/predict",
                json=valid_data,
                headers=headers_a
            )

            # Confirm that User A's prediction was successful.
            assert prediction_response.status_code == 200

            # Request prediction history as User B.
            history_response = client.get(
                "/predictions/history",
                headers=headers_b
            )

            # Confirm that User B's history request succeeded.
            assert history_response.status_code == 200

            # Confirm that User B cannot see User A's prediction.
            assert history_response.json() == []

    finally:
        # Delete only the temporary users and their prediction records.
        if user_a_id is not None:
            db.execute(
                delete(Prediction).where(
                    Prediction.user_id == user_a_id
                )
            )

            db.execute(
                delete(User).where(
                    User.id == user_a_id
                )
            )

        if user_b_id is not None:
            db.execute(
                delete(Prediction).where(
                    Prediction.user_id == user_b_id
                )
            )

            db.execute(
                delete(User).where(
                    User.id == user_b_id
                )
            )

        db.commit()
        db.close()


# ============================================================
# TEST 10: VERIFY EMPTY PREDICTION DASHBOARD
# ============================================================

def test_empty_prediction_dashboard(authenticated_client):

    # Receive the authenticated test client and its headers.
    client, headers, _ = authenticated_client

    # Request dashboard statistics for the newly created test user.
    response = client.get(
        "/predictions/dashboard",
        headers=headers
    )

    # Confirm that the dashboard request was successful.
    assert response.status_code == 200

    # Convert the JSON response into a Python dictionary.
    dashboard = response.json()

    # Verify that the user has no prediction records.
    assert dashboard["total_predictions"] == 0

    # Verify that both prediction classes have zero records.
    assert dashboard["class_0_predictions"] == 0
    assert dashboard["class_1_predictions"] == 0

    # Verify that both percentages are zero.
    assert dashboard["class_0_percentage"] == 0
    assert dashboard["class_1_percentage"] == 0

    # Verify that no latest prediction timestamp exists.
    assert dashboard["latest_prediction"] is None


# ============================================================
# TEST 11: VERIFY DASHBOARD STATISTICS AFTER A PREDICTION
# ============================================================

def test_prediction_dashboard_statistics(authenticated_client):

    # Receive the authenticated test client and its headers.
    client, headers, _ = authenticated_client

    # Generate a prediction using the ML prediction endpoint.
    prediction_response = client.post(
        "/predictions/predict",
        json=valid_data,
        headers=headers
    )

    # Confirm that the prediction request was successful.
    assert prediction_response.status_code == 200

    # Convert the prediction response into a Python dictionary.
    prediction = prediction_response.json()

    # Store the predicted class returned by the ML model.
    predicted_class = prediction["predicted_class"]

    # Request dashboard statistics after generating the prediction.
    dashboard_response = client.get(
        "/predictions/dashboard",
        headers=headers
    )

    # Confirm that the dashboard request was successful.
    assert dashboard_response.status_code == 200

    # Convert the dashboard response into a Python dictionary.
    dashboard = dashboard_response.json()

    # Verify that exactly one prediction has been recorded.
    assert dashboard["total_predictions"] == 1

    # Verify that the prediction was counted in the correct class.
    assert dashboard["class_0_predictions"] == (
        1 if predicted_class == 0 else 0
    )

    assert dashboard["class_1_predictions"] == (
        1 if predicted_class == 1 else 0
    )

    # Verify that the percentages match the predicted class.
    assert dashboard["class_0_percentage"] == (
        100.0 if predicted_class == 0 else 0.0
    )

    assert dashboard["class_1_percentage"] == (
        100.0 if predicted_class == 1 else 0.0
    )

    # Verify that the latest prediction timestamp is available.
    assert dashboard["latest_prediction"] is not None


# ============================================================
# TEST 12: VERIFY DASHBOARD USER DATA ISOLATION
# ============================================================

def test_user_cannot_see_another_users_dashboard_stats(authenticated_client):

    # Receive the authenticated client and User A's headers.
    client, headers_a, _ = authenticated_client

    # Open a database session to create another temporary user.
    db = SessionLocal()

    # Initialize the second user's ID for safe cleanup.
    user_b_id = None

    try:
        # Create User B with a unique email address.
        user_b = User(
            name="Dashboard Isolation Test User",
            email=f"dashboard-user-b-{uuid4().hex}@example.com",
            password_hash=hash_password("TestPassword123!"),
            role="user"
        )

        # Add User B to the database session.
        db.add(user_b)

        # Save User B in MySQL.
        db.commit()

        # Retrieve the generated database ID.
        db.refresh(user_b)

        # Store User B's ID for authentication and cleanup.
        user_b_id = user_b.id

        # Generate a JWT token for User B.
        token_b = create_access_token(
            data={"sub": str(user_b_id)}
        )

        # Prepare User B's Authorization header.
        headers_b = {
            "Authorization": f"Bearer {token_b}"
        }

        # Generate a prediction belonging to User A.
        prediction_response = client.post(
            "/predictions/predict",
            json=valid_data,
            headers=headers_a
        )

        # Confirm that User A's prediction was successful.
        assert prediction_response.status_code == 200

        # Request dashboard statistics as User B.
        dashboard_response = client.get(
            "/predictions/dashboard",
            headers=headers_b
        )

        # Confirm that User B's dashboard request was successful.
        assert dashboard_response.status_code == 200

        # Convert the dashboard response into a Python dictionary.
        dashboard = dashboard_response.json()

        # Verify that User B cannot see User A's prediction.
        assert dashboard["total_predictions"] == 0

        # Verify that both class counts remain zero.
        assert dashboard["class_0_predictions"] == 0
        assert dashboard["class_1_predictions"] == 0

        # Verify that both percentages remain zero.
        assert dashboard["class_0_percentage"] == 0
        assert dashboard["class_1_percentage"] == 0

        # Verify that User B has no latest prediction timestamp.
        assert dashboard["latest_prediction"] is None

    finally:
        # Clean up only User B's temporary database records.
        if user_b_id is not None:

            # Delete any prediction records belonging to User B.
            db.execute(
                delete(Prediction).where(
                    Prediction.user_id == user_b_id
                )
            )

            # Delete only User B.
            db.execute(
                delete(User).where(
                    User.id == user_b_id
                )
            )

            # Save the cleanup operations.
            db.commit()

        # Close the database session.
        db.close()