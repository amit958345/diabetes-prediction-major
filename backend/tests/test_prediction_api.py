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

# Import the database models required for prediction API tests.
from backend.app.models import (
    User,                   # Represents registered application users.
    Prediction,             # Stores generated diabetes predictions.
    PredictionExplanation   # Stores SHAP explanations linked to predictions.
)

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



# TEST 1: Verify that an authenticated user can generate a prediction
# and that both the prediction and its SHAP explanation are saved.
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

    # Open a database session to verify that both records were saved.
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

        # Retrieve the saved prediction record.
        saved_prediction = saved_predictions[0]

        # Find the SHAP explanation linked to this prediction.
        saved_explanation = db.scalars(
            select(PredictionExplanation).where(
                PredictionExplanation.prediction_id == saved_prediction.id
            )
        ).first()

        # Confirm that the SHAP explanation was saved.
        assert saved_explanation is not None

        # Confirm that the explanation method is SHAP.
        assert saved_explanation.explanation_method == "SHAP"

        # Confirm that the baseline value is stored.
        assert isinstance(saved_explanation.base_value, float)

        # Confirm that the model output is stored.
        assert isinstance(saved_explanation.model_output, float)

        # Confirm that all eight feature contributions are stored.
        assert len(saved_explanation.feature_contributions) == 8

        # Confirm that the explanation belongs to the correct prediction.
        assert saved_explanation.prediction_id == saved_prediction.id

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

    # --------------------------------------------------------
    # VERIFY THE SHAP EXPLANATION IN THE API RESPONSE
    # --------------------------------------------------------

    # Confirm that the response includes the SHAP explanation.
    assert "explanation" in result

    # Retrieve the SHAP explanation as a Python dictionary.
    explanation = result["explanation"]

    # Confirm that the explanation contains its baseline value.
    assert "base_value" in explanation

    # Confirm that the explanation contains the model output.
    assert "model_output" in explanation

    # Confirm that individual feature contributions are included.
    assert "feature_contributions" in explanation

    # Retrieve the feature-level SHAP contributions.
    contributions = explanation["feature_contributions"]

    # Define the eight features used by our trained ML model.
    expected_features = {
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    }

    # Verify that all eight expected features are explained.
    assert set(contributions.keys()) == expected_features

    # Verify that every SHAP contribution is a numeric value.
    assert all(
        isinstance(value, (int, float))
        for value in contributions.values()
    )

    # Verify that the baseline and model output are numeric.
    assert isinstance(explanation["base_value"], (int, float))
    assert isinstance(explanation["model_output"], (int, float))


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

# ===========================================================================
# TEST 13: Verify that a prediction is rolled back if SHAP explanation fails.
# ===========================================================================

def test_prediction_rollback_when_shap_fails(authenticated_client, monkeypatch):

    # Import the prediction API module so its SHAP function can be patched.
    import backend.app.api.predictions as prediction_api

    # Receive the authenticated test client, headers and temporary user ID.
    client, headers, test_user_id = authenticated_client

    # Define a replacement SHAP function that simulates an unexpected failure.
    def simulate_shap_failure(patient_data):

        # Raise an exception to simulate SHAP explanation generation failure.
        raise RuntimeError("Simulated SHAP explanation failure")

    # Replace the actual SHAP function used by the prediction API.
    monkeypatch.setattr(
        prediction_api,
        "explain_prediction",
        simulate_shap_failure
    )

    # Send a valid prediction request while SHAP generation is set to fail.
    response = client.post(
        "/predictions/predict",
        json=valid_data,
        headers=headers
    )

    # Confirm that the API returns HTTP 500 for the unexpected failure.
    assert response.status_code == 500

    # Open a database session to verify transaction rollback.
    db = SessionLocal()

    try:
        # Find predictions belonging to this temporary test user.
        saved_predictions = db.scalars(
            select(Prediction).where(
                Prediction.user_id == test_user_id
            )
        ).all()

        # Confirm that no incomplete prediction was saved.
        assert len(saved_predictions) == 0

    finally:
        # Close the database session after verification.
        db.close()


# ====================================================================
# TEST 14: Verify that a user can retrieve their own SHAP explanation.
# ====================================================================

def test_user_can_retrieve_own_shap_explanation(authenticated_client):

    # Receive the authenticated test client, headers and temporary user ID.
    client, headers, test_user_id = authenticated_client

    # Generate a prediction so that a SHAP explanation is saved.
    prediction_response = client.post(
        "/predictions/predict",
        json=valid_data,
        headers=headers
    )

    # Confirm that prediction generation was successful.
    assert prediction_response.status_code == 200

    # Open a database session to retrieve the generated prediction ID.
    db = SessionLocal()

    try:
        # Find the prediction belonging to the authenticated test user.
        saved_prediction = db.scalars(
            select(Prediction).where(
                Prediction.user_id == test_user_id
            )
        ).first()

        # Confirm that the prediction exists.
        assert saved_prediction is not None

        # Store the generated prediction ID.
        prediction_id = saved_prediction.id

    finally:
        # Close the database session after retrieving the prediction ID.
        db.close()

    # Request the SHAP explanation using the generated prediction ID.
    explanation_response = client.get(
        f"/predictions/{prediction_id}/explanation",
        headers=headers
    )

    # Confirm that the explanation endpoint returned HTTP 200.
    assert explanation_response.status_code == 200

    # Convert the response JSON into a Python dictionary.
    explanation = explanation_response.json()

    # Verify that the baseline value is returned.
    assert "base_value" in explanation

    # Verify that the model output is returned.
    assert "model_output" in explanation

    # Verify that feature contributions are returned.
    assert "feature_contributions" in explanation

    # Confirm that all eight feature contributions are available.
    assert len(explanation["feature_contributions"]) == 8


# ============================================================
# TEST 15: PREVENT USERS FROM ACCESSING ANOTHER USER'S SHAP DATA
# ============================================================

def test_user_cannot_retrieve_another_users_shap_explanation(authenticated_client):

    # Receive User B's authenticated client and headers.
    # The fixture also handles cleanup of User B.
    client, headers_b, test_user_b_id = authenticated_client

    # Open a database session to create and clean up User A.
    db = SessionLocal()

    # Store User A's ID so cleanup can safely identify this user.
    user_a_id = None

    try:
        # Create a unique email address for temporary User A.
        user_a_email = f"user-a-shap-{uuid4().hex}@example.com"

        # Create User A in the database.
        user_a = User(
            name="Test User A",
            email=user_a_email,
            password_hash=hash_password("TestPassword123!"),
            role="user"
        )

        # Save User A to the database.
        db.add(user_a)
        db.commit()

        # Refresh the object to retrieve its generated database ID.
        db.refresh(user_a)

        # Store User A's unique ID for authentication and cleanup.
        user_a_id = user_a.id

        # Generate a JWT access token for User A.
        token_a = create_access_token(
            data={"sub": str(user_a_id)}
        )

        # Prepare User A's authentication headers.
        headers_a = {
            "Authorization": f"Bearer {token_a}"
        }

        # Commit the current transaction to end its existing database snapshot.
        # This allows the next database query to see newly committed API records.
        db.commit()

        # Generate a prediction while authenticated as User A.
        prediction_response = client.post(
            "/predictions/predict",
            json=valid_data,
            headers=headers_a
        )

        # Confirm that User A's prediction was successfully created.
        assert prediction_response.status_code == 200

        # Retrieve the prediction belonging to User A.
        saved_prediction = db.scalars(
            select(Prediction).where(
                Prediction.user_id == user_a_id
            )
        ).first()

        # Confirm that User A's prediction exists.
        assert saved_prediction is not None

        # Store the prediction ID for the unauthorized access test.
        prediction_id = saved_prediction.id

        # Attempt to retrieve User A's SHAP explanation as User B.
        explanation_response = client.get(
            f"/predictions/{prediction_id}/explanation",
            headers=headers_b
        )

        # Confirm that User B receives HTTP 404 instead of the explanation.
        assert explanation_response.status_code == 404

        # Confirm that the API returns the expected error message.
        assert explanation_response.json()["detail"] == "Prediction not found."

    finally:
        # Delete only User A's temporary prediction records.
        if user_a_id is not None:
            db.execute(
                delete(Prediction).where(
                    Prediction.user_id == user_a_id
                )
            )

            # Delete only the temporary User A account.
            db.execute(
                delete(User).where(
                    User.id == user_a_id
                )
            )

            # Commit the cleanup operations.
            db.commit()

        # Close the database session.
        db.close()


# ============================================================
# TEST 16: RETURN 404 FOR A NONEXISTENT PREDICTION EXPLANATION
# ============================================================

def test_nonexistent_prediction_shap_explanation_returns_404(authenticated_client):

    # Receive an authenticated test client and its authorization headers.
    client, headers, test_user_id = authenticated_client

    # Use a very large ID that should not exist in the predictions table.
    nonexistent_prediction_id = 999999999

    # Request the SHAP explanation for the nonexistent prediction.
    response = client.get(
        f"/predictions/{nonexistent_prediction_id}/explanation",
        headers=headers
    )

    # Confirm that the API returns HTTP 404 Not Found.
    assert response.status_code == 404

    # Confirm that the API returns the expected error message.
    assert response.json()["detail"] == "Prediction not found."


# ============================================================
# TEST 17: VERIFY AUTHENTICATED GLOBAL SHAP FEATURE IMPORTANCE
# ============================================================

def test_authenticated_user_can_retrieve_global_shap_explanation(authenticated_client):

    # Receive the authenticated test client and its authorization headers.
    client, headers, _ = authenticated_client

    # Request global SHAP feature importance.
    response = client.get(
        "/predictions/explanations/global",
        headers=headers
    )

    # Confirm that the endpoint returns HTTP 200.
    assert response.status_code == 200

    # Convert the JSON response into a Python dictionary.
    result = response.json()

    # Confirm that all 768 dataset records were analysed.
    assert result["total_records_analysed"] == 768

    # Retrieve the feature importance dictionary.
    importance = result["feature_importance"]

    # Define the eight features used by the trained ML model.
    expected_features = {
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    }

    # Confirm that all eight expected features are included.
    assert set(importance.keys()) == expected_features

    # Confirm that all importance values are numeric.
    assert all(
        isinstance(value, (int, float))
        for value in importance.values()
    )

    # Confirm that global importance values are non-negative.
    assert all(
        value >= 0
        for value in importance.values()
    )

    # Retrieve importance values in their returned order.
    importance_values = list(importance.values())

    # Confirm that features are sorted from highest to lowest importance.
    assert importance_values == sorted(
        importance_values,
        reverse=True
    )


# ============================================================
# TEST 18: VERIFY GLOBAL SHAP ENDPOINT REQUIRES AUTHENTICATION
# ============================================================

def test_global_shap_explanation_requires_authentication():

    # Create a test client without an Authorization header.
    with TestClient(app) as client:

        # Request global SHAP feature importance without authentication.
        response = client.get(
            "/predictions/explanations/global"
        )

        # Confirm that the API rejects the request with HTTP 401.
        assert response.status_code == 401