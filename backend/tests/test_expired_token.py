
from datetime import timedelta

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.security import create_access_token


# Create a test client for our FastAPI application
client = TestClient(app)


# Generate a JWT that has already expired
expired_token = create_access_token(
    data={"sub": "4"},
    expires_delta=timedelta(seconds=-1)
)


# Send a request to the protected endpoint
response = client.get(
    "/auth/me",
    headers={
        "Authorization": f"Bearer {expired_token}"
    }
)


# Display the API response
print("HTTP Status:", response.status_code)
print("Response:", response.json())


# Verify that the expired token was rejected
assert response.status_code == 401

print("EXPIRED TOKEN TEST PASSED")
