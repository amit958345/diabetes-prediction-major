
from datetime import timedelta

import jwt

from backend.app.core.security import (
    create_access_token,
    decode_access_token
)


# Sample user information for JWT testing
test_data = {
    "sub": "123"
}


# TEST 1: Generate a JWT access token
access_token = create_access_token(test_data)

if access_token:
    print("TEST 1: JWT generated successfully")
else:
    print("TEST 1 FAILED: JWT generation failed")


# TEST 2: Decode a valid JWT access token
try:
    decoded_data = decode_access_token(access_token)

    if decoded_data.get("sub") == "123":
        print("TEST 2: Valid JWT decoded successfully")
        print("User ID:", decoded_data["sub"])
    else:
        print("TEST 2 FAILED: User ID does not match")

except jwt.InvalidTokenError:
    print("TEST 2 FAILED: Valid JWT was rejected")


# TEST 3: Check whether an expired token is rejected
expired_token = create_access_token(
    test_data,
    expires_delta=timedelta(seconds=-1)
)

try:
    decode_access_token(expired_token)
    print("TEST 3 FAILED: Expired token was accepted")

except jwt.ExpiredSignatureError:
    print("TEST 3: Expired token rejected successfully")


# TEST 4: Check whether an invalid token is rejected
invalid_token = "this.is.not.a.valid.jwt"

try:
    decode_access_token(invalid_token)
    print("TEST 4 FAILED: Invalid token was accepted")

except jwt.InvalidTokenError:
    print("TEST 4: Invalid token rejected successfully")
