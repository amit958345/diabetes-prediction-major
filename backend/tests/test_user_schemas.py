
from types import SimpleNamespace

from pydantic import ValidationError

from backend.app.schemas.user import (
    UserCreate,
    UserLogin,
    UserResponse
)


# TEST 1: User Registration Schema
registration_data = {
    "name": "Amit",
    "email": "amit@example.com",
    "password": "StrongPassword123"
}

try:
    new_user = UserCreate.model_validate(registration_data)

    print("TEST 1: UserCreate PASSED")
    print("Name:", new_user.name)
    print("Email:", new_user.email)

except ValidationError as error:
    print("TEST 1 FAILED")
    print(error)


# TEST 2: User Login Schema
login_data = {
    "email": "amit@example.com",
    "password": "StrongPassword123"
}

try:
    login_user = UserLogin.model_validate(login_data)

    print("\nTEST 2: UserLogin PASSED")
    print("Email:", login_user.email)

except ValidationError as error:
    print("TEST 2 FAILED")
    print(error)


# TEST 3: User Response Schema
database_user = SimpleNamespace(
    id=1,
    name="Amit",
    email="amit@example.com",
    role="user",
    password_hash="this_is_a_test_hash"
)

try:
    response_user = UserResponse.model_validate(database_user)

    print("\nTEST 3: UserResponse PASSED")
    print("User information:", response_user.model_dump())

except ValidationError as error:
    print("TEST 3 FAILED")
    print(error)
