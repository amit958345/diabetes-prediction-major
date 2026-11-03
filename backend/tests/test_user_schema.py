
from pydantic import ValidationError

from backend.app.schemas.user import UserCreate


# Test 1: Valid registration data
try:
    user = UserCreate.model_validate({
        "name": "Amit",
        "email": "amit@example.com",
        "password": "StrongPassword123"
    })

    print("VALID DATA ACCEPTED")
    print(user)

except ValidationError as error:
    print("VALIDATION FAILED")
    print(error)


# Test 2: Invalid email address
try:
    user = UserCreate.model_validate({
        "name": "Amit",
        "email": "amit123",
        "password": "StrongPassword123"
    })

    print("INVALID EMAIL ACCEPTED")

except ValidationError as error:
    print("\nINVALID EMAIL REJECTED")
    print(error)


# Test 3: Short password
try:
    user = UserCreate.model_validate({
        "name": "Amit",
        "email": "amit@example.com",
        "password": "123"
    })

    print("SHORT PASSWORD ACCEPTED")

except ValidationError as error:
    print("\nSHORT PASSWORD REJECTED")
    print(error)
