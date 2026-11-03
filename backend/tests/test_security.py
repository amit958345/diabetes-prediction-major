
from backend.app.core.security import (
    hash_password,
    verify_password
)


# Sample password used only for testing
test_password = "StrongPassword123"


# TEST 1: Generate a password hash
hashed_password = hash_password(test_password)

print("TEST 1: Password hash generated")


# TEST 2: Check whether the hash uses Argon2id
if hashed_password.startswith("$argon2id$"):
    print("TEST 2: Argon2id hash confirmed")
else:
    print("TEST 2: Argon2id hash NOT confirmed")


# TEST 3: Verify the correct password
if verify_password(test_password, hashed_password):
    print("TEST 3: Correct password accepted")
else:
    print("TEST 3: Correct password rejected")


# TEST 4: Verify an incorrect password
wrong_password = "WrongPassword123"

if verify_password(wrong_password, hashed_password):
    print("TEST 4: Incorrect password accepted")
else:
    print("TEST 4: Incorrect password rejected")
