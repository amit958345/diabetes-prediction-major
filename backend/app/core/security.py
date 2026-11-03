
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

import jwt
from dotenv import load_dotenv
from pwdlib import PasswordHash


# Find the project root and load the .env file
PROJECT_ROOT = Path(__file__).resolve().parents[3]

load_dotenv(PROJECT_ROOT / ".env")


# Read JWT configuration from environment variables
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
)


# Validate the JWT secret key
if not JWT_SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY is missing from .env")


# Create a reusable password hasher
password_hash = PasswordHash.recommended()


# Convert a plain-text password into a secure hash
def hash_password(password: str) -> str:
    return password_hash.hash(password)


# Verify a password against its stored hash
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return password_hash.verify(plain_password, hashed_password)


# Generate a JWT access token
def create_access_token(
    data: dict,
    expires_delta: timedelta | None = None
) -> str:

    # Create a copy of the original token data
    to_encode = data.copy()

    # Calculate the token expiration time
    if expires_delta is not None:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )

    # Add the expiration time to the token data
    to_encode.update({"exp": expire})

    # Generate and sign the JWT
    encoded_jwt = jwt.encode(
        to_encode,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM
    )

    return encoded_jwt


# Verify a JWT access token and return its claims
def decode_access_token(token: str) -> dict:

    payload = jwt.decode(
        token,
        JWT_SECRET_KEY,
        algorithms=[JWT_ALGORITHM]
    )

    return payload
