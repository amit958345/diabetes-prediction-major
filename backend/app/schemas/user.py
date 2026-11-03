
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, ConfigDict


# Schema for user registration
class UserCreate(BaseModel):

    name: str = Field(min_length=2, max_length=100)

    email: EmailStr

    password: str = Field(min_length=8, max_length=128)


# Schema for user login
class UserLogin(BaseModel):

    email: EmailStr

    password: str = Field(min_length=8, max_length=128)


# Schema for safe user information returned to the frontend
class UserResponse(BaseModel):

    model_config = ConfigDict(from_attributes=True)

    id: int

    name: str

    email: EmailStr

    role: Literal["user", "admin"]


# Schema for the successful login response
class TokenResponse(BaseModel):

    access_token: str

    token_type: str

    user: UserResponse
