
"""
============================================================
DIAPREDICT AI - MAIN FASTAPI APPLICATION
Major Project: Explainable AI-Based Diabetes Risk Prediction

Responsibilities:
1. Create the FastAPI application.
2. Configure CORS for React frontend communication.
3. Register authentication routes.
4. Register administration routes.
5. Register prediction routes.
6. Provide the root API endpoint.

Frontend:
http://localhost:5173

Backend:
http://127.0.0.1:8000
============================================================
"""

import os
# Import FastAPI to create the backend application.
from fastapi import FastAPI

# Import CORSMiddleware to allow communication between
# the React frontend and FastAPI backend.
from fastapi.middleware.cors import CORSMiddleware

# Import the authentication API router.
from backend.app.api.auth import router as auth_router

# Import the administration API router.
from backend.app.api.admin import router as admin_router

# Import the prediction API router.
from backend.app.api.predictions import router as predictions_router
from backend.app.api.insights import router as insights_router


# ==========================================================
# CREATE FASTAPI APPLICATION
# ==========================================================

# Create the main FastAPI application instance.
app = FastAPI(
    title="Diabetes Prediction Major Project",
    description=(
        "Explainable AI-Based Diabetes Risk Prediction "
        "and Decision Support System"
    ),
    version="1.0.0"
)

# ==========================================================
# CONFIGURE CORS MIDDLEWARE
# ==========================================================

# CORS stands for Cross-Origin Resource Sharing.
#
# It allows the React frontend to communicate with the
# FastAPI backend when they run on different origins.
#
# Local development:
# http://localhost:5173
# http://127.0.0.1:5173
#
# Production:
# The deployed frontend URL will be configured later
# through the FRONTEND_ORIGINS environment variable.


# Read allowed frontend origins from environment variables.
# Multiple frontend URLs can be separated using commas.
frontend_origins = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173"
)


# Convert the comma-separated URLs into a Python list.
# Remove unnecessary spaces and trailing slashes.
allowed_origins = [
    origin.strip().rstrip("/")
    for origin in frontend_origins.split(",")
    if origin.strip()
]


# Add CORS middleware to the FastAPI application.
app.add_middleware(
    CORSMiddleware,

    # Allow requests from configured frontend origins.
    allow_origins=allowed_origins,

    # Allow authorization credentials when required.
    allow_credentials=True,

    # Allow HTTP methods such as GET, POST, PUT and DELETE.
    allow_methods=["*"],

    # Allow headers such as Content-Type and Authorization.
    allow_headers=["*"],
)


# ==========================================================
# REGISTER API ROUTERS
# ==========================================================

# Register user registration, login and authentication routes.
app.include_router(auth_router)

# Register administrator dashboard and management routes.
app.include_router(admin_router)

# Register diabetes prediction and prediction history routes.
app.include_router(predictions_router)

# Register authenticated dataset and model research insights.
app.include_router(insights_router)


# ==========================================================
# ROOT ENDPOINT
# ==========================================================

# This endpoint checks whether the backend is running.
@app.get("/")
def home():

    # Return basic project and API information.
    return {
        "message": "Diabetes Prediction API is running successfully!",
        "project": "Explainable AI-Based Diabetes Risk Prediction",
        "version": "1.0.0"
    }
