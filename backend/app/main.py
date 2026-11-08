
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
# The frontend and backend use different origins:
# Frontend: http://localhost:5173
# Backend:  http://127.0.0.1:8000
#
# Browsers restrict cross-origin requests unless the backend
# explicitly allows the frontend origin.
#
# CORSMiddleware handles browser preflight OPTIONS requests
# and adds the appropriate CORS response headers.

app.add_middleware(
    CORSMiddleware,

    # Allow requests from the React development server.
    # Both addresses are included because localhost and
    # 127.0.0.1 are treated as different browser origins.
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],

    # Allow cookies and authorization credentials when needed.
    allow_credentials=True,

    # Allow HTTP methods such as GET, POST, PUT and DELETE.
    # This also permits OPTIONS preflight requests.
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
