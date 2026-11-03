from fastapi import FastAPI

from backend.app.api.auth import router as auth_router
from backend.app.api.admin import router as admin_router
from backend.app.api.predictions import router as predictions_router


# Create the FastAPI application
app = FastAPI(
    title="Diabetes Prediction Major Project",
    description="Explainable AI-Based Diabetes Risk Prediction and Decision Support System",
    version="1.0.0"
)


# Register the authentication router
app.include_router(auth_router)


# Register the administration router
app.include_router(admin_router)


# Register the prediction router
app.include_router(predictions_router)


# Root endpoint
@app.get("/")
def home():
    return {
        "message": "Diabetes Prediction API is running successfully!",
        "project": "Explainable AI-Based Diabetes Risk Prediction",
        "version": "1.0.0"
    }