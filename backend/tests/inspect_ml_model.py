from pathlib import Path
import sys
import joblib


# Find the main project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Add the src directory to Python's module search path
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))


# Define the saved model file paths
MODEL_PATH = PROJECT_ROOT / "models" / "diabetes_model.pkl"
FEATURE_PATH = PROJECT_ROOT / "models" / "feature_names.pkl"


# Load the trained ML pipeline
model = joblib.load(MODEL_PATH)


# Load the expected feature names
feature_names = joblib.load(FEATURE_PATH)


# Display model information
print("MODEL LOADED SUCCESSFULLY")
print("Model type:", type(model))

print("\nPIPELINE STEPS:")
print(model.named_steps)


# Display feature information
print("\nEXPECTED FEATURES:")
print(feature_names)

print("\nNUMBER OF FEATURES:", len(feature_names))