
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# Schema for registering new machine learning model metadata.
# This schema validates the information supplied by an admin.
class ModelCreate(BaseModel):

    # Store the human-readable name of the trained ML model.
    model_name: str = Field(min_length=2, max_length=100)

    # Store the version identifier of the model.
    # Example: 1.0, 2.0, or improved-v2.
    version: str = Field(min_length=1, max_length=50)

    # Store the algorithm used to train the model.
    # Example: Logistic Regression, Random Forest, or SVM.
    algorithm: str = Field(min_length=2, max_length=100)

    # Store the model's evaluation metrics as a JSON-compatible dictionary.
    # Example: accuracy, precision, recall, F1-score, and ROC-AUC.
    metrics: dict[str, Any] = Field(min_length=1)


# Schema for returning basic machine learning model information.
# ConfigDict allows Pydantic to read values directly from SQLAlchemy objects.
class ModelResponse(BaseModel):

    model_config = ConfigDict(from_attributes=True)

    # Unique database identifier of the registered model.
    id: int

    # Name and version identify the registered model.
    model_name: str
    version: str

    # Algorithm used by the model.
    algorithm: str

    # Indicates whether the database marks this model as active.
    # This does not independently confirm which model is loaded in memory.
    is_active: bool

    # Date and time when the model metadata was registered.
    created_at: datetime


# Schema for returning detailed model information, including metrics.
class ModelDetailResponse(ModelResponse):

    # Include the evaluation metrics in the detailed response.
    metrics: dict[str, Any]

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# Schema for registering new machine learning model metadata.
# This schema validates the information supplied by an admin.
class ModelCreate(BaseModel):

    # Store the human-readable name of the trained ML model.
    model_name: str = Field(min_length=2, max_length=100)

    # Store the version identifier of the model.
    # Example: 1.0, 2.0, or improved-v2.
    version: str = Field(min_length=1, max_length=50)

    # Store the algorithm used to train the model.
    # Example: Logistic Regression, Random Forest, or SVM.
    algorithm: str = Field(min_length=2, max_length=100)

    # Store the model's evaluation metrics as a JSON-compatible dictionary.
    # Example: accuracy, precision, recall, F1-score, and ROC-AUC.
    metrics: dict[str, Any] = Field(min_length=1)


# Schema for returning basic machine learning model information.
# ConfigDict allows Pydantic to read values directly from SQLAlchemy objects.
class ModelResponse(BaseModel):

    model_config = ConfigDict(from_attributes=True)

    # Unique database identifier of the registered model.
    id: int

    # Name and version identify the registered model.
    model_name: str
    version: str

    # Algorithm used by the model.
    algorithm: str

    # Indicates whether the database marks this model as active.
    # This does not independently confirm which model is loaded in memory.
    is_active: bool

    # Date and time when the model metadata was registered.
    created_at: datetime


# Schema for returning detailed model information, including metrics.
class ModelDetailResponse(ModelResponse):

    # Include the evaluation metrics in the detailed response.
    metrics: dict[str, Any]
