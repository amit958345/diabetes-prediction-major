from sqlalchemy import (
    Column,
    String,
    Double,
    JSON,
    TIMESTAMP,
    ForeignKey,
    UniqueConstraint
)
from sqlalchemy.dialects.mysql import BIGINT
from sqlalchemy.sql import func

from backend.app.db.database import Base


class PredictionExplanation(Base):
    __tablename__ = "prediction_explanations"

    id = Column(
        BIGINT(unsigned=True),
        primary_key=True,
        autoincrement=True
    )

    prediction_id = Column(
        BIGINT(unsigned=True),
        ForeignKey("predictions.id", ondelete="CASCADE"),
        nullable=False
    )

    explanation_method = Column(String(50), nullable=False)

    base_value = Column(Double, nullable=False)

    model_output = Column(Double, nullable=False)

    feature_contributions = Column(JSON, nullable=False)

    created_at = Column(
        TIMESTAMP,
        nullable=False,
        server_default=func.current_timestamp()
    )

    __table_args__ = (
        UniqueConstraint(
            "prediction_id",
            name="uq_prediction_explanation"
        ),
    )