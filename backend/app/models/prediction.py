from sqlalchemy import (
    Column,
    Integer,
    Float,
    TIMESTAMP,
    ForeignKey,
    CheckConstraint
)
from sqlalchemy.dialects.mysql import BIGINT, TINYINT
from sqlalchemy.sql import func

from backend.app.db.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(
        BIGINT(unsigned=True),
        primary_key=True,
        autoincrement=True
    )

    user_id = Column(
        BIGINT(unsigned=True),
        ForeignKey("users.id", ondelete="RESTRICT"),
        nullable=False
    )

    model_id = Column(
        BIGINT(unsigned=True),
        ForeignKey("model_metadata.id", ondelete="RESTRICT"),
        nullable=False
    )

    pregnancies = Column(Integer, nullable=False)
    glucose = Column(Float, nullable=False)
    blood_pressure = Column(Float, nullable=False)
    skin_thickness = Column(Float, nullable=False)
    insulin = Column(Float, nullable=False)
    bmi = Column(Float, nullable=False)
    diabetes_pedigree_function = Column(Float, nullable=False)
    age = Column(Integer, nullable=False)

    predicted_class = Column(TINYINT, nullable=False)

    predicted_probability = Column(Float, nullable=False)

    created_at = Column(
        TIMESTAMP,
        nullable=False,
        server_default=func.current_timestamp()
    )

    __table_args__ = (
        CheckConstraint(
            "predicted_class IN (0, 1)",
            name="chk_predicted_class"
        ),
        CheckConstraint(
            "predicted_probability >= 0 AND predicted_probability <= 1",
            name="chk_predicted_probability"
        ),
    )