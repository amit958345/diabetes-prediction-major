from sqlalchemy import (
    Column,
    String,
    JSON,
    Boolean,
    TIMESTAMP,
    UniqueConstraint
)
from sqlalchemy.dialects.mysql import BIGINT
from sqlalchemy.sql import func

from backend.app.db.database import Base


class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    id = Column(
        BIGINT(unsigned=True),
        primary_key=True,
        autoincrement=True
    )

    model_name = Column(String(100), nullable=False)

    version = Column(String(50), nullable=False)

    algorithm = Column(String(100), nullable=False)

    metrics = Column(JSON, nullable=False)

    is_active = Column(
        Boolean,
        nullable=False,
        default=False,
        server_default="0"
    )

    created_at = Column(
        TIMESTAMP,
        nullable=False,
        server_default=func.current_timestamp()
    )

    __table_args__ = (
        UniqueConstraint(
            "model_name",
            "version",
            name="uq_model_name_version"
        ),
    )