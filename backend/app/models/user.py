from sqlalchemy import Column, String, Enum, TIMESTAMP
from sqlalchemy.dialects.mysql import BIGINT
from sqlalchemy.sql import func

from backend.app.db.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(
        BIGINT(unsigned=True),
        primary_key=True,
        autoincrement=True
    )

    name = Column(String(100), nullable=False)

    email = Column(String(255), unique=True, nullable=False)

    password_hash = Column(String(255), nullable=False)

    role = Column(
        Enum("user", "admin"),
        nullable=False,
        default="user"
    )

    created_at = Column(
        TIMESTAMP,
        nullable=False,
        server_default=func.current_timestamp()
    )