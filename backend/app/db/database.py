
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker, declarative_base


# Find the project root and load the .env file
PROJECT_ROOT = Path(__file__).resolve().parents[3]

load_dotenv(PROJECT_ROOT / ".env")


# Read MySQL database configuration
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")


# Validate required database settings
required_settings = {
    "DB_HOST": DB_HOST,
    "DB_PORT": DB_PORT,
    "DB_USER": DB_USER,
    "DB_PASSWORD": DB_PASSWORD,
    "DB_NAME": DB_NAME,
}

missing_settings = [
    key for key, value in required_settings.items()
    if not value
]

if missing_settings:
    raise RuntimeError(
        f"Missing database settings in .env: {', '.join(missing_settings)}"
    )


# Build the MySQL connection URL safely
DATABASE_URL = URL.create(
    drivername="mysql+pymysql",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=int(DB_PORT),
    database=DB_NAME,
)


# Create SQLAlchemy database engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)


# Create database session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


# Base class for SQLAlchemy database models
Base = declarative_base()

# Provide a database session to API endpoints
def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()