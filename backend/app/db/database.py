
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker, declarative_base


# ==========================================================
# PROJECT ROOT AND ENVIRONMENT CONFIGURATION
# ==========================================================

# Find the project root directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


# Define the path to the TiDB Cloud SSL certificate.
CA_CERT_PATH = PROJECT_ROOT / "certs" / "isrgrootx1.pem"


# Load environment variables from the root .env file.
load_dotenv(PROJECT_ROOT / ".env", override=True)


# Verify that the SSL certificate exists.
if not CA_CERT_PATH.is_file():
    raise FileNotFoundError(
        f"SSL certificate not found: {CA_CERT_PATH}"
    )


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


# ==========================================================
# CREATE SQLALCHEMY DATABASE ENGINE
# ==========================================================

# Configure a secure TLS connection to the TiDB Cloud database.
# Verify both the server certificate and server identity.
engine = create_engine(
    DATABASE_URL,

    # Verify that the database server certificate is trusted.
    # Use the downloaded ISRG Root X1 CA certificate.
    connect_args={
        "ssl_ca": str(CA_CERT_PATH),
        "ssl_verify_cert": True,
        "ssl_verify_identity": True,
    },

    # Check database connections before reusing them.
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