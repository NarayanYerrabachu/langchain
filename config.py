import os
import logging
from typing import Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration
CONNECTION_STRING = os.getenv(
    "DATABASE_URL",
    "postgresql://phone_dev_user:5J1058^&*Y7g@localhost:15432/nemo_dev_vdb"
)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "langchain_documents")

# Feature flags
ENABLE_DATABASE = os.getenv("ENABLE_DATABASE", "true").lower() == "true"

# API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY environment variable not set. Some features will be limited.")
    os.environ["OPENAI_API_KEY"] = "dummy-key-for-development"

# File upload configuration
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "50")) * 1024 * 1024  # 50MB default
ALLOWED_EXTENSIONS = {
    'pdf', 'docx', 'doc', 'txt', 'md', 'html', 'htm',
    'csv', 'xlsx', 'xls', 'pptx', 'ppt'
}

# Text processing configuration
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "200"))
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "4000"))

# Web scraping configuration
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "10"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# Vector store configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
VECTOR_DIMENSIONS = int(os.getenv("VECTOR_DIMENSIONS", "1536"))

# Security configuration
ALLOWED_DOMAINS = os.getenv("ALLOWED_DOMAINS", "").split(",") if os.getenv("ALLOWED_DOMAINS") else []
BLOCKED_DOMAINS = os.getenv("BLOCKED_DOMAINS", "").split(",") if os.getenv("BLOCKED_DOMAINS") else []

# Rate limiting (if needed)
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))

# Logging level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.getLogger().setLevel(getattr(logging, LOG_LEVEL))

logger.info("Configuration loaded successfully")
logger.info(f"Database enabled: {ENABLE_DATABASE}")
logger.info(f"Collection name: {COLLECTION_NAME}")
logger.info(f"Max file size: {MAX_FILE_SIZE / 1024 / 1024}MB")