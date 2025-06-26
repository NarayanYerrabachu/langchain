import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
CONNECTION_STRING = "postgresql://phone_dev_user:5J1058^&*Y7g@localhost:15432/nemo_dev_vdb"
COLLECTION_NAME = "langchain_documents"

# Feature flags
ENABLE_DATABASE = True  # Set to False to run without database

# API configuration
if not os.getenv("OPENAI_API_KEY"):
    logger.warning("OPENAI_API_KEY environment variable not set. Some features will be limited.")
    os.environ["OPENAI_API_KEY"] = "dummy-key-for-development"
