import os

# Database configuration
CONNECTION_STRING = "postgresql+psycopg2://phone_dev_user:5J1058^&*Y7g@localhost:15432/nemo_dev_vdb"
COLLECTION_NAME = "langchain_documents"

# API configuration
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable must be set")
