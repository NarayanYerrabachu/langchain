import logging
from sqlalchemy import create_engine, text
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings

from config import CONNECTION_STRING, COLLECTION_NAME

# Set up logging
logger = logging.getLogger(__name__)

# Initialize database components with error handling
try:
    # Use psycopg2-binary explicitly in the connection string
    if "psycopg2://" in CONNECTION_STRING:
        CONNECTION_STRING = CONNECTION_STRING.replace("psycopg2://", "postgresql://")
    
    engine = create_engine(CONNECTION_STRING)
    
    # Test connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
        logger.info("Database connection successful")
    
    embeddings = OpenAIEmbeddings()
    
except Exception as e:
    logger.error(f"Failed to initialize database: {str(e)}")
    # Create a dummy engine that will raise appropriate errors when used
    engine = None
    embeddings = None

# Initialize vector store with error handling
def get_vectorstore():
    if engine is None or embeddings is None:
        raise RuntimeError("Database connection not available. Check your PostgreSQL connection and credentials.")
    
    try:
        return PGVector(
            connection=CONNECTION_STRING,
            collection_name=COLLECTION_NAME,
            embeddings=embeddings,
        )
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        raise RuntimeError(f"Vector store initialization failed: {str(e)}")

# Get retriever with configurable k value
def get_retriever(k=5):
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})
