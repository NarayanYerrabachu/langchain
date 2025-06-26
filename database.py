import logging
from sqlalchemy import create_engine, text
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
import os

from config import CONNECTION_STRING, COLLECTION_NAME, ENABLE_DATABASE

# Set up logging
logger = logging.getLogger(__name__)

# Initialize database components with error handling
engine = None
embeddings = None

if ENABLE_DATABASE:
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
        # We'll continue with engine=None, which will trigger mock implementations
else:
    logger.info("Database connection disabled by configuration")

# Mock implementations for when database is not available
class MockVectorStore:
    def add_documents(self, docs):
        logger.info(f"Mock: Added {len(docs)} documents")
        return len(docs)
    
    def as_retriever(self, search_kwargs=None):
        return MockRetriever()
    
    def delete_collection(self):
        logger.info("Mock: Collection cleared")
        return True

class MockRetriever:
    def get_relevant_documents(self, query):
        from langchain_core.documents import Document
        return [Document(page_content="This is mock content since the database is not available.", metadata={})]

# Initialize vector store with error handling
def get_vectorstore():
    if not ENABLE_DATABASE or engine is None or embeddings is None:
        logger.warning("Using mock vector store as database is not available")
        return MockVectorStore()
    
    try:
        return PGVector(
            connection=CONNECTION_STRING,
            collection_name=COLLECTION_NAME,
            embeddings=embeddings,
        )
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        return MockVectorStore()

# Get retriever with configurable k value
def get_retriever(k=5):
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})
