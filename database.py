import logging
from typing import Optional, List
from sqlalchemy import create_engine, text
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import os

from config import (
    CONNECTION_STRING, COLLECTION_NAME, ENABLE_DATABASE,
    OPENAI_API_KEY, EMBEDDING_MODEL
)

# Set up logging
logger = logging.getLogger(__name__)

# Initialize database components with error handling
engine = None
embeddings = None


def init_database():
    """Initialize database connection and embeddings"""
    global engine, embeddings

    if not ENABLE_DATABASE:
        logger.info("Database connection disabled by configuration")
        return

    try:
        # Fix the connection string to use postgresql instead of postgresql+psycopg2
        fixed_connection_string = CONNECTION_STRING
        if "postgresql+psycopg2" in fixed_connection_string:
            fixed_connection_string = fixed_connection_string.replace("postgresql+psycopg2", "postgresql")

        logger.info(f"Attempting to connect to database...")
        engine = create_engine(
            fixed_connection_string,
            pool_pre_ping=True,
            pool_recycle=300,
            echo=False
        )

        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info("Database connection successful")

        # Initialize embeddings
        if OPENAI_API_KEY and OPENAI_API_KEY != "dummy-key-for-development":
            embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
            logger.info(f"Initialized OpenAI embeddings with model: {EMBEDDING_MODEL}")
        else:
            logger.warning("OpenAI API key not available, using mock embeddings")
            embeddings = MockEmbeddings()

    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        logger.info("Continuing with mock implementations")
        engine = None
        embeddings = MockEmbeddings()


class MockEmbeddings:
    """Mock embeddings for development/testing"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return mock embeddings for documents"""
        return [[0.1] * 1536 for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        """Return mock embedding for query"""
        return [0.1] * 1536


class MockVectorStore:
    """Mock vector store for when database is not available"""

    def __init__(self):
        self._documents = []

    def add_documents(self, docs: List[Document]):
        """Mock add documents"""
        self._documents.extend(docs)
        logger.info(f"Mock: Added {len(docs)} documents (total: {len(self._documents)})")
        return [f"doc_{i}" for i in range(len(docs))]

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Mock similarity search"""
        logger.info(f"Mock: Searching for '{query}' with k={k}")
        # Return up to k documents from stored documents
        return self._documents[:k] if self._documents else [
            Document(
                page_content=f"This is mock content for query: '{query}'. Database is not available.",
                metadata={"source": "mock", "query": query}
            )
        ]

    def as_retriever(self, search_kwargs=None):
        """Return mock retriever"""
        return MockRetriever(self, search_kwargs or {})

    def delete_collection(self):
        """Mock delete collection"""
        self._documents.clear()
        logger.info("Mock: Collection cleared")
        return True


class MockRetriever:
    """Mock retriever for development/testing"""

    def __init__(self, vectorstore: MockVectorStore, search_kwargs: dict):
        self.vectorstore = vectorstore
        self.search_kwargs = search_kwargs

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents using mock similarity search"""
        k = self.search_kwargs.get("k", 5)
        return self.vectorstore.similarity_search(query, k=k)


def get_vectorstore() -> PGVector:
    """Get vector store instance with error handling"""
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


def get_retriever(k: int = 5):
    """Get retriever with configurable k value"""
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})


def health_check_database() -> dict:
    """Check database health"""
    if not ENABLE_DATABASE:
        return {
            "status": "disabled",
            "message": "Database is disabled by configuration"
        }

    if engine is None:
        return {
            "status": "unavailable",
            "message": "Database connection not initialized"
        }

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "message": "Database connection is working"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Database connection failed: {str(e)}"
        }


# Initialize on import
init_database()