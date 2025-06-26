from sqlalchemy import create_engine
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings

from config import CONNECTION_STRING, COLLECTION_NAME

# Initialize database components
engine = create_engine(CONNECTION_STRING)
embeddings = OpenAIEmbeddings()

# Initialize vector store
def get_vectorstore():
    return PGVector(
        connection=CONNECTION_STRING,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
    )

# Get retriever with configurable k value
def get_retriever(k=5):
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})
