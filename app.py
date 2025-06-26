import os
from typing import List, Dict, Any
import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI, OpenAIEmbeddings
from sqlalchemy import create_engine
from langchain_postgres.vectorstores import PGVector

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable must be set")

app = FastAPI(title="RAG Document Q&A API", version="1.0.0")

# Database configuration
CONNECTION_STRING = "postgresql+psycopg2://phone_dev_user:5J1058^&*Y7g@localhost:15432/nemo_dev_vdb"
COLLECTION_NAME = "langchain_documents"

# Initialize components
engine = create_engine(CONNECTION_STRING)
embeddings = OpenAIEmbeddings()

vectorstore = PGVector(
    connection=CONNECTION_STRING,
    collection_name=COLLECTION_NAME,
    embeddings=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Return top 5 relevant docs

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0, openai_api_key=""),
    chain_type="stuff",  # Explicitly specify chain type
    retriever=retriever,
    return_source_documents=True,
)


# Pydantic request models
class IngestInput(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}  # Optional metadata for documents


class QueryInput(BaseModel):
    query: str
    max_results: int = 5


class IngestResponse(BaseModel):
    status: str
    document_count: int


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    source_count: int


@app.get("/")
def root():
    return {"message": "RAG Document Q&A API", "status": "running"}


@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database connection failed: {str(e)}")


@app.post("/ingest", response_model=IngestResponse)
def ingest_document(data: IngestInput):
    """Ingest a document into the vector database"""
    try:
        if not data.content.strip():
            raise HTTPException(status_code=400, detail="Content cannot be empty")

        # Split input content into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separator="\n\n"  # Split on paragraphs first
        )

        # Create documents with metadata
        docs = text_splitter.create_documents(
            [data.content],
            metadatas=[data.metadata] if data.metadata else [{}]
        )

        # Add documents to PGVector
        vectorstore.add_documents(docs)

        return IngestResponse(
            status="success",
            document_count=len(docs)
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")


@app.post("/query", response_model=QueryResponse)
def query_documents(data: QueryInput):
    """Query the document database using RAG"""
    try:
        if not data.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Update retriever with max_results if provided
        if data.max_results != 5:
            retriever.search_kwargs = {"k": data.max_results}

        # Run query against RAG pipeline
        result = qa_chain.invoke({"query": data.query})  # Use invoke instead of run

        # Extract answer and sources
        answer = result.get("result", "No answer found")
        source_docs = result.get("source_documents", [])
        sources = [doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                   for doc in source_docs]

        return QueryResponse(
            answer=answer,
            sources=sources,
            source_count=len(sources)
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")


@app.get("/collections/info")
def get_collection_info():
    """Get information about the current collection"""
    try:
        # This is a basic implementation - you might want to add more detailed stats
        return {
            "collection_name": COLLECTION_NAME,
            "status": "active"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get collection info: {str(e)}")


@app.delete("/collections/clear")
def clear_collection():
    """Clear all documents from the collection (use with caution!)"""
    try:
        # Note: This is a destructive operation
        # You might want to implement proper authentication/authorization
        vectorstore.delete_collection()
        return {"status": "Collection cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear collection: {str(e)}")
