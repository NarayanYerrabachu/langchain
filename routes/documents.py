from fastapi import APIRouter, HTTPException
from langchain.text_splitter import CharacterTextSplitter

from models import IngestInput, IngestResponse, QueryInput, QueryResponse
from database import get_vectorstore
from qa_chain import create_qa_chain

router = APIRouter()

@router.post("/ingest", response_model=IngestResponse)
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
        vectorstore = get_vectorstore()
        vectorstore.add_documents(docs)

        return IngestResponse(
            status="success",
            document_count=len(docs)
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")


@router.post("/query", response_model=QueryResponse)
def query_documents(data: QueryInput):
    """Query the document database using RAG"""
    try:
        if not data.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Create QA chain with specified max_results
        qa_chain = create_qa_chain(k=data.max_results)

        # Run query against RAG pipeline
        result = qa_chain.invoke({"query": data.query})

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
