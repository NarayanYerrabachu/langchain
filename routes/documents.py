import logging
import time
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from models import (
    IngestInput, IngestResponse, QueryInput, QueryResponse,
    DocumentType, IngestFileInput
)
from database import get_vectorstore
from qa_chain import create_qa_chain
from document_loaders import DocumentProcessor

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize document processor
doc_processor = DocumentProcessor()


@router.post("/ingest", response_model=IngestResponse)
def ingest_document(data: IngestInput):
    """Ingest a document into the vector database"""
    try:
        # Validate input
        if data.document_type == DocumentType.WEB_URL and not data.url:
            raise HTTPException(status_code=400, detail="URL is required for web_url document type")

        if data.document_type == DocumentType.TEXT and not data.content:
            raise HTTPException(status_code=400, detail="Content is required for text document type")

        # Process document using the document processor
        chunks = doc_processor.process_document(
            content=data.content,
            url=str(data.url) if data.url else None,
            document_type=data.document_type,
            metadata=data.metadata,
            chunk_size=data.chunk_size,
            chunk_overlap=data.chunk_overlap
        )

        if not chunks:
            raise HTTPException(status_code=400, detail="No content extracted from document")

        # Add documents to vector store
        vectorstore = get_vectorstore()
        vectorstore.add_documents(chunks)

        logger.info(f"Successfully ingested {len(chunks)} chunks from {data.document_type.value} document")

        return IngestResponse(
            status="success",
            document_count=len(chunks),
            document_id=chunks[0].metadata.get("document_id") if chunks else None,
            message=f"Successfully processed {data.document_type.value} document into {len(chunks)} chunks"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to ingest document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
        file: UploadFile = File(...),
        document_type: DocumentType = Form(...),
        metadata: str = Form("{}"),
        chunk_size: int = Form(1000),
        chunk_overlap: int = Form(200)
):
    """Ingest a file upload into the vector database"""
    try:
        # Parse metadata if provided as JSON string
        import json
        try:
            parsed_metadata = json.loads(metadata) if metadata != "{}" else {}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in metadata field")

        # Add file information to metadata
        parsed_metadata.update({
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": file.size if hasattr(file, 'size') else None
        })

        # Read file content
        file_content = await file.read()

        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Process document using the document processor
        chunks = doc_processor.process_document(
            file_content=file_content,
            document_type=document_type,
            metadata=parsed_metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        if not chunks:
            raise HTTPException(status_code=400, detail="No content extracted from file")

        # Add documents to vector store
        vectorstore = get_vectorstore()
        vectorstore.add_documents(chunks)

        logger.info(f"Successfully ingested file {file.filename} into {len(chunks)} chunks")

        return IngestResponse(
            status="success",
            document_count=len(chunks),
            document_id=chunks[0].metadata.get("document_id") if chunks else None,
            message=f"Successfully processed file '{file.filename}' into {len(chunks)} chunks"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to ingest file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to ingest file: {str(e)}")


@router.post("/query", response_model=QueryResponse)
def query_documents(data: QueryInput):
    """Query the document database using RAG"""
    try:
        if not data.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        start_time = time.time()

        # Create QA chain with specified max_results
        qa_chain = create_qa_chain(k=data.max_results)

        # Run query against RAG pipeline
        result = qa_chain.invoke({"query": data.query})

        # Extract answer and sources
        answer = result.get("result", "No answer found")
        source_docs = result.get("source_documents", [])

        # Format sources with metadata if requested
        sources = []
        for doc in source_docs:
            source_info = {
                "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
            }

            if data.include_metadata:
                source_info["metadata"] = doc.metadata
            else:
                # Include only essential metadata
                essential_metadata = {}
                for key in ["document_id", "filename", "source_url", "title", "page"]:
                    if key in doc.metadata:
                        essential_metadata[key] = doc.metadata[key]
                if essential_metadata:
                    source_info["metadata"] = essential_metadata

            sources.append(source_info)

        query_time = time.time() - start_time

        return QueryResponse(
            answer=answer,
            sources=sources,
            source_count=len(sources),
            query_time=round(query_time, 3)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")


@router.get("/documents")
def list_documents():
    """List all documents in the collection (basic implementation)"""
    try:
        # This is a basic implementation - in a real system you might want to
        # store document metadata separately for easier querying
        return {
            "status": "success",
            "message": "Document listing not fully implemented. Use query endpoint to search documents."
        }
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.delete("/documents/{document_id}")
def delete_document(document_id: str):
    """Delete a specific document by ID"""
    try:
        # This would require implementing document deletion in the vector store
        # PGVector doesn't have built-in document deletion by metadata
        return {
            "status": "success",
            "message": f"Document deletion for ID {document_id} not fully implemented"
        }
    except Exception as e:
        logger.error(f"Failed to delete document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")