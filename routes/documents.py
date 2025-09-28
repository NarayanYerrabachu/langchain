import logging
import time
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from langchain_core.indexing import DeleteResponse

from models import (
    IngestInput, IngestResponse, QueryInput, QueryResponse,
    DocumentType, IngestFileInput, DocumentListResponse, DocsInfo, DocumentDeleteResponse
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


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(50, ge=1, le=100, description="Number of documents per page"),
        document_type: Optional[str] = Query(None, description="Filter by document type"),
        search: Optional[str] = Query(None, description="Search in document names/sources")
):
    """List all documents in the collection with pagination and filtering"""
    try:
        from database import get_vectorstore

        vectorstore = get_vectorstore()

        # Calculate offset for pagination
        offset = (page - 1) * page_size

        # For mock implementations, return sample data
        if hasattr(vectorstore, '_documents'):
            # Mock vectorstore case
            documents = vectorstore._documents or []

            # Apply filters
            if document_type:
                documents = [doc for doc in documents if doc.metadata.get('document_type') == document_type]

            if search:
                documents = [doc for doc in documents if
                             search.lower() in doc.metadata.get('source', '').lower() or
                             search.lower() in doc.metadata.get('filename', '').lower()]

            # Group by document_id to get unique documents
            doc_groups = {}
            for doc in documents:
                doc_id = doc.metadata.get('document_id')
                if doc_id not in doc_groups:
                    doc_groups[doc_id] = []
                doc_groups[doc_id].append(doc)

            # Create document info list
            doc_infos = []
            for doc_id, chunks in doc_groups.items():
                first_chunk = chunks[0]
                doc_info = DocsInfo(
                    document_id=doc_id,
                    document_type=first_chunk.metadata.get('document_type', 'unknown'),
                    filename=first_chunk.metadata.get('filename'),
                    source=first_chunk.metadata.get('source'),
                    chunk_count=len(chunks),
                    created_at=first_chunk.metadata.get('created_at'),
                    file_size=first_chunk.metadata.get('file_size')
                )
                doc_infos.append(doc_info)

            # Apply pagination
            total = len(doc_infos)
            paginated_docs = doc_infos[offset:offset + page_size]

            return DocumentListResponse(
                status="success",
                documents=paginated_docs,
                total=total,
                page=page,
                page_size=page_size,
                message="Document listing not fully implemented. Use query endpoint to search documents."
            )

        else:
            # For real PGVector implementation
            # This would require additional SQL queries to get document metadata
            # You might need to store document metadata in a separate table

            return DocumentListResponse(
                status="success",
                documents=[],
                total=0,
                page=page,
                page_size=page_size,
                message="Document listing not fully implemented. Use query endpoint to search documents."
            )

    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.delete("/documents/{document_id}", response_model=DocumentDeleteResponse)
def delete_document(document_id: str):
    """Delete a specific document by ID"""
    try:
        if not document_id.strip():
            raise HTTPException(status_code=400, detail="Document ID cannot be empty")

        vectorstore = get_vectorstore()
        chunks_deleted = 0
        document_found = False

        # Handle mock vector store implementation
        if hasattr(vectorstore, '_documents'):
            original_count = len(vectorstore._documents)

            # Find chunks belonging to this document
            chunks_to_keep = []
            for doc in vectorstore._documents:
                if doc.metadata.get('document_id') != document_id:
                    chunks_to_keep.append(doc)
                else:
                    document_found = True

            # Update the documents list
            vectorstore._documents = chunks_to_keep
            chunks_deleted = original_count - len(chunks_to_keep)

            if document_found:
                logger.info(f"Successfully deleted document {document_id} ({chunks_deleted} chunks)")
                return DocumentDeleteResponse(
                    status="success",
                    message=f"Successfully deleted document '{document_id}' and {chunks_deleted} associated chunks",
                    document_id=document_id,
                    chunks_deleted=chunks_deleted,
                    document_found=True
                )
            else:
                return DocumentDeleteResponse(
                    status="not_found",
                    message=f"Document '{document_id}' not found in the collection",
                    document_id=document_id,
                    chunks_deleted=0,
                    document_found=False
                )

        # Handle real PGVector implementation
        else:
            # For real PGVector, we need to use SQL to delete by metadata
            # This requires accessing the underlying database connection
            try:
                # Attempt to delete using vector store's delete method if available
                if hasattr(vectorstore, 'delete'):
                    # Some vector stores support deletion by filter
                    result = vectorstore.delete(filter={"document_id": document_id})
                    chunks_deleted = result if isinstance(result, int) else 1
                    document_found = chunks_deleted > 0

                elif hasattr(vectorstore, '_connection') or hasattr(vectorstore, 'connection'):
                    # Direct SQL approach for PGVector
                    # This would require the actual database connection
                    connection = getattr(vectorstore, '_connection', None) or getattr(vectorstore, 'connection', None)

                    if connection:
                        # SQL to delete from PGVector table where metadata contains document_id
                        delete_query = """
                                       DELETE \
                                       FROM langchain_pg_embedding
                                       WHERE metadata ->>'document_id' = %s \
                                       """

                        with connection.cursor() as cursor:
                            cursor.execute(delete_query, (document_id,))
                            chunks_deleted = cursor.rowcount
                            connection.commit()

                        document_found = chunks_deleted > 0
                    else:
                        raise Exception("No database connection available")

                else:
                    # Fallback: return not implemented for real vector stores without delete capability
                    return DocumentDeleteResponse(
                        status="not_implemented",
                        message=f"Document deletion not implemented for this vector store type. Document ID: {document_id}",
                        document_id=document_id,
                        chunks_deleted=0,
                        document_found=False
                    )

                if document_found:
                    logger.info(f"Successfully deleted document {document_id} ({chunks_deleted} chunks) from PGVector")
                    return DocumentDeleteResponse(
                        status="success",
                        message=f"Successfully deleted document '{document_id}' and {chunks_deleted} associated chunks",
                        document_id=document_id,
                        chunks_deleted=chunks_deleted,
                        document_found=True
                    )
                else:
                    return DocumentDeleteResponse(
                        status="not_found",
                        message=f"Document '{document_id}' not found in the collection",
                        document_id=document_id,
                        chunks_deleted=0,
                        document_found=False
                    )

            except Exception as db_error:
                logger.error(f"Database error during deletion: {str(db_error)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Database error during deletion: {str(db_error)}"
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


# Helper function to find documents before deletion (useful for confirmation)
@router.get("/documents/{document_id}/info")
def get_document_info(document_id: str):
    """Get information about a specific document before deletion"""
    try:
        vectorstore = get_vectorstore()

        if hasattr(vectorstore, '_documents'):
            # Mock implementation
            matching_chunks = [
                doc for doc in vectorstore._documents
                if doc.metadata.get('document_id') == document_id
            ]

            if not matching_chunks:
                raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")

            # Get document info from first chunk
            first_chunk = matching_chunks[0]

            return {
                "status": "found",
                "document_id": document_id,
                "document_type": first_chunk.metadata.get('document_type'),
                "filename": first_chunk.metadata.get('filename'),
                "source": first_chunk.metadata.get('source'),
                "chunk_count": len(matching_chunks),
                "created_at": first_chunk.metadata.get('created_at'),
                "total_content_length": sum(len(chunk.page_content) for chunk in matching_chunks)
            }
        else:
            # For real implementations, you'd query the database
            return {
                "status": "not_implemented",
                "message": "Document info lookup not implemented for this vector store type",
                "document_id": document_id
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get document info: {str(e)}")


# Bulk deletion endpoint
@router.delete("/documents/bulk")
def delete_multiple_documents(document_ids: List[str]):
    """Delete multiple documents by their IDs"""
    try:
        if not document_ids:
            raise HTTPException(status_code=400, detail="No document IDs provided")

        results = []
        total_deleted = 0

        for doc_id in document_ids:
            try:
                # Reuse the single document deletion logic
                result = delete_document(doc_id)
                results.append({
                    "document_id": doc_id,
                    "status": result.status,
                    "chunks_deleted": result.chunks_deleted
                })
                total_deleted += result.chunks_deleted

            except HTTPException as e:
                results.append({
                    "document_id": doc_id,
                    "status": "error",
                    "error": str(e.detail),
                    "chunks_deleted": 0
                })

        return {
            "status": "completed",
            "total_documents_requested": len(document_ids),
            "total_chunks_deleted": total_deleted,
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed bulk deletion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed bulk deletion: {str(e)}")