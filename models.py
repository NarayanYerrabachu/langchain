from typing import List, Dict, Any, Optional
from pydantic import BaseModel, HttpUrl
from enum import Enum

class DocumentType(str, Enum):
    TEXT = "text"
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    WEB_URL = "web_url"
    MARKDOWN = "markdown"
    CSV = "csv"
    EXCEL = "excel"

class IngestInput(BaseModel):
    content: Optional[str] = None  # For direct text input
    url: Optional[HttpUrl] = None  # For web URLs
    document_type: DocumentType = DocumentType.TEXT
    metadata: Dict[str, Any] = {}  # Optional metadata for documents
    chunk_size: int = 1000
    chunk_overlap: int = 200

class IngestFileInput(BaseModel):
    document_type: DocumentType
    metadata: Dict[str, Any] = {}
    chunk_size: int = 1000
    chunk_overlap: int = 200

class QueryInput(BaseModel):
    query: str
    max_results: int = 5
    include_metadata: bool = False
    similarity_threshold: Optional[float] = None

class IngestResponse(BaseModel):
    status: str
    document_count: int
    document_id: Optional[str] = None
    message: str = ""

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]  # Changed to include metadata
    source_count: int
    query_time: Optional[float] = None

class DocumentInfo(BaseModel):
    id: str
    content_preview: str
    metadata: Dict[str, Any]
    created_at: str
    document_type: DocumentType