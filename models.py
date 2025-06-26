from typing import List, Dict, Any
from pydantic import BaseModel

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
