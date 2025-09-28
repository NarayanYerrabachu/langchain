import unittest
from datetime import datetime

from pydantic import ValidationError
from models import (
    DocumentType, IngestInput, IngestFileInput, QueryInput,
    IngestResponse, QueryResponse, DocsInfo
)


class TestModels(unittest.TestCase):

    def test_document_type_enum(self):
        """Test DocumentType enum values"""
        self.assertEqual(DocumentType.TEXT.value, "text")
        self.assertEqual(DocumentType.PDF.value, "pdf")
        self.assertEqual(DocumentType.DOCX.value, "docx")
        self.assertEqual(DocumentType.HTML.value, "html")
        self.assertEqual(DocumentType.WEB_URL.value, "web_url")
        self.assertEqual(DocumentType.MARKDOWN.value, "markdown")
        self.assertEqual(DocumentType.CSV.value, "csv")
        self.assertEqual(DocumentType.EXCEL.value, "excel")

    def test_ingest_input_text_content(self):
        """Test IngestInput model with text content"""
        data = {
            "content": "Test content",
            "document_type": DocumentType.TEXT,
            "metadata": {"source": "test"},
            "chunk_size": 500,
            "chunk_overlap": 100
        }
        model = IngestInput(**data)

        self.assertEqual(model.content, "Test content")
        self.assertEqual(model.document_type, DocumentType.TEXT)
        self.assertEqual(model.metadata, {"source": "test"})
        self.assertEqual(model.chunk_size, 500)
        self.assertEqual(model.chunk_overlap, 100)
        self.assertIsNone(model.url)

    def test_ingest_input_web_url(self):
        """Test IngestInput model with web URL"""
        data = {
            "url": "https://example.com",
            "document_type": DocumentType.WEB_URL,
            "metadata": {"source": "web"}
        }
        model = IngestInput(**data)

        # Pydantic HttpUrl automatically normalizes URLs by adding trailing slash
        self.assertEqual(str(model.url), "https://example.com/")
        self.assertEqual(model.document_type, DocumentType.WEB_URL)
        self.assertIsNone(model.content)
        # Test defaults
        self.assertEqual(model.chunk_size, 1000)
        self.assertEqual(model.chunk_overlap, 200)

    def test_ingest_input_defaults(self):
        """Test IngestInput model defaults"""
        data = {"content": "Test content"}
        model = IngestInput(**data)

        self.assertEqual(model.document_type, DocumentType.TEXT)
        self.assertEqual(model.metadata, {})
        self.assertEqual(model.chunk_size, 1000)
        self.assertEqual(model.chunk_overlap, 200)

    def test_ingest_file_input(self):
        """Test IngestFileInput model"""
        data = {
            "document_type": DocumentType.PDF,
            "metadata": {"filename": "test.pdf"},
            "chunk_size": 800,
            "chunk_overlap": 150
        }
        model = IngestFileInput(**data)

        self.assertEqual(model.document_type, DocumentType.PDF)
        self.assertEqual(model.metadata, {"filename": "test.pdf"})
        self.assertEqual(model.chunk_size, 800)
        self.assertEqual(model.chunk_overlap, 150)

    def test_query_input_basic(self):
        """Test QueryInput model with required fields only"""
        data = {"query": "Test query"}
        model = QueryInput(**data)

        self.assertEqual(model.query, "Test query")
        self.assertEqual(model.max_results, 5)  # Default value
        self.assertEqual(model.include_metadata, False)  # Default value
        self.assertIsNone(model.similarity_threshold)

    def test_query_input_full(self):
        """Test QueryInput model with all fields"""
        data = {
            "query": "Test query",
            "max_results": 10,
            "include_metadata": True,
            "similarity_threshold": 0.8
        }
        model = QueryInput(**data)

        self.assertEqual(model.query, "Test query")
        self.assertEqual(model.max_results, 10)
        self.assertEqual(model.include_metadata, True)
        self.assertEqual(model.similarity_threshold, 0.8)

    def test_ingest_response_basic(self):
        """Test IngestResponse model with required fields"""
        data = {"status": "success", "document_count": 3}
        model = IngestResponse(**data)

        self.assertEqual(model.status, "success")
        self.assertEqual(model.document_count, 3)
        self.assertIsNone(model.document_id)
        self.assertEqual(model.message, "")  # Default value

    def test_ingest_response_full(self):
        """Test IngestResponse model with all fields"""
        data = {
            "status": "success",
            "document_count": 5,
            "document_id": "doc-123",
            "message": "Document processed successfully"
        }
        model = IngestResponse(**data)

        self.assertEqual(model.status, "success")
        self.assertEqual(model.document_count, 5)
        self.assertEqual(model.document_id, "doc-123")
        self.assertEqual(model.message, "Document processed successfully")

    def test_query_response_basic(self):
        """Test QueryResponse model with required fields"""
        data = {
            "answer": "Test answer",
            "sources": [
                {"content": "Source 1", "metadata": {"id": "1"}},
                {"content": "Source 2", "metadata": {"id": "2"}}
            ],
            "source_count": 2
        }
        model = QueryResponse(**data)

        self.assertEqual(model.answer, "Test answer")
        self.assertEqual(len(model.sources), 2)
        self.assertEqual(model.source_count, 2)
        self.assertIsNone(model.query_time)

    def test_query_response_with_query_time(self):
        """Test QueryResponse model with query time"""
        data = {
            "answer": "Test answer",
            "sources": [],
            "source_count": 0,
            "query_time": 0.125
        }
        model = QueryResponse(**data)

        self.assertEqual(model.query_time, 0.125)

    def test_document_info(self):
        """Test DocumentInfo model"""
        data = {
            "document_id": "doc-456",  # Changed from "id"
            "document_type": "pdf",  # Or DocumentType.PDF.value
            "filename": "test.pdf",
            "chunk_count": 5,  # Added required field
            "created_at": datetime(2024, 1, 1),  # Use datetime object
        }

        model = DocsInfo(**data)

        self.assertEqual(model.filename, "test.pdf")
        self.assertEqual(model.chunk_count, 5)

    def test_invalid_document_type(self):
        """Test validation with invalid document type"""
        with self.assertRaises(ValidationError):
            IngestInput(content="test", document_type="invalid_type")

    def test_invalid_url(self):
        """Test validation with invalid URL"""
        with self.assertRaises(ValidationError):
            IngestInput(url="not-a-valid-url", document_type=DocumentType.WEB_URL)

    def test_empty_query(self):
        """Test that empty query string is still valid (validation happens in endpoint)"""
        data = {"query": ""}
        model = QueryInput(**data)
        self.assertEqual(model.query, "")


if __name__ == "__main__":
    unittest.main()