import unittest
import time
from typing import List
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from io import BytesIO

from app import app
from models import DocumentType


class TestDocumentRoutes(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    @patch('routes.documents.doc_processor')
    @patch('routes.documents.get_vectorstore')
    def test_ingest_document_text_success(self, mock_get_vectorstore, mock_doc_processor):
        """Test successful text document ingestion"""
        mock_vectorstore = MagicMock()
        mock_get_vectorstore.return_value = mock_vectorstore

        mock_chunks = [MagicMock()]
        mock_chunks[0].metadata = {"document_id": "test-123"}
        mock_doc_processor.process_document.return_value = mock_chunks

        response = self.client.post(
            "/api/v1/ingest",
            json={
                "content": "Test document content",
                "document_type": "text",
                "metadata": {"source": "test"},
                "chunk_size": 1000,
                "chunk_overlap": 200
            }
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["document_count"], 1)
        self.assertIn("document_id", data)
        mock_vectorstore.add_documents.assert_called_once()
        mock_doc_processor.process_document.assert_called_once()

    @patch('routes.documents.doc_processor')
    @patch('routes.documents.get_vectorstore')
    def test_ingest_document_web_url_success(self, mock_get_vectorstore, mock_doc_processor):
        """Test successful web URL document ingestion"""
        mock_vectorstore = MagicMock()
        mock_get_vectorstore.return_value = mock_vectorstore

        mock_chunks = [MagicMock()]
        mock_chunks[0].metadata = {"document_id": "test-456"}
        mock_doc_processor.process_document.return_value = mock_chunks

        response = self.client.post(
            "/api/v1/ingest",
            json={
                "url": "https://example.com",
                "document_type": "web_url",
                "metadata": {"source": "web"}
            }
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["document_count"], 1)

    def test_ingest_document_web_url_missing_url(self):
        """Test ingestion with web_url type but missing URL"""
        response = self.client.post(
            "/api/v1/ingest",
            json={
                "document_type": "web_url",
                "metadata": {"source": "test"}
            }
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        # Check for our custom error format or FastAPI's default format
        if "error" in data:
            self.assertIn("URL is required for web_url document type", data["error"]["message"])
        else:
            self.assertIn("URL is required for web_url document type", data["detail"])

    def test_ingest_document_text_missing_content(self):
        """Test ingestion with text type but missing content"""
        response = self.client.post(
            "/api/v1/ingest",
            json={
                "document_type": "text",
                "metadata": {"source": "test"}
            }
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        # Check for our custom error format or FastAPI's default format
        if "error" in data:
            self.assertIn("Content is required for text document type", data["error"]["message"])
        else:
            self.assertIn("Content is required for text document type", data["detail"])

    @patch('routes.documents.doc_processor')
    def test_ingest_document_processing_error(self, mock_doc_processor):
        """Test error handling during document processing"""
        mock_doc_processor.process_document.side_effect = Exception("Processing error")

        response = self.client.post(
            "/api/v1/ingest",
            json={
                "content": "Test document content",
                "document_type": "text"
            }
        )

        self.assertEqual(response.status_code, 500)
        data = response.json()
        # Check for our custom error format or FastAPI's default format
        if "error" in data:
            self.assertIn("Failed to ingest document", data["error"]["message"])
        else:
            self.assertIn("Failed to ingest document", data["detail"])

    @patch('routes.documents.doc_processor')
    @patch('routes.documents.get_vectorstore')
    def test_ingest_file_success(self, mock_get_vectorstore, mock_doc_processor):
        """Test successful file upload and ingestion"""
        mock_vectorstore = MagicMock()
        mock_get_vectorstore.return_value = mock_vectorstore

        mock_chunks = [MagicMock()]
        mock_chunks[0].metadata = {"document_id": "test-789"}
        mock_doc_processor.process_document.return_value = mock_chunks

        # Create a mock file
        file_content = b"Test PDF content"

        response = self.client.post(
            "/api/v1/ingest/file",
            files={"file": ("test.pdf", BytesIO(file_content), "application/pdf")},
            data={
                "document_type": "pdf",
                "metadata": '{"source": "upload"}',
                "chunk_size": "1000",
                "chunk_overlap": "200"
            }
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["document_count"], 1)
        mock_doc_processor.process_document.assert_called_once()

    def test_ingest_file_invalid_metadata_json(self):
        """Test file upload with invalid JSON metadata"""
        file_content = b"Test content"

        response = self.client.post(
            "/api/v1/ingest/file",
            files={"file": ("test.txt", BytesIO(file_content), "text/plain")},
            data={
                "document_type": "text",
                "metadata": "invalid json",
            }
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        # Check for our custom error format or FastAPI's default format
        if "error" in data:
            self.assertIn("Invalid JSON in metadata field", data["error"]["message"])
        else:
            self.assertIn("Invalid JSON in metadata field", data["detail"])

    def test_ingest_file_empty_file(self):
        """Test file upload with empty file"""
        response = self.client.post(
            "/api/v1/ingest/file",
            files={"file": ("empty.txt", BytesIO(b""), "text/plain")},
            data={
                "document_type": "text",
                "metadata": "{}",
            }
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        # Check for our custom error format or FastAPI's default format
        if "error" in data:
            self.assertIn("Empty file uploaded", data["error"]["message"])
        else:
            self.assertIn("Empty file uploaded", data["detail"])

    @patch('routes.documents.create_qa_chain')
    def test_query_documents_success(self, mock_create_qa_chain):
        """Test successful document query"""
        mock_chain = MagicMock()
        mock_source_doc = MagicMock()
        mock_source_doc.page_content = "Test source content"
        mock_source_doc.metadata = {"document_id": "test-123", "filename": "test.pdf"}

        mock_chain.invoke.return_value = {
            "result": "Test answer",
            "source_documents": [mock_source_doc]
        }
        mock_create_qa_chain.return_value = mock_chain

        response = self.client.post(
            "/api/v1/query",
            json={
                "query": "Test question",
                "max_results": 3,
                "include_metadata": True
            }
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["answer"], "Test answer")
        self.assertEqual(len(data["sources"]), 1)
        self.assertEqual(data["source_count"], 1)
        self.assertIn("query_time", data)

        # Check source structure
        source = data["sources"][0]
        self.assertIn("content", source)
        self.assertIn("metadata", source)
        mock_create_qa_chain.assert_called_once_with(k=3)

    def test_query_documents_empty_query(self):
        """Test query with empty question"""
        response = self.client.post(
            "/api/v1/query",
            json={"query": ""}
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        # Check for our custom error format or FastAPI's default format
        if "error" in data:
            self.assertIn("Query cannot be empty", data["error"]["message"])
        else:
            self.assertIn("Query cannot be empty", data["detail"])

    @patch('routes.documents.create_qa_chain')
    def test_query_documents_error(self, mock_create_qa_chain):
        """Test error handling during query"""
        mock_create_qa_chain.side_effect = Exception("Query error")

        response = self.client.post(
            "/api/v1/query",
            json={"query": "Test question"}
        )

        self.assertEqual(response.status_code, 500)
        data = response.json()
        # Check for our custom error format or FastAPI's default format
        if "error" in data:
            self.assertIn("Failed to process query", data["error"]["message"])
        else:
            self.assertIn("Failed to process query", data["detail"])

    def test_list_documents_endpoint(self):
        """Test document listing endpoint"""
        response = self.client.get("/api/v1/documents")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("message", data)

    def test_delete_document_endpoint(self):
        """Test document deletion endpoint"""
        response = self.client.delete("/api/v1/documents/test-123")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("message", data)


if __name__ == "__main__":
    unittest.main()