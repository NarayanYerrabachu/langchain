import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app import app
from models import IngestInput, QueryInput


class TestDocumentRoutes(unittest.TestCase):
    
    def setUp(self):
        self.client = TestClient(app)
    
    @patch('routes.documents.get_vectorstore')
    def test_ingest_document_success(self, mock_get_vectorstore):
        """Test successful document ingestion"""
        mock_vectorstore = MagicMock()
        mock_get_vectorstore.return_value = mock_vectorstore
        
        response = self.client.post(
            "/ingest",
            json={"content": "Test document content", "metadata": {"source": "test"}}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertGreater(data["document_count"], 0)
        mock_vectorstore.add_documents.assert_called_once()
    
    def test_ingest_document_empty_content(self):
        """Test ingestion with empty content"""
        response = self.client.post(
            "/ingest",
            json={"content": "", "metadata": {"source": "test"}}
        )
        
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("Content cannot be empty", data["detail"])
    
    @patch('routes.documents.get_vectorstore')
    def test_ingest_document_error(self, mock_get_vectorstore):
        """Test error handling during ingestion"""
        mock_get_vectorstore.side_effect = Exception("Test error")
        
        response = self.client.post(
            "/ingest",
            json={"content": "Test document content"}
        )
        
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("Failed to ingest document", data["detail"])
    
    @patch('routes.documents.create_qa_chain')
    def test_query_documents_success(self, mock_create_qa_chain):
        """Test successful document query"""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "result": "Test answer",
            "source_documents": [
                MagicMock(page_content="Source 1"),
                MagicMock(page_content="Source 2")
            ]
        }
        mock_create_qa_chain.return_value = mock_chain
        
        response = self.client.post(
            "/query",
            json={"query": "Test question", "max_results": 3}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["answer"], "Test answer")
        self.assertEqual(len(data["sources"]), 2)
        self.assertEqual(data["source_count"], 2)
        mock_create_qa_chain.assert_called_once_with(k=3)
    
    def test_query_documents_empty_query(self):
        """Test query with empty question"""
        response = self.client.post(
            "/query",
            json={"query": ""}
        )
        
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("Query cannot be empty", data["detail"])
    
    @patch('routes.documents.create_qa_chain')
    def test_query_documents_error(self, mock_create_qa_chain):
        """Test error handling during query"""
        mock_create_qa_chain.side_effect = Exception("Test error")
        
        response = self.client.post(
            "/query",
            json={"query": "Test question"}
        )
        
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("Failed to process query", data["detail"])


if __name__ == "__main__":
    unittest.main()
