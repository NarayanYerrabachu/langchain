import unittest
from models import IngestInput, QueryInput, IngestResponse, QueryResponse


class TestModels(unittest.TestCase):
    
    def test_ingest_input(self):
        """Test IngestInput model"""
        # Test with required fields only
        data = {"content": "Test content"}
        model = IngestInput(**data)
        self.assertEqual(model.content, "Test content")
        self.assertEqual(model.metadata, {})
        
        # Test with metadata
        data = {"content": "Test content", "metadata": {"source": "test", "author": "tester"}}
        model = IngestInput(**data)
        self.assertEqual(model.content, "Test content")
        self.assertEqual(model.metadata, {"source": "test", "author": "tester"})
    
    def test_query_input(self):
        """Test QueryInput model"""
        # Test with required fields only
        data = {"query": "Test query"}
        model = QueryInput(**data)
        self.assertEqual(model.query, "Test query")
        self.assertEqual(model.max_results, 5)  # Default value
        
        # Test with max_results
        data = {"query": "Test query", "max_results": 10}
        model = QueryInput(**data)
        self.assertEqual(model.query, "Test query")
        self.assertEqual(model.max_results, 10)
    
    def test_ingest_response(self):
        """Test IngestResponse model"""
        data = {"status": "success", "document_count": 3}
        model = IngestResponse(**data)
        self.assertEqual(model.status, "success")
        self.assertEqual(model.document_count, 3)
    
    def test_query_response(self):
        """Test QueryResponse model"""
        data = {
            "answer": "Test answer",
            "sources": ["Source 1", "Source 2"],
            "source_count": 2
        }
        model = QueryResponse(**data)
        self.assertEqual(model.answer, "Test answer")
        self.assertEqual(model.sources, ["Source 1", "Source 2"])
        self.assertEqual(model.source_count, 2)


if __name__ == "__main__":
    unittest.main()
