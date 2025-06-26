import unittest
from fastapi.testclient import TestClient

from app import app


class TestApp(unittest.TestCase):
    
    def setUp(self):
        self.client = TestClient(app)
    
    def test_app_title(self):
        """Test the FastAPI app title"""
        self.assertEqual(app.title, "RAG Document Q&A API")
    
    def test_app_version(self):
        """Test the FastAPI app version"""
        self.assertEqual(app.version, "1.0.0")
    
    def test_openapi_schema(self):
        """Test that OpenAPI schema is generated correctly"""
        openapi_schema = app.openapi()
        self.assertIsNotNone(openapi_schema)
        self.assertIn("info", openapi_schema)
        self.assertIn("paths", openapi_schema)
        
        # Check that our endpoints are in the schema
        paths = openapi_schema["paths"]
        self.assertIn("/", paths)
        self.assertIn("/health", paths)
        self.assertIn("/ingest", paths)
        self.assertIn("/query", paths)
        self.assertIn("/collections/info", paths)
        self.assertIn("/collections/clear", paths)


if __name__ == "__main__":
    unittest.main()
