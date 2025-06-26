import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app import app
from routes import health


class TestHealthRoutes(unittest.TestCase):
    
    def setUp(self):
        self.client = TestClient(app)
    
    def test_root_endpoint(self):
        """Test the root endpoint returns correct response"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["message"], "RAG Document Q&A API")
        self.assertEqual(data["status"], "running")
    
    @patch('routes.health.engine', None)
    def test_health_check_no_engine(self):
        """Test health check when database engine is None"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "degraded")
        self.assertEqual(data["database"], "not connected")
    
    @patch('routes.health.engine')
    def test_health_check_success(self, mock_engine):
        """Test health check when database is connected"""
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["database"], "connected")
    
    @patch('routes.health.engine')
    def test_health_check_db_error(self, mock_engine):
        """Test health check when database connection fails"""
        mock_engine.connect.side_effect = SQLAlchemyError("Test database error")
        
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("Database connection failed", data["detail"])


if __name__ == "__main__":
    unittest.main()
