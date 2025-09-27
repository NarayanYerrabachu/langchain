import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app import app


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
        self.assertEqual(data["version"], "2.0.0")
        self.assertIn("features", data)
        self.assertIsInstance(data["features"], list)

    @patch('routes.health.health_check_database')
    @patch('routes.health.run_qa_chain_test')
    @patch('routes.health.OPENAI_API_KEY', None)
    def test_health_check_comprehensive(self, mock_qa_test, mock_db_health):
        """Test comprehensive health check endpoint"""
        mock_db_health.return_value = {
            "status": "healthy",
            "message": "Database connection is working"
        }
        mock_qa_test.return_value = {
            "status": "success",
            "answer": "Test answer",
            "source_count": 1
        }

        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("status", data)
        self.assertIn("timestamp", data)
        self.assertIn("services", data)
        self.assertIn("configuration", data)
        self.assertIn("database", data["services"])
        self.assertIn("openai", data["services"])
        self.assertIn("qa_chain", data["services"])

    @patch('routes.health.health_check_database')
    @patch('routes.health.run_qa_chain_test')
    def test_health_check_unhealthy_database(self, mock_qa_test, mock_db_health):
        """Test health check when database is unhealthy"""
        mock_db_health.return_value = {
            "status": "unhealthy",
            "message": "Database connection failed"
        }
        mock_qa_test.return_value = {"status": "success"}

        response = self.client.get("/health")
        self.assertEqual(response.status_code, 503)

    def test_simple_health_check(self):
        """Test simple health check endpoint"""
        response = self.client.get("/health/simple")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")

    @patch('routes.health.engine', None)
    @patch('routes.health.ENABLE_DATABASE', True)
    def test_simple_health_check_no_engine(self):
        """Test simple health check when engine is None but database enabled"""
        response = self.client.get("/health/simple")
        self.assertEqual(response.status_code, 200)  # Should still pass without database

    @patch('routes.health.health_check_database')
    def test_database_health_endpoint(self, mock_db_health):
        """Test database-specific health endpoint"""
        mock_db_health.return_value = {
            "status": "healthy",
            "message": "Database connection is working"
        }

        response = self.client.get("/health/database")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")

    def test_services_health_endpoint(self):
        """Test services health endpoint"""
        response = self.client.get("/health/services")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("services", data)
        self.assertIn("database", data["services"])
        self.assertIn("openai", data["services"])
        self.assertIn("qa_chain", data["services"])

    def test_system_info_endpoint(self):
        """Test system info endpoint"""
        response = self.client.get("/info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["api_version"], "2.0.0")
        self.assertIn("python_version", data)
        self.assertIn("configuration", data)
        self.assertIn("supported_formats", data)


if __name__ == "__main__":
    unittest.main()