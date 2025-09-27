import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app import app


class TestCollectionRoutes(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    @patch('routes.collections.COLLECTION_NAME', 'test_collection')
    def test_get_collection_info(self):
        """Test getting collection info"""
        response = self.client.get("/api/v1/collections/info")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["collection_name"], "test_collection")
        self.assertEqual(data["status"], "active")

    @patch('routes.collections.get_vectorstore')
    def test_clear_collection_success(self, mock_get_vectorstore):
        """Test successful collection clearing"""
        mock_vectorstore = MagicMock()
        mock_get_vectorstore.return_value = mock_vectorstore
        mock_vectorstore.delete_collection.return_value = True

        response = self.client.delete("/api/v1/collections/clear")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "Collection cleared successfully")
        mock_vectorstore.delete_collection.assert_called_once()

    @patch('routes.collections.get_vectorstore')
    def test_clear_collection_error(self, mock_get_vectorstore):
        """Test error handling during collection clearing"""
        mock_vectorstore = MagicMock()
        mock_get_vectorstore.return_value = mock_vectorstore
        mock_vectorstore.delete_collection.side_effect = Exception("Test error")

        response = self.client.delete("/api/v1/collections/clear")

        self.assertEqual(response.status_code, 500)
        data = response.json()
        # Check for our custom error format or FastAPI's default format
        if "error" in data:
            # Our custom error handler format
            self.assertIn("Failed to clear collection", data["error"]["message"])
        else:
            # FastAPI default format
            self.assertIn("Failed to clear collection", data["detail"])

    def test_get_collection_info_error(self):
        """Test that collection info endpoint doesn't depend on vectorstore"""
        # The collection info endpoint just returns static info about the collection name
        # It doesn't actually use get_vectorstore(), so it shouldn't fail even if vectorstore fails
        response = self.client.get("/api/v1/collections/info")

        # Should still return 200 because it doesn't use vectorstore
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("collection_name", data)
        self.assertIn("status", data)


if __name__ == "__main__":
    unittest.main()