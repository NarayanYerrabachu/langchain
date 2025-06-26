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
        response = self.client.get("/collections/info")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["collection_name"], "test_collection")
        self.assertEqual(data["status"], "active")
    
    @patch('routes.collections.get_vectorstore')
    def test_clear_collection_success(self, mock_get_vectorstore):
        """Test successful collection clearing"""
        mock_vectorstore = MagicMock()
        mock_get_vectorstore.return_value = mock_vectorstore
        
        response = self.client.delete("/collections/clear")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "Collection cleared successfully")
        mock_vectorstore.delete_collection.assert_called_once()
    
    @patch('routes.collections.get_vectorstore')
    def test_clear_collection_error(self, mock_get_vectorstore):
        """Test error handling during collection clearing"""
        mock_get_vectorstore.return_value.delete_collection.side_effect = Exception("Test error")
        
        response = self.client.delete("/collections/clear")
        
        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertIn("Failed to clear collection", data["detail"])


if __name__ == "__main__":
    unittest.main()
