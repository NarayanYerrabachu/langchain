import os
import unittest
from unittest.mock import patch

import config


class TestConfig(unittest.TestCase):
    
    def test_logging_setup(self):
        """Test that logging is properly configured"""
        self.assertIsNotNone(config.logger)
    
    def test_database_config(self):
        """Test database configuration variables"""
        self.assertIsNotNone(config.CONNECTION_STRING)
        self.assertTrue(isinstance(config.CONNECTION_STRING, str))
        self.assertIsNotNone(config.COLLECTION_NAME)
        self.assertTrue(isinstance(config.COLLECTION_NAME, str))
    
    @patch.dict(os.environ, {}, clear=True)
    def test_openai_api_key_fallback(self):
        """Test that a dummy API key is set when environment variable is missing"""
        # Re-import to trigger the code that sets the dummy key
        import importlib
        importlib.reload(config)
        
        self.assertEqual(os.environ.get("OPENAI_API_KEY"), "dummy-key-for-development")


if __name__ == "__main__":
    unittest.main()
