import os
import unittest
from unittest.mock import patch

import config


class TestConfig(unittest.TestCase):

    def test_logging_setup(self):
        """Test that logging is properly configured"""
        self.assertIsNotNone(config.logger)

    def test_database_config_defaults(self):
        """Test database configuration default values"""
        self.assertIsNotNone(config.CONNECTION_STRING)
        self.assertTrue(isinstance(config.CONNECTION_STRING, str))
        self.assertIsNotNone(config.COLLECTION_NAME)
        self.assertTrue(isinstance(config.COLLECTION_NAME, str))
        self.assertTrue(isinstance(config.ENABLE_DATABASE, bool))

    @patch.dict(os.environ, {"DATABASE_URL": "postgresql://test:test@localhost:5432/test"})
    def test_database_config_from_env(self):
        """Test database configuration from environment variables"""
        import importlib
        importlib.reload(config)

        self.assertEqual(config.CONNECTION_STRING, "postgresql://test:test@localhost:5432/test")

    @patch.dict(os.environ, {"COLLECTION_NAME": "test_collection"})
    def test_collection_name_from_env(self):
        """Test collection name from environment variable"""
        import importlib
        importlib.reload(config)

        self.assertEqual(config.COLLECTION_NAME, "test_collection")

    @patch.dict(os.environ, {"ENABLE_DATABASE": "false"})
    def test_enable_database_false(self):
        """Test ENABLE_DATABASE set to false"""
        import importlib
        importlib.reload(config)

        self.assertEqual(config.ENABLE_DATABASE, False)

    @patch.dict(os.environ, {"ENABLE_DATABASE": "true"})
    def test_enable_database_true(self):
        """Test ENABLE_DATABASE set to true"""
        import importlib
        importlib.reload(config)

        self.assertEqual(config.ENABLE_DATABASE, True)

    @patch.dict(os.environ, {}, clear=True)
    def test_openai_api_key_fallback(self):
        """Test that a dummy API key is set when environment variable is missing"""
        import importlib
        importlib.reload(config)

        self.assertEqual(os.environ.get("OPENAI_API_KEY"), "dummy-key-for-development")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-real-key"})
    def test_openai_api_key_from_env(self):
        """Test OpenAI API key from environment variable"""
        import importlib
        importlib.reload(config)

        self.assertEqual(config.OPENAI_API_KEY, "sk-real-key")

    def test_file_upload_config(self):
        """Test file upload configuration"""
        self.assertIsInstance(config.MAX_FILE_SIZE, int)
        self.assertGreater(config.MAX_FILE_SIZE, 0)
        self.assertIsInstance(config.ALLOWED_EXTENSIONS, set)
        self.assertIn('pdf', config.ALLOWED_EXTENSIONS)
        self.assertIn('docx', config.ALLOWED_EXTENSIONS)
        self.assertIn('txt', config.ALLOWED_EXTENSIONS)

    @patch.dict(os.environ, {"MAX_FILE_SIZE": "100"})
    def test_max_file_size_from_env(self):
        """Test MAX_FILE_SIZE from environment variable"""
        import importlib
        importlib.reload(config)

        expected_size = 100 * 1024 * 1024  # 100MB in bytes
        self.assertEqual(config.MAX_FILE_SIZE, expected_size)

    def test_text_processing_config(self):
        """Test text processing configuration"""
        self.assertIsInstance(config.DEFAULT_CHUNK_SIZE, int)
        self.assertIsInstance(config.DEFAULT_CHUNK_OVERLAP, int)
        self.assertIsInstance(config.MAX_CHUNK_SIZE, int)
        self.assertGreater(config.DEFAULT_CHUNK_SIZE, 0)
        self.assertGreater(config.MAX_CHUNK_SIZE, config.DEFAULT_CHUNK_SIZE)

    @patch.dict(os.environ, {
        "DEFAULT_CHUNK_SIZE": "500",
        "DEFAULT_CHUNK_OVERLAP": "50",
        "MAX_CHUNK_SIZE": "2000"
    })
    def test_text_processing_config_from_env(self):
        """Test text processing config from environment variables"""
        import importlib
        importlib.reload(config)

        self.assertEqual(config.DEFAULT_CHUNK_SIZE, 500)
        self.assertEqual(config.DEFAULT_CHUNK_OVERLAP, 50)
        self.assertEqual(config.MAX_CHUNK_SIZE, 2000)

    def test_web_scraping_config(self):
        """Test web scraping configuration"""
        self.assertIsInstance(config.REQUEST_TIMEOUT, int)
        self.assertIsInstance(config.MAX_RETRIES, int)
        self.assertGreater(config.REQUEST_TIMEOUT, 0)
        self.assertGreater(config.MAX_RETRIES, 0)

    def test_vector_store_config(self):
        """Test vector store configuration"""
        self.assertIsInstance(config.EMBEDDING_MODEL, str)
        self.assertIsInstance(config.VECTOR_DIMENSIONS, int)
        self.assertGreater(config.VECTOR_DIMENSIONS, 0)
        self.assertEqual(config.VECTOR_DIMENSIONS, 1536)  # OpenAI ada-002 dimension

    @patch.dict(os.environ, {
        "EMBEDDING_MODEL": "text-embedding-3-large",
        "VECTOR_DIMENSIONS": "3072"
    })
    def test_vector_store_config_from_env(self):
        """Test vector store config from environment variables"""
        import importlib
        importlib.reload(config)

        self.assertEqual(config.EMBEDDING_MODEL, "text-embedding-3-large")
        self.assertEqual(config.VECTOR_DIMENSIONS, 3072)

    def test_security_config(self):
        """Test security configuration"""
        self.assertIsInstance(config.ALLOWED_DOMAINS, list)
        self.assertIsInstance(config.BLOCKED_DOMAINS, list)
        self.assertIsInstance(config.RATE_LIMIT_PER_MINUTE, int)

    @patch.dict(os.environ, {
        "ALLOWED_DOMAINS": "example.com,trusted.org",
        "BLOCKED_DOMAINS": "malicious.com,spam.net"
    })
    def test_security_config_from_env(self):
        """Test security config from environment variables"""
        import importlib
        importlib.reload(config)

        self.assertEqual(config.ALLOWED_DOMAINS, ["example.com", "trusted.org"])
        self.assertEqual(config.BLOCKED_DOMAINS, ["malicious.com", "spam.net"])

    @patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"})
    def test_log_level_from_env(self):
        """Test log level configuration from environment"""
        import importlib
        import logging
        importlib.reload(config)

        # Check that the log level was set
        self.assertEqual(logging.getLogger().level, logging.DEBUG)

    def test_config_constants_exist(self):
        """Test that all expected configuration constants exist"""
        expected_constants = [
            'CONNECTION_STRING', 'COLLECTION_NAME', 'ENABLE_DATABASE',
            'OPENAI_API_KEY', 'MAX_FILE_SIZE', 'ALLOWED_EXTENSIONS',
            'DEFAULT_CHUNK_SIZE', 'DEFAULT_CHUNK_OVERLAP', 'MAX_CHUNK_SIZE',
            'REQUEST_TIMEOUT', 'MAX_RETRIES',
            'EMBEDDING_MODEL', 'VECTOR_DIMENSIONS',
            'ALLOWED_DOMAINS', 'BLOCKED_DOMAINS', 'RATE_LIMIT_PER_MINUTE',
            'LOG_LEVEL', 'logger'
        ]

        for constant in expected_constants:
            self.assertTrue(hasattr(config, constant), f"Missing constant: {constant}")


if __name__ == "__main__":
    unittest.main()