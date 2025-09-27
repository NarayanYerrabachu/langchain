import pytest
import os
import sys
from unittest.mock import patch

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables"""
    test_env = {
        "OPENAI_API_KEY": "dummy-key-for-testing",
        "DATABASE_URL": "postgresql://test:test@localhost:5432/test_db",
        "COLLECTION_NAME": "test_collection",
        "ENABLE_DATABASE": "false",  # Use mock database for tests
        "MAX_FILE_SIZE": "10",  # 10MB for tests
        "LOG_LEVEL": "WARNING",  # Reduce log noise during tests
    }

    with patch.dict(os.environ, test_env):
        yield


@pytest.fixture
def mock_database():
    """Mock database components for tests"""
    with patch('database.engine', None), \
            patch('database.embeddings', None), \
            patch('database.ENABLE_DATABASE', False):
        yield


@pytest.fixture
def mock_openai():
    """Mock OpenAI API for tests"""
    with patch('qa_chain.OPENAI_API_KEY', 'dummy-key-for-development'):
        yield


@pytest.fixture
def sample_text_content():
    """Sample text content for testing"""
    return """
    This is sample text content for testing purposes.
    It contains multiple sentences and paragraphs.

    This is a second paragraph with more content.
    It should be long enough to test text splitting functionality.
    """


@pytest.fixture
def sample_html_content():
    """Sample HTML content for testing"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Document</title>
        <meta name="description" content="Test description">
    </head>
    <body>
        <h1>Main Heading</h1>
        <p>This is a test paragraph with <strong>bold text</strong>.</p>
        <div>
            <p>Another paragraph in a div.</p>
        </div>
        <script>
            // This script should be removed
            console.log("test");
        </script>
        <style>
            /* This style should be removed */
            body { color: red; }
        </style>
    </body>
    </html>
    """


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing"""
    return {
        "source": "test",
        "author": "test_author",
        "created_at": "2024-01-01T00:00:00Z",
        "topic": "testing"
    }