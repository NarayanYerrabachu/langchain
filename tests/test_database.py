import unittest
from unittest.mock import patch, MagicMock

import database
from database import (
    MockVectorStore, MockRetriever, MockEmbeddings,
    get_vectorstore, get_retriever, health_check_database, init_database
)
from langchain_core.documents import Document


class TestDatabase(unittest.TestCase):

    def test_mock_embeddings(self):
        """Test MockEmbeddings functionality"""
        mock_embeddings = MockEmbeddings()

        # Test embed_documents
        texts = ["text1", "text2", "text3"]
        embeddings = mock_embeddings.embed_documents(texts)
        self.assertEqual(len(embeddings), 3)
        self.assertEqual(len(embeddings[0]), 1536)  # Standard embedding dimension

        # Test embed_query
        query_embedding = mock_embeddings.embed_query("test query")
        self.assertEqual(len(query_embedding), 1536)

    def test_mock_vector_store(self):
        """Test the mock vector store implementation"""
        mock_store = MockVectorStore()

        # Test add_documents
        docs = [
            Document(page_content="Doc 1", metadata={"id": "1"}),
            Document(page_content="Doc 2", metadata={"id": "2"})
        ]
        result = mock_store.add_documents(docs)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(mock_store._documents), 2)

        # Test similarity_search
        search_results = mock_store.similarity_search("test query", k=5)
        self.assertEqual(len(search_results), 2)  # Should return stored documents

        # Test similarity_search with no stored documents
        mock_store._documents.clear()
        search_results = mock_store.similarity_search("test query", k=5)
        self.assertEqual(len(search_results), 1)  # Should return default mock document
        self.assertIn("mock content", search_results[0].page_content.lower())

        # Test as_retriever
        retriever = mock_store.as_retriever(search_kwargs={"k": 3})
        self.assertIsInstance(retriever, MockRetriever)

        # Test delete_collection
        mock_store.add_documents(docs)  # Add docs back
        result = mock_store.delete_collection()
        self.assertTrue(result)
        self.assertEqual(len(mock_store._documents), 0)

    def test_mock_retriever(self):
        """Test the mock retriever implementation"""
        mock_store = MockVectorStore()
        docs = [Document(page_content="Test content", metadata={"id": "1"})]
        mock_store.add_documents(docs)

        retriever = MockRetriever(mock_store, {"k": 3})

        # Test get_relevant_documents
        results = retriever.get_relevant_documents("test query")
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], Document)

    @patch('database.ENABLE_DATABASE', False)
    def test_get_vectorstore_when_disabled(self):
        """Test get_vectorstore returns mock when database is disabled"""
        vectorstore = get_vectorstore()
        self.assertIsInstance(vectorstore, MockVectorStore)

    @patch('database.engine', None)
    def test_get_vectorstore_when_engine_none(self):
        """Test get_vectorstore returns mock when engine is None"""
        vectorstore = get_vectorstore()
        self.assertIsInstance(vectorstore, MockVectorStore)

    @patch('database.embeddings', None)
    def test_get_vectorstore_when_embeddings_none(self):
        """Test get_vectorstore returns mock when embeddings is None"""
        with patch('database.engine', MagicMock()):
            vectorstore = get_vectorstore()
            self.assertIsInstance(vectorstore, MockVectorStore)

    @patch('database.PGVector')
    @patch('database.engine', MagicMock())
    @patch('database.embeddings', MagicMock())
    @patch('database.ENABLE_DATABASE', True)
    def test_get_vectorstore_success(self, mock_pgvector):
        """Test get_vectorstore returns PGVector when everything is set up"""
        mock_pgvector_instance = MagicMock()
        mock_pgvector.return_value = mock_pgvector_instance

        vectorstore = get_vectorstore()

        mock_pgvector.assert_called_once()
        self.assertEqual(vectorstore, mock_pgvector_instance)

    @patch('database.PGVector')
    @patch('database.engine', MagicMock())
    @patch('database.embeddings', MagicMock())
    @patch('database.ENABLE_DATABASE', True)
    def test_get_vectorstore_exception(self, mock_pgvector):
        """Test get_vectorstore returns mock when PGVector raises exception"""
        mock_pgvector.side_effect = Exception("Test exception")

        vectorstore = get_vectorstore()

        self.assertIsInstance(vectorstore, MockVectorStore)

    def test_get_retriever(self):
        """Test get_retriever function"""
        with patch('database.get_vectorstore') as mock_get_vectorstore:
            mock_vectorstore = MagicMock()
            mock_retriever = MagicMock()
            mock_vectorstore.as_retriever.return_value = mock_retriever
            mock_get_vectorstore.return_value = mock_vectorstore

            retriever = get_retriever(k=7)

            mock_vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 7})
            self.assertEqual(retriever, mock_retriever)

    @patch('database.ENABLE_DATABASE', False)
    def test_health_check_database_disabled(self):
        """Test health check when database is disabled"""
        result = health_check_database()

        self.assertEqual(result["status"], "disabled")
        self.assertIn("disabled by configuration", result["message"])

    @patch('database.engine', None)
    @patch('database.ENABLE_DATABASE', True)
    def test_health_check_database_no_engine(self):
        """Test health check when engine is None"""
        result = health_check_database()

        self.assertEqual(result["status"], "unavailable")
        self.assertIn("not initialized", result["message"])

    @patch('database.engine')
    @patch('database.ENABLE_DATABASE', True)
    def test_health_check_database_success(self, mock_engine):
        """Test health check when database is healthy"""
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        result = health_check_database()

        self.assertEqual(result["status"], "healthy")
        self.assertIn("working", result["message"])
        mock_conn.execute.assert_called_once()

    @patch('database.engine')
    @patch('database.ENABLE_DATABASE', True)
    def test_health_check_database_error(self, mock_engine):
        """Test health check when database connection fails"""
        mock_engine.connect.side_effect = Exception("Connection failed")

        result = health_check_database()

        self.assertEqual(result["status"], "unhealthy")
        self.assertIn("Connection failed", result["message"])

    @patch('database.ENABLE_DATABASE', True)
    @patch('database.OPENAI_API_KEY', 'sk-test-key')
    @patch('database.OpenAIEmbeddings')
    @patch('database.create_engine')
    def test_init_database_success(self, mock_create_engine, mock_openai_embeddings):
        """Test successful database initialization"""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_embeddings = MagicMock()
        mock_openai_embeddings.return_value = mock_embeddings

        # Reset global variables
        database.engine = None
        database.embeddings = None

        init_database()

        self.assertEqual(database.engine, mock_engine)
        self.assertEqual(database.embeddings, mock_embeddings)
        mock_conn.execute.assert_called_once()

    @patch('database.ENABLE_DATABASE', False)
    def test_init_database_disabled(self):
        """Test database initialization when disabled"""
        database.engine = None
        database.embeddings = None

        init_database()

        self.assertIsNone(database.engine)
        self.assertIsNone(database.embeddings)

    @patch('database.ENABLE_DATABASE', True)
    @patch('database.create_engine')
    def test_init_database_connection_error(self, mock_create_engine):
        """Test database initialization with connection error"""
        mock_create_engine.side_effect = Exception("Connection error")

        # Reset global variables
        database.engine = None
        database.embeddings = None

        init_database()

        self.assertIsNone(database.engine)
        self.assertIsInstance(database.embeddings, MockEmbeddings)


if __name__ == "__main__":
    unittest.main()