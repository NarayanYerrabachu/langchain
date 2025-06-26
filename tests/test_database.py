import unittest
from unittest.mock import patch, MagicMock

import database
from database import MockVectorStore, MockRetriever


class TestDatabase(unittest.TestCase):
    
    def test_mock_vector_store(self):
        """Test the mock vector store implementation"""
        mock_store = MockVectorStore()
        
        # Test add_documents
        docs = [MagicMock(), MagicMock()]
        result = mock_store.add_documents(docs)
        self.assertEqual(result, 2)
        
        # Test as_retriever
        retriever = mock_store.as_retriever()
        self.assertIsInstance(retriever, MockRetriever)
        
        # Test delete_collection
        result = mock_store.delete_collection()
        self.assertTrue(result)
    
    def test_mock_retriever(self):
        """Test the mock retriever implementation"""
        retriever = MockRetriever()
        docs = retriever.get_relevant_documents("test query")
        
        self.assertEqual(len(docs), 1)
        self.assertTrue(hasattr(docs[0], 'page_content'))
        self.assertTrue(isinstance(docs[0].page_content, str))
    
    @patch('database.ENABLE_DATABASE', False)
    def test_get_vectorstore_when_disabled(self):
        """Test get_vectorstore returns mock when database is disabled"""
        vectorstore = database.get_vectorstore()
        self.assertIsInstance(vectorstore, MockVectorStore)
    
    @patch('database.engine', None)
    def test_get_vectorstore_when_engine_none(self):
        """Test get_vectorstore returns mock when engine is None"""
        vectorstore = database.get_vectorstore()
        self.assertIsInstance(vectorstore, MockVectorStore)
    
    @patch('database.PGVector')
    @patch('database.engine', MagicMock())
    @patch('database.embeddings', MagicMock())
    @patch('database.ENABLE_DATABASE', True)
    def test_get_vectorstore_success(self, mock_pgvector):
        """Test get_vectorstore returns PGVector when everything is set up"""
        mock_pgvector_instance = MagicMock()
        mock_pgvector.return_value = mock_pgvector_instance
        
        vectorstore = database.get_vectorstore()
        
        mock_pgvector.assert_called_once()
        self.assertEqual(vectorstore, mock_pgvector_instance)
    
    @patch('database.PGVector')
    @patch('database.engine', MagicMock())
    @patch('database.embeddings', MagicMock())
    @patch('database.ENABLE_DATABASE', True)
    def test_get_vectorstore_exception(self, mock_pgvector):
        """Test get_vectorstore returns mock when PGVector raises exception"""
        mock_pgvector.side_effect = Exception("Test exception")
        
        vectorstore = database.get_vectorstore()
        
        self.assertIsInstance(vectorstore, MockVectorStore)


if __name__ == "__main__":
    unittest.main()
