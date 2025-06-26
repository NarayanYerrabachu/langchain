import unittest
from unittest.mock import patch, MagicMock

from qa_chain import create_qa_chain


class TestQAChain(unittest.TestCase):
    
    @patch('qa_chain.get_retriever')
    def test_create_qa_chain_exception(self, mock_get_retriever):
        """Test error handling when retriever creation fails"""
        mock_get_retriever.side_effect = Exception("Test exception")
        
        # Should return a fallback function
        result = create_qa_chain()
        self.assertTrue(callable(result))
        
        # Call the fallback function and check the result
        response = result({"query": "test"})
        self.assertIn("Error creating QA chain", response["result"])
        self.assertEqual(len(response["source_documents"]), 0)
    
    @patch('qa_chain.os.getenv')
    @patch('qa_chain.get_retriever')
    def test_create_qa_chain_mock_llm(self, mock_get_retriever, mock_getenv):
        """Test mock LLM creation when API key is dummy"""
        mock_retriever = MagicMock()
        mock_get_retriever.return_value = mock_retriever
        mock_getenv.return_value = "dummy-key-for-development"
        
        # Should return a mock function
        result = create_qa_chain()
        self.assertTrue(callable(result))
        
        # Call the mock function and check the result
        response = result({"query": "test question"})
        self.assertIn("mock answer", response["result"].lower())
        self.assertIn("test question", response["result"])
        mock_retriever.get_relevant_documents.assert_called_once_with("test question")
    
    @patch('qa_chain.os.getenv')
    @patch('qa_chain.get_retriever')
    @patch('qa_chain.RetrievalQA.from_chain_type')
    def test_create_qa_chain_real_llm(self, mock_retrieval_qa, mock_get_retriever, mock_getenv):
        """Test real LLM creation when API key is valid"""
        mock_retriever = MagicMock()
        mock_get_retriever.return_value = mock_retriever
        mock_getenv.return_value = "sk-valid-api-key"
        mock_chain = MagicMock()
        mock_retrieval_qa.return_value = mock_chain
        
        result = create_qa_chain()
        
        # Should return the RetrievalQA chain
        self.assertEqual(result, mock_chain)
        mock_retrieval_qa.assert_called_once()


if __name__ == "__main__":
    unittest.main()
