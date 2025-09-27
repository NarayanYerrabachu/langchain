import unittest
from unittest.mock import patch, MagicMock

from qa_chain import create_qa_chain, create_llm, run_qa_chain_test, MockQAChain, MockLLM


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

    @patch('qa_chain.OPENAI_API_KEY', 'dummy-key-for-development')
    @patch('qa_chain.get_retriever')
    def test_create_qa_chain_mock_implementation(self, mock_get_retriever):
        """Test mock QA chain creation when API key is dummy"""
        mock_retriever = MagicMock()
        mock_get_retriever.return_value = mock_retriever

        result = create_qa_chain()
        self.assertIsInstance(result, MockQAChain)

    @patch('qa_chain.OPENAI_API_KEY', 'sk-valid-api-key')
    @patch('qa_chain.get_retriever')
    @patch('qa_chain.RetrievalQA.from_chain_type')
    @patch('qa_chain.create_llm')
    def test_create_qa_chain_real_llm(self, mock_create_llm, mock_retrieval_qa, mock_get_retriever):
        """Test real LLM creation when API key is valid"""
        mock_retriever = MagicMock()
        mock_get_retriever.return_value = mock_retriever
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm
        mock_chain = MagicMock()
        mock_retrieval_qa.return_value = mock_chain

        result = create_qa_chain()

        # Should return the RetrievalQA chain
        self.assertEqual(result, mock_chain)
        mock_retrieval_qa.assert_called_once()
        mock_create_llm.assert_called_once()

    @patch('qa_chain.OPENAI_API_KEY', 'sk-valid-api-key')
    @patch('qa_chain.ChatOpenAI')
    def test_create_llm_real(self, mock_chat_openai):
        """Test creating real LLM with valid API key"""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm

        result = create_llm()

        self.assertEqual(result, mock_llm)
        mock_chat_openai.assert_called_once_with(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=500
        )

    @patch('qa_chain.OPENAI_API_KEY', 'dummy-key-for-development')
    def test_create_llm_mock(self):
        """Test creating mock LLM when API key is dummy"""
        result = create_llm()

        self.assertIsInstance(result, MockLLM)

    @patch('qa_chain.OPENAI_API_KEY', None)
    def test_create_llm_no_key(self):
        """Test creating mock LLM when no API key"""
        result = create_llm()

        self.assertIsInstance(result, MockLLM)

    def test_mock_llm_functionality(self):
        """Test MockLLM functionality"""
        mock_llm = MockLLM()

        # Test __call__ method
        result = mock_llm("test prompt")
        self.assertIsInstance(result, str)
        self.assertIn("mock answer", result.lower())

        # Test invoke method
        result = mock_llm.invoke({"query": "test question"})
        self.assertIn("text", result)
        self.assertIn("test question", result["text"])

    @patch('qa_chain.get_retriever')
    def test_mock_qa_chain_functionality(self, mock_get_retriever):
        """Test MockQAChain functionality"""
        mock_retriever = MagicMock()
        from langchain_core.documents import Document
        mock_source_doc = Document(
            page_content="Test content",
            metadata={"source": "test"}
        )
        mock_retriever.get_relevant_documents.return_value = [mock_source_doc]
        mock_get_retriever.return_value = mock_retriever

        mock_qa_chain = MockQAChain(mock_retriever, k=3)

        # Test invoke method
        result = mock_qa_chain.invoke({"query": "test question"})

        self.assertIn("result", result)
        self.assertIn("source_documents", result)
        self.assertIsInstance(result["result"], str)
        self.assertEqual(len(result["source_documents"]), 1)

        # Test __call__ method
        result2 = mock_qa_chain({"query": "another test"})
        self.assertIn("result", result2)
        self.assertIn("source_documents", result2)

    @patch('qa_chain.create_qa_chain')
    def test_run_qa_chain_test_success(self, mock_create_qa_chain):
        """Test the run_qa_chain_test function with successful result"""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = {
            "result": "Test result",
            "source_documents": [MagicMock(), MagicMock()]
        }
        mock_create_qa_chain.return_value = mock_chain

        result = run_qa_chain_test()

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["answer"], "Test result")
        self.assertEqual(result["source_count"], 2)

    @patch('qa_chain.create_qa_chain')
    def test_run_qa_chain_test_error(self, mock_create_qa_chain):
        """Test the run_qa_chain_test function with error"""
        mock_create_qa_chain.side_effect = Exception("Test error")

        result = run_qa_chain_test()

        self.assertEqual(result["status"], "error")
        self.assertIn("message", result)
        self.assertIn("Test error", result["message"])


if __name__ == "__main__":
    unittest.main()