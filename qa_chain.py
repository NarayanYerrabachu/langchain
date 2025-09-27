import logging
import os
from typing import Dict, Any, List
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.schema import BaseRetriever

from database import get_retriever
from config import OPENAI_API_KEY

# Set up logging
logger = logging.getLogger(__name__)

# Custom prompt template for better RAG responses
RAG_PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

Context:
{context}

Question: {question}

Helpful Answer:"""

RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE, input_variables=["context", "question"]
)


class ErrorQAChain:
    """Fallback QA chain that returns error messages"""

    def __init__(self, error_message: str):
        self.error_message = error_message

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "result": f"Error creating QA chain: {self.error_message}. Please check your configuration.",
            "source_documents": []
        }

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self.invoke(inputs)


class MockLLM:
    """Mock LLM for development/testing when OpenAI API is not available"""

    def __init__(self):
        self.temperature = 0

    def __call__(self, prompt: str, **kwargs) -> str:
        """Generate mock response"""
        return f"This is a mock answer. OpenAI API key is not available. Query was about: {prompt[:100]}..."

    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock invoke method for compatibility"""
        if isinstance(input_data, str):
            query = input_data
        else:
            query = input_data.get("query", str(input_data))

        return {
            "text": f"Mock response for: {query}. Please set OPENAI_API_KEY for real answers."
        }

    """Mock LLM for development/testing when OpenAI API is not available"""

    def __init__(self):
        self.temperature = 0

    def __call__(self, prompt: str, **kwargs) -> str:
        """Generate mock response"""
        return f"This is a mock answer. OpenAI API key is not available. Query was about: {prompt[:100]}..."

    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock invoke method for compatibility"""
        if isinstance(input_data, str):
            query = input_data
        else:
            query = input_data.get("query", str(input_data))

        return {
            "text": f"Mock response for: {query}. Please set OPENAI_API_KEY for real answers."
        }


class MockQAChain:
    """Mock QA chain for development/testing"""

    def __init__(self, retriever: BaseRetriever, k: int = 5):
        self.retriever = retriever
        self.k = k

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Mock invoke method that simulates RAG pipeline"""
        query = inputs.get("query", "")

        # Get relevant documents
        try:
            source_docs = self.retriever.get_relevant_documents(query)
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            source_docs = [Document(
                page_content=f"Error retrieving documents for query: {query}",
                metadata={"error": str(e)}
            )]

        # Generate mock answer
        if source_docs and any(doc.page_content.strip() for doc in source_docs):
            context_preview = source_docs[0].page_content[:200] + "..." if len(source_docs[0].page_content) > 200 else \
            source_docs[0].page_content
            answer = f"Based on the available context, here's a mock answer for '{query}': {context_preview}. Please set OPENAI_API_KEY for real AI-powered answers."
        else:
            answer = f"No relevant context found for query: '{query}'. This is a mock response - please set OPENAI_API_KEY for real answers."

        return {
            "result": answer,
            "source_documents": source_docs
        }

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Support callable interface"""
        return self.invoke(inputs)


def create_llm():
    """Create LLM instance with fallback to mock"""
    if OPENAI_API_KEY and OPENAI_API_KEY != "dummy-key-for-development":
        try:
            # Use ChatOpenAI for better performance
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                max_tokens=500
            )
        except Exception as e:
            logger.error(f"Failed to create OpenAI LLM: {str(e)}")
            return MockLLM()
    else:
        logger.warning("Using mock LLM - OpenAI API key not available")
        return MockLLM()


def create_qa_chain(k: int = 5):
    """Create QA chain with retriever"""
    try:
        retriever = get_retriever(k)

        # Check if we have a valid API key
        if OPENAI_API_KEY and OPENAI_API_KEY != "dummy-key-for-development":
            try:
                llm = create_llm()

                # Create RetrievalQA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": RAG_PROMPT}
                )

                logger.info("Created real QA chain with OpenAI LLM")
                return qa_chain

            except Exception as e:
                logger.error(f"Failed to create real QA chain: {str(e)}")
                return MockQAChain(retriever, k)
        else:
            logger.warning("Using mock QA chain - OpenAI API key not available")
            return MockQAChain(retriever, k)

    except Exception as e:
        logger.error(f"Failed to create QA chain: {str(e)}")

        # Return a fallback function that explains the error
        def error_chain(inputs: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "result": f"Error creating QA chain: . Please check your configuration.",
                "source_documents": []
            }

        return error_chain


def run_qa_chain_test():
    """Test the QA chain functionality"""
    try:
        qa_chain = create_qa_chain()
        result = qa_chain.invoke({"query": "What is this system about?"})

        logger.info("QA Chain test successful")
        return {
            "status": "success",
            "answer": result.get("result", "No result"),
            "source_count": len(result.get("source_documents", []))
        }
    except Exception as e:
        logger.error(f"QA Chain test failed: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }