import logging
import os
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from database import get_retriever

# Set up logging
logger = logging.getLogger(__name__)

# Create QA chain with retriever
def create_qa_chain(k=5):
    try:
        retriever = get_retriever(k)
        
        # Check if we have a valid API key
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key == "dummy-key-for-development":
            # Create a simple mock chain for development without API key
            logger.warning("Using mock LLM as OpenAI API key is not available")
            
            def mock_generate_answer(inputs):
                query = inputs.get("query", "")
                return {
                    "result": f"This is a mock answer to: '{query}'. Please set OPENAI_API_KEY to get real answers.",
                    "source_documents": retriever.get_relevant_documents(query)
                }
            
            return mock_generate_answer
        
        # Create a real chain with OpenAI
        return RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0),
            chain_type="stuff",  # Explicitly specify chain type
            retriever=retriever,
            return_source_documents=True,
        )
    except Exception as e:
        logger.error(f"Failed to create QA chain: {str(e)}")
        
        # Store the error message to use in the closure
        error_message = str(e)
        
        # Return a fallback function that explains the error
        def error_chain(inputs):
            return {
                "result": f"Error creating QA chain: {error_message}. Please check your configuration.",
                "source_documents": []
            }
        
        return error_chain
