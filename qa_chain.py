from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

from database import get_retriever

# Create QA chain with retriever
def create_qa_chain(k=5):
    retriever = get_retriever(k)
    
    return RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0, openai_api_key=""),
        chain_type="stuff",  # Explicitly specify chain type
        retriever=retriever,
        return_source_documents=True,
    )
