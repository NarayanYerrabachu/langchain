#!/usr/bin/env python3
"""
Example usage script for the RAG Document Q&A API

This script demonstrates how to:
1. Check API health
2. Ingest different types of documents
3. Query the knowledge base
4. Handle API responses

Usage:
    python example_usage.py
"""

import requests
import json
import time
from pathlib import Path

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_V1_BASE = f"{API_BASE_URL}/api/v1"


class RAGAPIClient:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.v1_base = f"{base_url}/api/v1"

    def health_check(self):
        """Check API health"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {"error": str(e)}

    def ingest_text(self, content: str, metadata: dict = None, chunk_size: int = 1000):
        """Ingest plain text content"""
        data = {
            "content": content,
            "document_type": "text",
            "metadata": metadata or {},
            "chunk_size": chunk_size,
            "chunk_overlap": 200
        }

        response = requests.post(
            f"{self.v1_base}/ingest",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        return response.json()

    def ingest_url(self, url: str, metadata: dict = None):
        """Ingest content from a web URL"""
        data = {
            "url": url,
            "document_type": "web_url",
            "metadata": metadata or {},
            "chunk_size": 1000,
            "chunk_overlap": 200
        }

        response = requests.post(
            f"{self.v1_base}/ingest",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        return response.json()

    def ingest_file(self, file_path: str, document_type: str, metadata: dict = None):
        """Ingest a file upload"""
        files = {"file": open(file_path, "rb")}
        data = {
            "document_type": document_type,
            "metadata": json.dumps(metadata or {}),
            "chunk_size": 1000,
            "chunk_overlap": 200
        }

        try:
            response = requests.post(
                f"{self.v1_base}/ingest/file",
                files=files,
                data=data
            )
            return response.json()
        finally:
            files["file"].close()

    def query(self, question: str, max_results: int = 5, include_metadata: bool = True):
        """Query the knowledge base"""
        data = {
            "query": question,
            "max_results": max_results,
            "include_metadata": include_metadata
        }

        response = requests.post(
            f"{self.v1_base}/query",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        return response.json()


def main():
    """Example usage of the RAG API"""
    client = RAGAPIClient()

    print("🚀 RAG Document Q&A API Example Usage")
    print("=" * 50)

    # 1. Health Check
    print("\n📊 Checking API Health...")
    health = client.health_check()
    print(f"Status: {health.get('status', 'unknown')}")

    if health.get('status') != 'healthy' and health.get('status') != 'degraded':
        print("❌ API is not healthy. Please check the service.")
        return

    # 2. Ingest Text Content
    print("\n📄 Ingesting text content...")

    sample_text = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create 
    intelligent machines that work and react like humans. Some of the activities 
    computers with artificial intelligence are designed for include:

    - Speech recognition
    - Learning
    - Planning
    - Problem solving

    Machine Learning is a subset of AI that provides systems the ability to automatically 
    learn and improve from experience without being explicitly programmed. Machine 
    learning focuses on the development of computer programs that can access data and 
    use it to learn for themselves.

    Deep Learning is a subset of machine learning that makes the computation of 
    multi-layer neural networks feasible. It uses multiple layers to progressively 
    extract higher-level features from the raw input.
    """

    result = client.ingest_text(
        content=sample_text,
        metadata={
            "title": "Introduction to AI and Machine Learning",
            "author": "Example Author",
            "topic": "artificial_intelligence"
        }
    )
    print(f"✅ Text ingested: {result.get('message', 'Success')}")
    print(f"   Document ID: {result.get('document_id', 'N/A')}")
    print(f"   Chunks created: {result.get('document_count', 0)}")

    # 3. Ingest from URL (example - this might fail if URL is not accessible)
    print("\n🌐 Ingesting content from URL...")
    try:
        url_result = client.ingest_url(
            url="https://en.wikipedia.org/wiki/Machine_learning",
            metadata={
                "source": "wikipedia",
                "topic": "machine_learning"
            }
        )
        print(f"✅ URL content ingested: {url_result.get('message', 'Success')}")
        print(f"   Chunks created: {url_result.get('document_count', 0)}")
    except Exception as e:
        print(f"⚠️ URL ingestion failed (this is normal in demo): {str(e)}")

    # 4. Wait a moment for indexing
    print("\n⏳ Waiting for document indexing...")
    time.sleep(2)

    # 5. Query the Knowledge Base
    print("\n❓ Querying the knowledge base...")

    questions = [
        "What is artificial intelligence?",
        "What are the main activities of AI systems?",
        "How does machine learning differ from AI?",
        "What is deep learning?",
        "Tell me about neural networks"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n🔍 Question {i}: {question}")

        try:
            answer_result = client.query(
                question=question,
                max_results=3,
                include_metadata=True
            )

            print(f"📝 Answer: {answer_result.get('answer', 'No answer found')}")
            print(f"⏱️  Query time: {answer_result.get('query_time', 'N/A')} seconds")
            print(f"📚 Sources found: {answer_result.get('source_count', 0)}")

            # Show source previews
            sources = answer_result.get('sources', [])
            for j, source in enumerate(sources[:2]):  # Show first 2 sources
                content = source.get('content', '')[:100] + "..."
                metadata = source.get('metadata', {})
                doc_id = metadata.get('document_id', 'N/A')[:8]
                print(f"   📖 Source {j + 1} (ID: {doc_id}): {content}")

        except Exception as e:
            print(f"❌ Query failed: {str(e)}")

        if i < len(questions):
            time.sleep(1)  # Brief pause between queries

    # 6. File Upload Example (if you have a sample file)
    print("\n📁 File upload example...")
    sample_files = ["sample.pdf", "sample.docx", "sample.txt", "document.pdf"]

    for file_name in sample_files:
        if Path(file_name).exists():
            print(f"📤 Uploading {file_name}...")
            try:
                # Determine document type from extension
                ext = Path(file_name).suffix.lower()
                doc_type_map = {
                    '.pdf': 'pdf',
                    '.docx': 'docx',
                    '.txt': 'text',
                    '.html': 'html',
                    '.md': 'markdown'
                }
                doc_type = doc_type_map.get(ext, 'text')

                file_result = client.ingest_file(
                    file_path=file_name,
                    document_type=doc_type,
                    metadata={"uploaded_by": "example_script", "filename": file_name}
                )
                print(f"✅ File uploaded: {file_result.get('message', 'Success')}")
                break
            except Exception as e:
                print(f"⚠️ File upload failed: {str(e)}")
    else:
        print("ℹ️  No sample files found. Create a 'sample.pdf' or 'sample.txt' to test file upload.")

    print("\n✨ Example completed!")
    print("\n💡 Next steps:")
    print("   - Try uploading your own documents")
    print("   - Experiment with different chunk sizes")
    print("   - Use the web interface at http://localhost:8000/docs")
    print("   - Check health status at http://localhost:8000/health")


if __name__ == "__main__":
    main()