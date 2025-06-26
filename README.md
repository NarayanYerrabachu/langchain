# LangChain RAG API

A modular FastAPI application for Retrieval-Augmented Generation (RAG) using LangChain and PostgreSQL with pgvector.

## Features

- Document ingestion with automatic chunking
- Vector storage in PostgreSQL using pgvector
- Question answering with source attribution
- Modular architecture for easy extension

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL with pgvector extension
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd langchain
```

2. Install dependencies:
```bash
pip install -e .
```

3. Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your-api-key
```

### Running the API

Start the FastAPI server:
```bash
uvicorn app:app --reload
```

The API will be available at http://localhost:8000

## API Endpoints

### Health Check
- `GET /` - Root endpoint
- `GET /health` - Health check endpoint

### Document Management
- `POST /ingest` - Ingest a document
- `POST /query` - Query the document database

### Collection Management
- `GET /collections/info` - Get collection information
- `DELETE /collections/clear` - Clear the collection

## Project Structure

```
langchain/
├── app.py                # FastAPI application entry point
├── config.py             # Configuration settings
├── database.py           # Database and vector store setup
├── models.py             # Pydantic models
├── qa_chain.py           # QA chain setup
├── routes/               # API routes
│   ├── __init__.py
│   ├── collections.py    # Collection management endpoints
│   ├── documents.py      # Document management endpoints
│   └── health.py         # Health check endpoints
└── pyproject.toml        # Project metadata and dependencies
```

## License

MIT
