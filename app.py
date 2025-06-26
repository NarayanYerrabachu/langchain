from fastapi import FastAPI
from routes import health, documents, collections

# Create FastAPI application
app = FastAPI(title="RAG Document Q&A API", version="1.0.0")

# Include routers
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(collections.router)
