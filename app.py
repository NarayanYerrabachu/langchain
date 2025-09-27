import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import time

from routes import health, documents, collections
from config import MAX_FILE_SIZE

# Set up logging
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="RAG Document Q&A API",
    version="2.0.0",
    description="""
    A comprehensive RAG (Retrieval-Augmented Generation) API that supports:

    - Multiple document formats (PDF, DOCX, HTML, Markdown, CSV, Excel)
    - Web page scraping and processing
    - File uploads with automatic format detection
    - Vector similarity search using PGVector
    - OpenAI-powered question answering
    - Configurable text chunking strategies

    ## Supported Document Types

    - **Text**: Plain text content
    - **PDF**: Portable Document Format files
    - **DOCX**: Microsoft Word documents
    - **HTML**: Web pages and HTML files
    - **Web URLs**: Automatic web scraping
    - **Markdown**: Markdown formatted text
    - **CSV**: Comma-separated values
    - **Excel**: Microsoft Excel spreadsheets
    """,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add request size limit middleware
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    """Limit file upload size"""
    if request.url.path.startswith("/ingest/file"):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_FILE_SIZE:
            return JSONResponse(
                status_code=413,
                content={"detail": f"File too large. Maximum size: {MAX_FILE_SIZE // 1024 // 1024}MB"}
            )

    response = await call_next(request)
    return response


# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time, 4))
    return response


# Global exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "type": "http_error"
            }
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    logger.error(f"Validation Error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": 422,
                "message": "Request validation failed",
                "type": "validation_error",
                "details": exc.errors()
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled Exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "type": "server_error"
            }
        }
    )


# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])
app.include_router(collections.router, prefix="/api/v1", tags=["Collections"])


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting RAG Document Q&A API v2.0.0")

    # Import database module to trigger initialization
    from database import init_database

    logger.info("API startup complete")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down RAG Document Q&A API")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )