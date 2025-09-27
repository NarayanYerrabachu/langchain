import logging
import os
from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from database import engine, health_check_database
from qa_chain import run_qa_chain_test
from config import OPENAI_API_KEY, ENABLE_DATABASE, COLLECTION_NAME

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/")
def root():
    """Root endpoint with basic API information"""
    return {
        "message": "RAG Document Q&A API",
        "status": "running",
        "version": "2.0.0",
        "features": [
            "Text ingestion",
            "PDF processing",
            "DOCX processing",
            "HTML processing",
            "Web scraping",
            "Markdown processing",
            "CSV processing",
            "Excel processing",
            "File uploads",
            "Vector search",
            "Question answering"
        ]
    }


@router.get("/health")
def health_check():
    """Comprehensive health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": None,
        "services": {},
        "configuration": {}
    }

    import datetime
    health_status["timestamp"] = datetime.datetime.utcnow().isoformat()

    # Check database
    db_health = health_check_database()
    health_status["services"]["database"] = db_health

    # Check OpenAI API
    if OPENAI_API_KEY and OPENAI_API_KEY != "dummy-key-for-development":
        health_status["services"]["openai"] = {
            "status": "configured",
            "message": "OpenAI API key is set"
        }
    else:
        health_status["services"]["openai"] = {
            "status": "mock",
            "message": "OpenAI API key not configured, using mock responses"
        }

    # Test QA chain
    qa_test = run_qa_chain_test()
    health_status["services"]["qa_chain"] = qa_test

    # Configuration info
    health_status["configuration"] = {
        "database_enabled": ENABLE_DATABASE,
        "collection_name": COLLECTION_NAME,
        "openai_configured": bool(OPENAI_API_KEY and OPENAI_API_KEY != "dummy-key-for-development")
    }

    # Determine overall status
    service_statuses = [service.get("status", "unknown") for service in health_status["services"].values()]

    if any(status == "unhealthy" for status in service_statuses):
        health_status["status"] = "unhealthy"
        raise HTTPException(status_code=503, detail=health_status)
    elif any(status in ["degraded", "mock", "unavailable"] for status in service_statuses):
        health_status["status"] = "degraded"

    return health_status


@router.get("/health/simple")
def simple_health_check():
    """Simple health check for load balancers"""
    try:
        if ENABLE_DATABASE and engine is not None:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))

        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail={"status": "error", "message": str(e)})


@router.get("/health/database")
def database_health():
    """Detailed database health check"""
    return health_check_database()


@router.get("/health/services")
def services_health():
    """Check health of individual services"""
    services = {}

    # Database
    services["database"] = health_check_database()

    # OpenAI
    if OPENAI_API_KEY and OPENAI_API_KEY != "dummy-key-for-development":
        services["openai"] = {"status": "available", "configured": True}
    else:
        services["openai"] = {"status": "unavailable", "configured": False}

    # QA Chain
    services["qa_chain"] = run_qa_chain_test()

    return {"services": services}


@router.get("/info")
def system_info():
    """Get system information"""
    return {
        "api_version": "2.0.0",
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "configuration": {
            "database_enabled": ENABLE_DATABASE,
            "collection_name": COLLECTION_NAME,
            "openai_configured": bool(OPENAI_API_KEY and OPENAI_API_KEY != "dummy-key-for-development")
        },
        "supported_formats": [
            "text/plain",
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/html",
            "text/markdown",
            "text/csv",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "web_urls"
        ]
    }