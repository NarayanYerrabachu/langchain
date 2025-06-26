from fastapi import APIRouter, HTTPException
from sqlalchemy import text
from database import engine

router = APIRouter()

@router.get("/")
def root():
    return {"message": "RAG Document Q&A API", "status": "running"}


@router.get("/health")
def health_check():
    """Health check endpoint"""
    if engine is None:
        return {
            "status": "degraded",
            "database": "not connected",
            "message": "Application is running but database connection is not available"
        }
    
    try:
        # Test database connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database connection failed: {str(e)}")
