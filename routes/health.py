from fastapi import APIRouter, HTTPException
from database import engine

router = APIRouter()

@router.get("/")
def root():
    return {"message": "RAG Document Q&A API", "status": "running"}


@router.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database connection failed: {str(e)}")
