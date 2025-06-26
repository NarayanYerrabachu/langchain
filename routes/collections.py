from fastapi import APIRouter, HTTPException
from config import COLLECTION_NAME
from database import get_vectorstore

router = APIRouter(prefix="/collections")

@router.get("/info")
def get_collection_info():
    """Get information about the current collection"""
    try:
        # This is a basic implementation - you might want to add more detailed stats
        return {
            "collection_name": COLLECTION_NAME,
            "status": "active"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get collection info: {str(e)}")


@router.delete("/clear")
def clear_collection():
    """Clear all documents from the collection (use with caution!)"""
    try:
        # Note: This is a destructive operation
        # You might want to implement proper authentication/authorization
        vectorstore = get_vectorstore()
        vectorstore.delete_collection()
        return {"status": "Collection cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear collection: {str(e)}")
