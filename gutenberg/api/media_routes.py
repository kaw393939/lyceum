"""
Gutenberg Content Generation System - Media API Routes
=====================================================
Contains routes related to media storage, retrieval, and generation.
"""

from fastapi import APIRouter, Request, Response, HTTPException, Depends, UploadFile, File, Form, Query, status
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from typing import Dict, Any, List, Optional
import uuid
import io
import base64
from datetime import datetime

from models.content import MediaType, MediaItem
from core.media_generator import MediaGenerator, MediaRequest
from integrations.mongodb_service import MongoDBService
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter()

async def get_media_generator(request: Request) -> MediaGenerator:
    """Dependency to get media generator from app state."""
    if not hasattr(request.app.state, "media_generator"):
        request.app.state.media_generator = MediaGenerator()
    return request.app.state.media_generator

async def get_mongodb_service(request: Request) -> MongoDBService:
    """Dependency to get MongoDB service from app state."""
    return request.app.state.mongodb_service

# Define routes
@router.post("/generate", summary="Generate media", tags=["media"], status_code=status.HTTP_202_ACCEPTED)
async def generate_media(
    request: MediaRequest,
    media_generator: MediaGenerator = Depends(get_media_generator),
    mongodb: MongoDBService = Depends(get_mongodb_service)
) -> Dict[str, Any]:
    """
    Generate media based on a prompt and media type.
    
    This endpoint initiates media generation and returns an ID for tracking.
    """
    try:
        # Generate media
        media_response = await media_generator.generate_media(request)
        
        # Store metadata in MongoDB
        media_metadata = {
            "media_id": media_response.media_id,
            "title": media_response.title,
            "description": media_response.description,
            "media_type": media_response.media_type.value,
            "alt_text": media_response.alt_text,
            "created_at": datetime.now(),
            "metadata": media_response.metadata
        }
        
        # If we have generated data, store it
        if media_response.data:
            # Determine content type
            content_type = "image/png"  # Default
            if media_response.media_type == MediaType.IMAGE:
                if media_response.metadata.get("format") == "svg+xml":
                    content_type = "image/svg+xml"
                else:
                    content_type = "image/png"
            elif media_response.media_type == MediaType.AUDIO:
                content_type = "audio/mp3"
            elif media_response.media_type == MediaType.VIDEO:
                content_type = "video/mp4"
            
            # Store file in GridFS
            filename = f"{media_response.media_type.value}_{media_response.media_id}.{media_response.metadata.get('format', 'bin')}"
            await mongodb.store_media_file(
                file_data=media_response.data,
                filename=filename,
                content_type=content_type,
                metadata=media_metadata
            )
        
        # Return response
        return {
            "media_id": media_response.media_id,
            "status": "generated",
            "media_type": media_response.media_type.value,
            "title": media_response.title,
            "url": f"/media/{media_response.media_id}" if media_response.data else media_response.url
        }
    except Exception as e:
        logger.error(f"Error generating media: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating media: {str(e)}"
        )

@router.get("/{media_id}", summary="Get media", tags=["media"])
async def get_media(
    media_id: str,
    as_json: bool = Query(False, description="Return media data as JSON instead of binary"),
    mongodb: MongoDBService = Depends(get_mongodb_service)
) -> Any:
    """
    Get media by ID.
    """
    if not media_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid media ID"
        )
    
    # Get media file
    media_file = await mongodb.get_media_file(media_id)
    
    if not media_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media not found"
        )
    
    # If requested as JSON, return all data including base64 encoded file
    if as_json:
        # Convert binary data to base64 if available
        if media_file.get("data"):
            media_file["data_base64"] = base64.b64encode(media_file["data"]).decode("utf-8")
            # Remove binary data from response
            del media_file["data"]
        
        return media_file
    
    # Otherwise stream the binary file
    if media_file.get("data"):
        # Convert to stream
        content_type = media_file.get("content_type", "application/octet-stream")
        filename = media_file.get("filename", f"media_{media_id}")
        
        # Create streaming response
        return StreamingResponse(
            io.BytesIO(media_file["data"]),
            media_type=content_type,
            headers={"Content-Disposition": f"inline; filename={filename}"}
        )
    else:
        # No data available
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media data not available"
        )

@router.delete("/{media_id}", summary="Delete media", tags=["media"])
async def delete_media(
    media_id: str,
    mongodb: MongoDBService = Depends(get_mongodb_service)
) -> Dict[str, Any]:
    """
    Delete media by ID.
    """
    if not media_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid media ID"
        )
    
    # Delete media file
    result = await mongodb.delete_media_file(media_id)
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media not found"
        )
    
    return {
        "media_id": media_id,
        "status": "deleted",
        "message": "Media deleted successfully"
    }

@router.get("/", summary="List media files", tags=["media"])
async def list_media(
    media_type: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    mongodb: MongoDBService = Depends(get_mongodb_service)
) -> Dict[str, Any]:
    """
    List media files with pagination and filtering.
    
    - **media_type**: Filter by media type (e.g., "image", "audio", "video")
    - **limit**: Maximum number of items to return
    - **offset**: Number of items to skip
    """
    # Build filters
    filters = {}
    if media_type:
        filters["metadata.media_type"] = media_type
    
    # Get media list
    media_list, total = await mongodb.list_media_files(
        filters=filters,
        limit=limit,
        offset=offset
    )
    
    return {
        "items": media_list,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": (offset + limit) < total
    }

@router.post("/upload", summary="Upload media file", tags=["media"], status_code=status.HTTP_201_CREATED)
async def upload_media(
    file: UploadFile = File(...),
    title: str = Form(...),
    description: str = Form(...),
    media_type: str = Form(...),
    alt_text: str = Form(...),
    mongodb: MongoDBService = Depends(get_mongodb_service)
) -> Dict[str, Any]:
    """
    Upload a media file.
    """
    try:
        # Read file data
        file_data = await file.read()
        
        # Determine content type
        content_type = file.content_type or "application/octet-stream"
        
        # Build metadata
        media_id = str(uuid.uuid4())
        media_metadata = {
            "media_id": media_id,
            "title": title,
            "description": description,
            "media_type": media_type,
            "alt_text": alt_text,
            "created_at": datetime.now(),
            "metadata": {
                "filename": file.filename,
                "content_type": content_type,
                "size": len(file_data)
            }
        }
        
        # Store file in GridFS
        await mongodb.store_media_file(
            file_data=file_data,
            filename=file.filename,
            content_type=content_type,
            metadata=media_metadata
        )
        
        return {
            "media_id": media_id,
            "status": "uploaded",
            "title": title,
            "media_type": media_type,
            "url": f"/media/{media_id}"
        }
    except Exception as e:
        logger.error(f"Error uploading media: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading media: {str(e)}"
        )