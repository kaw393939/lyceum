"""
Gutenberg Content Generation System - Content API Routes
=======================================================
Contains routes related to content generation and management.
"""

from fastapi import APIRouter, Request, Response, HTTPException, Depends, BackgroundTasks, status, Body
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import uuid
import time
import asyncio
from datetime import datetime

from models.content import ContentRequest, ContentStatus, ContentResponse, ContentType, ContentDifficulty
from services.content_service import ContentService
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter()

async def get_content_service(request: Request) -> ContentService:
    """Dependency to get content service from app state."""
    return request.app.state.content_service

# Define routes
@router.post(
    "/generate", 
    summary="Generate new educational content", 
    tags=["content"], 
    status_code=status.HTTP_202_ACCEPTED,
    description="""
    Generate new educational content based on the provided parameters.
    
    The content generation process runs asynchronously in the background. This endpoint
    returns immediately with a request ID that you can use to check the status and 
    retrieve the completed content.
    
    ## Process Flow
    
    1. Submit content generation request with required parameters
    2. Receive a request ID for tracking
    3. Use the `/content/status/{request_id}` endpoint to check progress
    4. When status is "completed", retrieve content with `/content/{content_id}`
    
    ## Content Types
    
    - `LESSON`: Complete structured lesson (takes ~2 minutes)
    - `CONCEPT_EXPLANATION`: Focused explanation of a concept (takes ~1.5 minutes)
    - `EXERCISE`: Practice activities or problems (takes ~1 minute)
    - `ASSESSMENT`: Quiz or test questions (takes ~1 minute)
    - `SUMMARY`: Concise overview of a topic (takes ~30 seconds)
    
    ## Media Generation
    
    Set `include_media=true` to generate supporting media along with the content.
    """,
    response_description="Accepted status with request ID for tracking",
    responses={
        202: {
            "description": "Content generation request accepted",
            "content": {
                "application/json": {
                    "example": {
                        "request_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
                        "status": "accepted",
                        "message": "Content generation started",
                        "estimated_completion_time": 1709683712
                    }
                }
            }
        },
        400: {
            "description": "Invalid request parameters",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid concept ID or content type"}
                }
            }
        },
        500: {
            "description": "Server error",
            "content": {
                "application/json": {
                    "example": {"detail": "Failed to initialize content generation"}
                }
            }
        }
    }
)
async def generate_content(
    request_data: ContentRequest = Body(
        ...,
        description="Parameters for content generation",
        example={
            "content_type": "LESSON",
            "concept_id": "stoicism-virtues",
            "difficulty": "INTERMEDIATE",
            "target_audience": "high_school",
            "include_media": True,
            "template_id": "default",
            "max_length": 2000,
            "style": "conversational"
        }
    ),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    content_service: ContentService = Depends(get_content_service)
) -> Dict[str, Any]:
    """
    Generate new educational content asynchronously.
    
    This endpoint initiates the content generation process and returns a request ID
    that can be used to track the status and retrieve the completed content.
    """
    # Create a request ID
    request_id = str(uuid.uuid4())
    
    # Initialize status in database
    request_data_dict = request_data.dict()
    await content_service.create_content_request(request_id=request_id, request_data=request_data_dict)
    
    # Create initial status
    await content_service.update_content_status(
        request_id=request_id,
        status=ContentStatus.PENDING,
        message="Content generation queued",
        progress=0.0
    )
    
    # Add to background tasks
    background_tasks.add_task(
        content_service.generate_content_task,
        request_id=request_id,
        request_data=request_data
    )
    
    # Estimate completion time based on content type complexity
    completion_estimate = 60  # Default 60 seconds
    if request_data.content_type == ContentType.LESSON:
        completion_estimate = 120
    elif request_data.content_type == ContentType.CONCEPT_EXPLANATION:
        completion_estimate = 90
    
    return {
        "request_id": request_id,
        "status": "accepted",
        "message": "Content generation started",
        "estimated_completion_time": int(time.time()) + completion_estimate
    }

@router.get("/{content_id}", summary="Get generated content", tags=["content"])
async def get_content(
    content_id: str,
    content_service: ContentService = Depends(get_content_service)
) -> Dict[str, Any]:
    """
    Retrieve previously generated content by its ID.
    """
    if not content_id or content_id == "undefined":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid content ID"
        )
    
    # Retrieve content from database
    content = await content_service.get_content(content_id)
    
    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Content not found"
        )
    
    # For security, convert any ObjectId to string
    if isinstance(content, dict) and "_id" in content:
        content["_id"] = str(content["_id"])
    
    return content

@router.get("/status/{request_id}", summary="Check generation status", tags=["content"])
async def get_generation_status(
    request_id: str,
    content_service: ContentService = Depends(get_content_service)
) -> Dict[str, Any]:
    """
    Check the status of a content generation request.
    """
    if not request_id or request_id == "undefined":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid request ID"
        )
    
    # Get status from the database
    status_data = await content_service.get_content_status(request_id)
    
    if not status_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Request not found"
        )
    
    # For security, convert any ObjectId to string
    if isinstance(status_data, dict) and "_id" in status_data:
        status_data["_id"] = str(status_data["_id"])
        
    return status_data

@router.delete("/{content_id}", summary="Delete generated content", tags=["content"])
async def delete_content(
    content_id: str,
    content_service: ContentService = Depends(get_content_service)
) -> Dict[str, Any]:
    """
    Delete previously generated content by its ID.
    """
    if not content_id or content_id == "undefined":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid content ID"
        )
    
    # Check if content exists first
    exists = await content_service.check_content_exists(content_id)
    if not exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Content not found"
        )
    
    # Delete the content
    success = await content_service.delete_content(content_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete content"
        )
    
    return {
        "status": "success",
        "message": f"Content {content_id} deleted successfully"
    }
    
@router.patch("/{content_id}", summary="Update content properties", tags=["content"])
async def update_content(
    content_id: str,
    update_data: Dict[str, Any],
    content_service: ContentService = Depends(get_content_service)
) -> Dict[str, Any]:
    """
    Update properties of existing content.
    
    This endpoint allows updating metadata, status, and other properties
    without regenerating the entire content.
    """
    if not content_id or content_id == "undefined":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid content ID"
        )
    
    # Check if content exists first
    exists = await content_service.check_content_exists(content_id)
    if not exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Content not found"
        )
    
    # Update the content
    updated_content = await content_service.update_content(content_id, update_data)
    
    if not updated_content:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update content"
        )
    
    return updated_content

@router.get("/", summary="List all generated content", tags=["content"])
async def list_content(
    limit: int = 10,
    offset: int = 0,
    status: Optional[str] = None,
    content_type: Optional[str] = None,
    content_service: ContentService = Depends(get_content_service)
) -> Dict[str, Any]:
    """
    List all generated content with pagination and filtering.
    
    - **limit**: Maximum number of items to return
    - **offset**: Number of items to skip
    - **status**: Filter by content status
    - **content_type**: Filter by content type
    """
    # Build filters from query params
    filters = {}
    if status:
        filters["metadata.status"] = status
    if content_type:
        filters["metadata.content_type"] = content_type
    
    # Get content list with pagination
    items, total = await content_service.list_content(
        limit=limit,
        offset=offset,
        filters=filters
    )
    
    # For security, convert any ObjectId to string
    for item in items:
        if "_id" in item:
            item["_id"] = str(item["_id"])
    
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": (offset + limit) < total,
        "filters_applied": filters
    }