"""
Gutenberg Content Generation System - Feedback API Routes
========================================================
Contains routes related to feedback submission and management.
"""

from fastapi import APIRouter, Request, Response, HTTPException, Depends, Query, status
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime

from models.feedback import Feedback, FeedbackResponse, FeedbackSummary, FeedbackType, FeedbackStatus, FeedbackSeverity
from services.content_service import ContentService
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter()

async def get_content_service(request: Request) -> ContentService:
    """Dependency to get content service from app state."""
    return request.app.state.content_service

# Define routes
@router.post("/", summary="Submit feedback", tags=["feedback"], status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    feedback: Feedback,
    content_service: ContentService = Depends(get_content_service)
) -> Dict[str, Any]:
    """
    Submit feedback for generated content.
    """
    # Process the feedback
    result = await content_service.submit_feedback(feedback)
    
    return {
        "feedback_id": result["feedback_id"],
        "message": "Feedback submitted successfully",
        "needs_improvement": result.get("needs_improvement", False)
    }

@router.get("/{feedback_id}", summary="Get feedback details", tags=["feedback"])
async def get_feedback(
    feedback_id: str,
    content_service: ContentService = Depends(get_content_service)
) -> Dict[str, Any]:
    """
    Get details of a specific feedback submission.
    """
    if not feedback_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid feedback ID"
        )
    
    feedback = await content_service.get_feedback(feedback_id)
    
    if not feedback:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feedback not found"
        )
    
    return feedback

@router.get("/content/{content_id}", summary="Get feedback for content", tags=["feedback"])
async def get_content_feedback(
    content_id: str,
    content_service: ContentService = Depends(get_content_service)
) -> List[Dict[str, Any]]:
    """
    Get all feedback for a specific content item.
    """
    if not content_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid content ID"
        )
    
    # Check if content exists
    content_exists = await content_service.check_content_exists(content_id)
    if not content_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Content not found"
        )
    
    # Get all feedback for the content
    feedback_list = await content_service.get_content_feedback(content_id)
    
    return feedback_list

@router.post("/{feedback_id}/respond", summary="Respond to feedback", tags=["feedback"])
async def respond_to_feedback(
    feedback_id: str,
    response: FeedbackResponse,
    content_service: ContentService = Depends(get_content_service)
) -> Dict[str, Any]:
    """
    Respond to submitted feedback and update its status.
    """
    if not feedback_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid feedback ID"
        )
    
    # Check if feedback exists
    feedback = await content_service.get_feedback(feedback_id)
    if not feedback:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feedback not found"
        )
    
    # Create response
    result = await content_service.create_feedback_response(response)
    
    return {
        "response_id": result["response_id"],
        "message": "Response submitted successfully",
        "status": response.status
    }

@router.get("/summary/{content_id}", summary="Get feedback summary", tags=["feedback"])
async def get_feedback_summary(
    content_id: str,
    content_service: ContentService = Depends(get_content_service)
) -> FeedbackSummary:
    """
    Get a summary of all feedback for a content item.
    """
    if not content_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid content ID"
        )
    
    # Check if content exists
    content_exists = await content_service.check_content_exists(content_id)
    if not content_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Content not found"
        )
    
    # Generate feedback summary
    summary = await content_service.generate_feedback_summary(content_id)
    
    return summary

@router.patch("/{feedback_id}/status", summary="Update feedback status", tags=["feedback"])
async def update_feedback_status(
    feedback_id: str,
    status: FeedbackStatus,
    content_service: ContentService = Depends(get_content_service)
) -> Dict[str, Any]:
    """
    Update the status of a feedback submission.
    """
    if not feedback_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid feedback ID"
        )
    
    # Update feedback status
    result = await content_service.update_feedback_status(feedback_id, status)
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feedback not found"
        )
    
    return {
        "feedback_id": feedback_id,
        "status": status,
        "message": "Feedback status updated successfully"
    }

@router.get("/", summary="List all feedback", tags=["feedback"])
async def list_feedback(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    content_id: Optional[str] = None,
    status: Optional[FeedbackStatus] = None,
    feedback_type: Optional[FeedbackType] = None,
    content_service: ContentService = Depends(get_content_service)
) -> Dict[str, Any]:
    """
    List all feedback with pagination and filtering.
    
    - **limit**: Maximum number of items to return
    - **offset**: Number of items to skip
    - **content_id**: Filter by content ID
    - **status**: Filter by feedback status
    - **feedback_type**: Filter by feedback type
    """
    # Build filters
    filters = {}
    if content_id:
        filters["content_id"] = content_id
    if status:
        filters["status"] = status
    if feedback_type:
        filters["feedback_type"] = feedback_type
    
    # Get feedback list
    items, total = await content_service.list_feedback(
        limit=limit,
        offset=offset,
        filters=filters
    )
    
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": (offset + limit) < total
    }