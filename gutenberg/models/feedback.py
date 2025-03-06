"""
Feedback Data Models
==================
Models for user feedback and content improvement.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class FeedbackType(str, Enum):
    """Types of feedback that can be provided."""
    CONTENT_QUALITY = "content_quality"
    ACCURACY = "accuracy"
    ENGAGEMENT = "engagement"
    CLARITY = "clarity"
    DIFFICULTY = "difficulty"
    TECHNICAL = "technical"
    SUGGESTION = "suggestion"
    GENERAL = "general"


class FeedbackStatus(str, Enum):
    """Status of feedback processing."""
    RECEIVED = "received"
    UNDER_REVIEW = "under_review"
    IMPLEMENTED = "implemented"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class FeedbackSeverity(str, Enum):
    """Severity level of feedback."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ContentRating(BaseModel):
    """Rating for various aspects of content."""
    overall: Optional[int] = Field(None, description="Overall rating (1-5)")
    accuracy: Optional[int] = Field(None, description="Accuracy rating (1-5)")
    clarity: Optional[int] = Field(None, description="Clarity rating (1-5)")
    engagement: Optional[int] = Field(None, description="Engagement rating (1-5)")
    relevance: Optional[int] = Field(None, description="Relevance rating (1-5)")
    difficulty: Optional[int] = Field(None, description="Appropriateness of difficulty (1-5)")


class FeedbackItem(BaseModel):
    """A specific item of feedback."""
    item_id: str = Field(..., description="Unique ID for the feedback item")
    feedback_type: FeedbackType = Field(..., description="Type of feedback")
    description: str = Field(..., description="Detailed description of the feedback")
    severity: FeedbackSeverity = Field(FeedbackSeverity.MEDIUM, description="Severity level")
    section_reference: Optional[str] = Field(None, description="Reference to specific content section")
    suggested_improvement: Optional[str] = Field(None, description="Suggested improvement")


class Feedback(BaseModel):
    """Complete feedback for a piece of content."""
    feedback_id: str = Field(..., description="Unique ID for the feedback")
    content_id: str = Field(..., description="ID of the content being rated")
    user_id: Optional[str] = Field(None, description="ID of the user providing feedback")
    created_at: datetime = Field(default_factory=datetime.now, description="Feedback timestamp")
    ratings: ContentRating = Field(..., description="Numerical ratings")
    feedback_items: List[FeedbackItem] = Field(default_factory=list, description="Specific feedback items")
    general_comments: Optional[str] = Field(None, description="General comments")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    status: FeedbackStatus = Field(FeedbackStatus.RECEIVED, description="Status of feedback")


class FeedbackResponse(BaseModel):
    """Response to feedback."""
    response_id: str = Field(..., description="Unique ID for the response")
    feedback_id: str = Field(..., description="ID of the feedback being responded to")
    responder_id: str = Field(..., description="ID of the person/system responding")
    created_at: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    response_text: str = Field(..., description="Response text")
    action_taken: Optional[str] = Field(None, description="Action taken in response to feedback")
    new_content_id: Optional[str] = Field(None, description="ID of new content if generated")
    status: FeedbackStatus = Field(..., description="Updated status of feedback")


class FeedbackSummary(BaseModel):
    """Summary of feedback for content."""
    content_id: str = Field(..., description="ID of the content")
    feedback_count: int = Field(..., description="Total number of feedback submissions")
    average_ratings: ContentRating = Field(..., description="Average ratings")
    common_issues: List[Dict[str, Any]] = Field(default_factory=list, description="Common issues identified")
    positive_aspects: List[Dict[str, Any]] = Field(default_factory=list, description="Positive aspects identified")
    improvement_recommendations: List[str] = Field(default_factory=list, description="Recommended improvements")
    generated_at: datetime = Field(default_factory=datetime.now, description="Summary generation timestamp")