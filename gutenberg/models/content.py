"""
Content Data Models
==================
Models for content generation requests and responses.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class ContentStatus(str, Enum):
    """Status of content generation requests."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class ContentType(str, Enum):
    """Types of content that can be generated."""
    LESSON = "lesson"
    CONCEPT_EXPLANATION = "concept_explanation"
    EXERCISE = "exercise"
    ASSESSMENT = "assessment"
    SUMMARY = "summary"
    RELATED_CONCEPT = "related_concept"
    PREREQUISITE = "prerequisite"


class ContentDifficulty(str, Enum):
    """Difficulty levels for content."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    
    @classmethod
    def from_learner_level(cls, level: str) -> "ContentDifficulty":
        """Convert a learner level string to a ContentDifficulty enum."""
        mapping = {
            "beginner": cls.BEGINNER,
            "intermediate": cls.INTERMEDIATE,
            "advanced": cls.ADVANCED,
            "expert": cls.EXPERT
        }
        return mapping.get(level.lower(), cls.INTERMEDIATE)


class MediaType(str, Enum):
    """Types of media that can be generated."""
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DIAGRAM = "diagram"
    CHART = "chart"
    ILLUSTRATION = "illustration"
    INFOGRAPHIC = "infographic"
    ANIMATION = "animation"
    INTERACTIVE = "interactive"
    DOCUMENT = "document"


class TemplateVariable(str, Enum):
    """Template variable types for substitution."""
    CONCEPT_NAME = "concept_name"
    CONCEPT_DESCRIPTION = "concept_description"
    RELATED_CONCEPTS = "related_concepts"
    EXAMPLES = "examples"
    AGE_RANGE = "age_range"
    DIFFICULTY = "difficulty"
    PREREQUISITES = "prerequisites"
    LEARNING_OBJECTIVES = "learning_objectives"


class MediaItem(BaseModel):
    """Model for media items (images, videos, etc.)."""
    media_id: str = Field(..., description="Unique ID for the media item")
    media_type: MediaType = Field(..., description="Type of media")
    title: str = Field(..., description="Title of the media item")
    description: Optional[str] = Field(None, description="Description of the media item")
    url: Optional[str] = Field(None, description="URL to the media (if external)")
    data: Optional[str] = Field(None, description="Base64 encoded data (if embedded)")
    alt_text: Optional[str] = Field(None, description="Alt text for accessibility")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class InteractiveElement(BaseModel):
    """Model for interactive elements in content."""
    element_id: str = Field(..., description="Unique ID for the element")
    element_type: str = Field(..., description="Type of interactive element")
    title: str = Field(..., description="Title of the interactive element")
    description: str = Field(..., description="Description of what this element does")
    instructions: str = Field(..., description="Instructions for using this element")
    data: Dict[str, Any] = Field(..., description="Element-specific data")


class AssessmentItem(BaseModel):
    """Model for assessment items (questions, quizzes, etc.)."""
    item_id: str = Field(..., description="Unique ID for the assessment item")
    item_type: str = Field(..., description="Type of assessment item")
    question: str = Field(..., description="The question or prompt")
    options: Optional[List[str]] = Field(None, description="Response options (for multiple choice)")
    correct_answer: Optional[Union[str, int, List[int]]] = Field(None, description="Correct answer or indices")
    explanation: Optional[str] = Field(None, description="Explanation of the correct answer")
    difficulty: ContentDifficulty = Field(ContentDifficulty.INTERMEDIATE, description="Difficulty level")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")


class ContentSection(BaseModel):
    """Model for a section of content."""
    section_id: str = Field(..., description="Unique ID for the section")
    title: str = Field(..., description="Section title")
    content: str = Field(..., description="Section content")
    order: int = Field(..., description="Order in the overall content")
    media: List[MediaItem] = Field(default_factory=list, description="Media items in this section")
    interactive_elements: List[InteractiveElement] = Field(default_factory=list, description="Interactive elements")
    subsections: List["ContentSection"] = Field(default_factory=list, description="Subsections")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ContentRequest(BaseModel):
    """Request model for content generation."""
    concept_id: str = Field(..., description="ID of the concept in Ptolemy")
    content_type: ContentType = Field(..., description="Type of content to generate")
    template_id: Optional[str] = Field(None, description="Optional template ID, defaults to type-specific template")
    difficulty: ContentDifficulty = Field(ContentDifficulty.INTERMEDIATE, description="Content difficulty level")
    age_range: str = Field("14-18", description="Target age range (e.g., '14-18')")
    path_id: Optional[str] = Field(None, description="Learning path ID if part of a path")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for generation")


class ContentResponse(BaseModel):
    """Response model for generated content."""
    content_id: str = Field(..., description="Unique ID for the content")
    concept_id: str = Field(..., description="ID of the concept this content is for")
    path_id: Optional[str] = Field(None, description="ID of the learning path this is part of")
    content: str = Field(..., description="The full generated content")
    sections: List[ContentSection] = Field(default_factory=list, description="Structured content sections")
    media: List[MediaItem] = Field(default_factory=list, description="Media items in the content")
    interactive_elements: List[InteractiveElement] = Field(default_factory=list, description="Interactive elements")
    assessment_items: List[AssessmentItem] = Field(default_factory=list, description="Assessment items")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class ContentFeedback(BaseModel):
    """Feedback on generated content."""
    content_id: str = Field(..., description="ID of the content being rated")
    user_id: Optional[str] = Field(None, description="ID of the user providing feedback")
    rating: Optional[int] = Field(None, description="Rating (1-5 scale)")
    feedback_text: Optional[str] = Field(None, description="Text feedback")
    improvements_needed: List[str] = Field(default_factory=list, description="Areas needing improvement")
    accuracy_score: Optional[float] = Field(None, description="Accuracy score (0-1)")
    engagement_score: Optional[float] = Field(None, description="Engagement score (0-1)")
    created_at: datetime = Field(default_factory=datetime.now, description="Feedback timestamp")