"""
Content Service
============
Service for managing content generation and retrieval.
This service acts as the central coordination point for content generation,
handling request tracking, status updates, and content storage.
"""

import logging
import time
import uuid
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple

from models.content import ContentRequest, ContentResponse, ContentFeedback
from models.template import ContentTemplate
from models.feedback import Feedback, FeedbackResponse, FeedbackSummary, ContentRating
from core.content_generator import ContentGenerator
from integrations.mongodb_service import MongoDBService
from utils.error_handling import NotFoundError, ValidationError
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class ContentService:
    """Service for managing content generation and retrieval."""
    
    def __init__(self, mongodb: Optional[MongoDBService] = None, content_generator: Optional[ContentGenerator] = None):
        """
        Initialize the content service.
        
        Args:
            mongodb: Optional MongoDB service instance
            content_generator: Optional content generator instance
        """
        self.mongodb = mongodb or MongoDBService()
        self.content_generator = content_generator or ContentGenerator()
        logger.info("ContentService initialized")
    
    async def create_content_request(self, request_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a content generation request.
        
        Args:
            request_id: Unique ID for the request
            request_data: Content generation request data
            
        Returns:
            Dictionary with request ID and status info
        """
        # Create ContentRequest object to validate
        request = ContentRequest(**request_data)
        
        logger.info(f"Creating content request for concept: {request.concept_id}")
        
        # Initialize request status
        status_info = {
            "request_id": request_id,
            "concept_id": request.concept_id,
            "content_type": request.content_type.value,
            "status": "pending",
            "progress": 0.0,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "message": "Request created, awaiting processing"
        }
        
        # Store request status
        await self.mongodb.create_content_status(status_info)
        
        # Store request
        await self.mongodb.create_content_request({
            "request_id": request_id,
            "request": request_data,
            "created_at": datetime.now().isoformat()
        })
        
        logger.info(f"Content request created: {request_id}")
        
        # Calculate estimated time
        request_complexity = self._calculate_request_complexity(request)
        estimated_time = request_complexity * 30  # 30 seconds per complexity unit
        
        return {
            "request_id": request_id,
            "status": "pending",
            "estimated_time_seconds": estimated_time
        }
    
    async def generate_content_task(self, request_id: str, request_data: ContentRequest) -> None:
        """
        Background task to process a content generation request.
        
        Args:
            request_id: ID of the request to process
            request_data: ContentRequest object with request parameters
        """
        logger.info(f"Processing content generation for request: {request_id}")
        
        try:
            # Update status to processing
            await self._update_generation_status(
                request_id, 
                status="processing",
                progress=0.1,
                message="Content generation started"
            )
            
            # Generate content
            try:
                # Update status
                await self._update_generation_status(
                    request_id, 
                    status="processing",
                    progress=0.2,
                    message="Processing template"
                )
                
                # Generate content
                response = await self.content_generator.generate_content(request)
                
                # Store content
                await self.mongodb.create_content({
                    "request_id": request_id,
                    "content": response.dict(),
                    "created_at": datetime.now().isoformat()
                })
                
                # Update status to completed
                await self._update_generation_status(
                    request_id, 
                    status="completed",
                    progress=1.0,
                    message="Content generation completed",
                    content_id=response.content_id
                )
                
                logger.info(f"Content generation completed for request: {request_id}")
                
            except Exception as e:
                logger.error(f"Error generating content: {str(e)}")
                await self._update_generation_status(
                    request_id, 
                    status="failed",
                    message=f"Error generating content: {str(e)}"
                )
                raise
                
        except Exception as e:
            logger.error(f"Error processing request {request_id}: {str(e)}")
            await self._update_generation_status(
                request_id, 
                status="failed",
                message=f"Error: {str(e)}"
            )
            raise
    
    async def get_generation_status(self, request_id: str) -> Dict[str, Any]:
        """
        Get the status of a content generation request.
        
        Args:
            request_id: ID of the request
            
        Returns:
            Status information
        """
        logger.info(f"Getting generation status for request: {request_id}")
        
        # Get status from database
        status = await self.mongodb.get_content_status(request_id)
        if not status:
            logger.warning(f"Status not found for request: {request_id}")
            return {
                "request_id": request_id,
                "status": "not_found",
                "message": "Request ID not found"
            }
        
        # Add content URL if completed
        if status.get("status") == "completed" and status.get("content_id"):
            status["content_url"] = f"/content/{status['content_id']}"
        
        return status
    
    async def get_content(self, content_id: str, 
                        include_sections: bool = True,
                        include_media: bool = True,
                        include_interactive: bool = True,
                        include_assessment: bool = True) -> ContentResponse:
        """
        Get generated content by ID.
        
        Args:
            content_id: ID of the content
            include_sections: Whether to include content sections
            include_media: Whether to include media items
            include_interactive: Whether to include interactive elements
            include_assessment: Whether to include assessment items
            
        Returns:
            Content response
        """
        logger.info(f"Getting content: {content_id}")
        
        # Get content from database
        content_data = await self.mongodb.get_content_by_id(content_id)
        if not content_data:
            logger.warning(f"Content not found: {content_id}")
            raise NotFoundError(
                message=f"Content not found",
                resource_type="content",
                resource_id=content_id
            )
        
        # Extract content from data
        content = content_data.get("content", {})
        
        # Filter fields based on parameters
        if not include_sections and "sections" in content:
            del content["sections"]
        if not include_media and "media" in content:
            del content["media"]
        if not include_interactive and "interactive_elements" in content:
            del content["interactive_elements"]
        if not include_assessment and "assessment_items" in content:
            del content["assessment_items"]
        
        # Create ContentResponse object
        return ContentResponse(**content)
    
    async def list_content(self, limit: int = 10, offset: int = 0,
                         filters: Dict[str, Any] = None) -> Tuple[List[Dict[str, Any]], int]:
        """
        List generated content with pagination and filtering.
        
        Args:
            limit: Maximum number of items to return
            offset: Number of items to skip
            filters: Optional filters
            
        Returns:
            Tuple containing (list of content items, total count)
        """
        logger.info(f"Listing content: limit={limit}, offset={offset}, filters={filters}")
        
        # Get content from database
        content_list, total = await self.mongodb.list_content(
            limit=limit, 
            offset=offset, 
            filters=filters
        )
        
        # Return the list and total count directly
        return content_list, total
    
    async def submit_feedback(self, feedback: Feedback) -> Dict[str, Any]:
        """
        Submit feedback for generated content.
        
        Args:
            feedback: Feedback data
            
        Returns:
            Dictionary with feedback ID and status
        """
        logger.info(f"Submitting feedback for content: {feedback.content_id}")
        
        # Validate feedback
        await self._validate_feedback(feedback)
        
        # Generate feedback ID if not provided
        feedback_id = feedback.feedback_id or str(uuid.uuid4())
        
        # Update feedback object
        feedback_dict = feedback.dict()
        feedback_dict["feedback_id"] = feedback_id
        if not feedback_dict.get("created_at"):
            feedback_dict["created_at"] = datetime.now().isoformat()
        if not feedback_dict.get("status"):
            feedback_dict["status"] = "received"
        
        # Store feedback
        await self.mongodb.create_feedback({
            "feedback_id": feedback_id,
            "feedback": feedback_dict,
            "created_at": datetime.now().isoformat()
        })
        
        # Check if regeneration is needed
        needs_improvement = False
        
        # Check both overall rating and individual feedback items for improvement needs
        if feedback.ratings and feedback.ratings.overall is not None and feedback.ratings.overall < 3:
            needs_improvement = True
        elif any(item.severity in ["high", "critical"] for item in feedback.feedback_items):
            needs_improvement = True
        
        logger.info(f"Feedback submitted: {feedback_id}")
        
        return {
            "feedback_id": feedback_id,
            "needs_improvement": needs_improvement,
            "message": "Feedback submitted successfully"
        }
        
    async def get_feedback(self, feedback_id: str) -> Optional[Dict[str, Any]]:
        """
        Get feedback by ID.
        
        Args:
            feedback_id: ID of the feedback
            
        Returns:
            Feedback data or None if not found
        """
        logger.info(f"Getting feedback: {feedback_id}")
        
        feedback = await self.mongodb.get_feedback(feedback_id)
        return feedback
        
    async def get_content_feedback(self, content_id: str) -> List[Dict[str, Any]]:
        """
        Get all feedback for a content item.
        
        Args:
            content_id: ID of the content
            
        Returns:
            List of feedback items
        """
        logger.info(f"Getting feedback for content: {content_id}")
        
        feedback_list = await self.mongodb.get_content_feedback(content_id)
        return feedback_list
        
    async def update_feedback_status(self, feedback_id: str, status: str) -> Dict[str, Any]:
        """
        Update the status of a feedback item.
        
        Args:
            feedback_id: ID of the feedback
            status: New status
            
        Returns:
            Updated feedback data
        """
        logger.info(f"Updating feedback status: {feedback_id} -> {status}")
        
        result = await self.mongodb.update_feedback_status(feedback_id, status)
        return result
        
    async def create_feedback_response(self, response: FeedbackResponse) -> Dict[str, Any]:
        """
        Create a response to feedback.
        
        Args:
            response: Response data
            
        Returns:
            Dictionary with response ID and status
        """
        logger.info(f"Creating response to feedback: {response.feedback_id}")
        
        # Generate response ID if not provided
        response_id = response.response_id or str(uuid.uuid4())
        
        # Update response object
        response_dict = response.dict()
        response_dict["response_id"] = response_id
        if not response_dict.get("created_at"):
            response_dict["created_at"] = datetime.now().isoformat()
        
        # Store response
        await self.mongodb.create_feedback_response({
            "response_id": response_id,
            "feedback_id": response.feedback_id,
            "response": response_dict,
            "created_at": datetime.now().isoformat()
        })
        
        # Update feedback status
        await self.update_feedback_status(response.feedback_id, response.status)
        
        logger.info(f"Feedback response created: {response_id}")
        
        return {
            "response_id": response_id,
            "status": response.status
        }
        
    async def generate_feedback_summary(self, content_id: str) -> FeedbackSummary:
        """
        Generate a summary of all feedback for a content item.
        
        Args:
            content_id: ID of the content
            
        Returns:
            Feedback summary
        """
        logger.info(f"Generating feedback summary for content: {content_id}")
        
        # Get all feedback for the content
        feedback_list = await self.get_content_feedback(content_id)
        
        if not feedback_list:
            # Return empty summary if no feedback exists
            return FeedbackSummary(
                content_id=content_id,
                feedback_count=0,
                average_ratings=ContentRating(),
                common_issues=[],
                positive_aspects=[],
                improvement_recommendations=[],
                generated_at=datetime.now()
            )
        
        # Calculate average ratings
        ratings_sum = ContentRating()
        ratings_count = ContentRating()
        
        for feedback_item in feedback_list:
            feedback = feedback_item.get("feedback", {})
            ratings = feedback.get("ratings", {})
            
            # Sum up all ratings
            if ratings.get("overall") is not None:
                ratings_sum.overall = (ratings_sum.overall or 0) + ratings["overall"]
                ratings_count.overall = (ratings_count.overall or 0) + 1
                
            if ratings.get("accuracy") is not None:
                ratings_sum.accuracy = (ratings_sum.accuracy or 0) + ratings["accuracy"]
                ratings_count.accuracy = (ratings_count.accuracy or 0) + 1
                
            if ratings.get("clarity") is not None:
                ratings_sum.clarity = (ratings_sum.clarity or 0) + ratings["clarity"]
                ratings_count.clarity = (ratings_count.clarity or 0) + 1
                
            if ratings.get("engagement") is not None:
                ratings_sum.engagement = (ratings_sum.engagement or 0) + ratings["engagement"]
                ratings_count.engagement = (ratings_count.engagement or 0) + 1
                
            if ratings.get("relevance") is not None:
                ratings_sum.relevance = (ratings_sum.relevance or 0) + ratings["relevance"]
                ratings_count.relevance = (ratings_count.relevance or 0) + 1
                
            if ratings.get("difficulty") is not None:
                ratings_sum.difficulty = (ratings_sum.difficulty or 0) + ratings["difficulty"]
                ratings_count.difficulty = (ratings_count.difficulty or 0) + 1
        
        # Calculate averages
        average_ratings = ContentRating()
        
        if ratings_count.overall and ratings_count.overall > 0:
            average_ratings.overall = round(ratings_sum.overall / ratings_count.overall, 1)
            
        if ratings_count.accuracy and ratings_count.accuracy > 0:
            average_ratings.accuracy = round(ratings_sum.accuracy / ratings_count.accuracy, 1)
            
        if ratings_count.clarity and ratings_count.clarity > 0:
            average_ratings.clarity = round(ratings_sum.clarity / ratings_count.clarity, 1)
            
        if ratings_count.engagement and ratings_count.engagement > 0:
            average_ratings.engagement = round(ratings_sum.engagement / ratings_count.engagement, 1)
            
        if ratings_count.relevance and ratings_count.relevance > 0:
            average_ratings.relevance = round(ratings_sum.relevance / ratings_count.relevance, 1)
            
        if ratings_count.difficulty and ratings_count.difficulty > 0:
            average_ratings.difficulty = round(ratings_sum.difficulty / ratings_count.difficulty, 1)
        
        # Extract common issues and positive aspects
        common_issues = {}
        positive_aspects = {}
        
        for feedback_item in feedback_list:
            feedback = feedback_item.get("feedback", {})
            
            # Process feedback items
            for item in feedback.get("feedback_items", []):
                if item.get("feedback_type") in ["accuracy", "clarity", "technical"] and item.get("severity") in ["high", "critical"]:
                    key = f"{item.get('feedback_type')}:{item.get('description')[:50]}"
                    if key not in common_issues:
                        common_issues[key] = {
                            "type": item.get("feedback_type"),
                            "description": item.get("description"),
                            "count": 0,
                            "severity": item.get("severity")
                        }
                    common_issues[key]["count"] += 1
                
                if item.get("feedback_type") in ["engagement", "suggestion"] and item.get("severity") in ["low", "medium"]:
                    key = f"{item.get('feedback_type')}:{item.get('description')[:50]}"
                    if key not in positive_aspects:
                        positive_aspects[key] = {
                            "type": item.get("feedback_type"),
                            "description": item.get("description"),
                            "count": 0
                        }
                    positive_aspects[key]["count"] += 1
        
        # Sort by count
        common_issues_list = sorted(common_issues.values(), key=lambda x: x["count"], reverse=True)
        positive_aspects_list = sorted(positive_aspects.values(), key=lambda x: x["count"], reverse=True)
        
        # Generate improvement recommendations
        improvement_recommendations = []
        
        # Add recommendations based on ratings
        if average_ratings.clarity is not None and average_ratings.clarity < 3.5:
            improvement_recommendations.append("Improve clarity of explanations")
            
        if average_ratings.engagement is not None and average_ratings.engagement < 3.5:
            improvement_recommendations.append("Make content more engaging and interactive")
            
        if average_ratings.accuracy is not None and average_ratings.accuracy < 4.0:
            improvement_recommendations.append("Review content for accuracy and correctness")
        
        # Add recommendations based on common issues
        for issue in common_issues_list[:3]:  # Top 3 issues
            if issue["count"] > 1:  # Only if reported multiple times
                improvement_recommendations.append(f"Address issue: {issue['description'][:100]}")
        
        # Create feedback summary
        summary = FeedbackSummary(
            content_id=content_id,
            feedback_count=len(feedback_list),
            average_ratings=average_ratings,
            common_issues=common_issues_list[:5],  # Top 5 issues
            positive_aspects=positive_aspects_list[:5],  # Top 5 positive aspects
            improvement_recommendations=improvement_recommendations,
            generated_at=datetime.now()
        )
        
        return summary
        
    async def list_feedback(self, limit: int = 10, offset: int = 0, 
                          filters: Dict[str, Any] = None) -> Tuple[List[Dict[str, Any]], int]:
        """
        List feedback with pagination and filtering.
        
        Args:
            limit: Maximum number of items
            offset: Number of items to skip
            filters: Optional filters
            
        Returns:
            Tuple of (feedback items, total count)
        """
        logger.info(f"Listing feedback: limit={limit}, offset={offset}, filters={filters}")
        
        feedback_list, total = await self.mongodb.list_feedback(
            limit=limit,
            offset=offset,
            filters=filters
        )
        
        return feedback_list, total
    
    async def create_regeneration_request(self, content_id: str, 
                                       modifications: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a request to regenerate content with modifications.
        
        Args:
            content_id: ID of the content to regenerate
            modifications: Optional modifications to the request
            
        Returns:
            Dictionary with request ID and status info
        """
        logger.info(f"Creating regeneration request for content: {content_id}")
        
        # Get original content
        content_data = await self.mongodb.get_content_by_id(content_id)
        if not content_data:
            logger.warning(f"Content not found: {content_id}")
            raise NotFoundError(
                message=f"Content not found",
                resource_type="content",
                resource_id=content_id
            )
        
        # Get the original request
        original_request = content_data.get("content", {}).get("original_request")
        if not original_request:
            # Try to retrieve from the request ID if available
            request_id = content_data.get("request_id")
            if request_id:
                request_data = await self.mongodb.get_content_request(request_id)
                if request_data and "request" in request_data:
                    original_request = request_data["request"]
        
        if not original_request:
            logger.warning(f"Original request not found for content: {content_id}")
            # Create a minimal request based on content data
            content_obj = content_data.get("content", {})
            original_request = {
                "concept_id": content_obj.get("concept_id"),
                "content_type": content_obj.get("content_type", "lesson"),
                "template_id": content_obj.get("template_id"),
                "difficulty": content_obj.get("difficulty", "intermediate"),
                "age_range": content_obj.get("age_range", "14-18")
            }
        
        # Apply modifications
        if modifications:
            for key, value in modifications.items():
                original_request[key] = value
        
        # Create a new ContentRequest
        request = ContentRequest(**original_request)
        
        # Create new request
        result = await self.create_content_request(request)
        
        # Add reference to original content
        await self.mongodb.update_content_request(
            result["request_id"],
            {"regenerated_from": content_id}
        )
        
        logger.info(f"Regeneration request created: {result['request_id']}")
        
        return result
    
    async def check_database(self) -> bool:
        """
        Check database connection.
        
        Returns:
            True if database is accessible, False otherwise
        """
        try:
            # Try a simple operation
            result = await self.mongodb.check_connection()
            return result
        except Exception as e:
            logger.error(f"Database check failed: {str(e)}")
            return False
    
    async def check_templates(self) -> bool:
        """
        Check if templates are available.
        
        Returns:
            True if templates are available, False otherwise
        """
        try:
            # Check if at least one template exists
            templates = await self.mongodb.list_templates(limit=1)
            return len(templates[0]) > 0
        except Exception as e:
            logger.error(f"Template check failed: {str(e)}")
            return False
    
    # -------- Private Helper Methods -------- #
    
    def _validate_content_request(self, request: ContentRequest) -> None:
        """
        Validate a content request.
        
        Args:
            request: Content request to validate
            
        Raises:
            ValidationError: If the request is invalid
        """
        errors = {}
        
        # Check required fields
        if not request.concept_id:
            errors["concept_id"] = "Concept ID is required"
        
        # Validate age range format
        if request.age_range:
            if not self._is_valid_age_range(request.age_range):
                errors["age_range"] = "Age range must be in format 'X-Y' where X and Y are numbers"
        
        if errors:
            raise ValidationError(
                message="Invalid content request",
                field_errors=errors
            )
    
    def _is_valid_age_range(self, age_range: str) -> bool:
        """
        Check if age range is valid.
        
        Args:
            age_range: Age range string
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if "-" not in age_range:
                return False
            
            min_age, max_age = age_range.split("-")
            min_age = int(min_age)
            max_age = int(max_age)
            
            return min_age < max_age and min_age >= 0 and max_age <= 100
        except:
            return False
    
    async def _validate_feedback(self, feedback: ContentFeedback) -> None:
        """
        Validate feedback for a content.
        
        Args:
            feedback: Feedback to validate
            
        Raises:
            ValidationError: If the feedback is invalid
            NotFoundError: If the content does not exist
        """
        errors = {}
        
        # Check if content exists
        content_exists = await self.mongodb.check_content_exists(feedback.content_id)
        if not content_exists:
            raise NotFoundError(
                message="Content not found",
                resource_type="content",
                resource_id=feedback.content_id
            )
        
        # Validate rating if provided
        if feedback.rating is not None and (feedback.rating < 1 or feedback.rating > 5):
            errors["rating"] = "Rating must be between 1 and 5"
        
        if errors:
            raise ValidationError(
                message="Invalid feedback",
                field_errors=errors
            )
    
    def _calculate_request_complexity(self, request: ContentRequest) -> float:
        """
        Calculate complexity of a content request for time estimation.
        
        Args:
            request: Content request
            
        Returns:
            Complexity score (higher is more complex)
        """
        complexity = 1.0
        
        # Adjust based on content type
        content_type_factors = {
            "lesson": 2.0,
            "concept_explanation": 1.5,
            "exercise": 1.2,
            "assessment": 1.3,
            "summary": 1.0,
            "related_concept": 1.2,
            "prerequisite": 1.1
        }
        
        complexity *= content_type_factors.get(request.content_type.value, 1.0)
        
        # Adjust based on difficulty
        difficulty_factors = {
            "beginner": 0.8,
            "intermediate": 1.0,
            "advanced": 1.3,
            "expert": 1.6
        }
        
        complexity *= difficulty_factors.get(request.difficulty.value, 1.0)
        
        return complexity
    
    async def _update_generation_status(self, request_id: str, 
                                    status: str, 
                                    progress: float = None,
                                    message: str = None,
                                    content_id: str = None) -> bool:
        """
        Update the status of a content generation request.
        
        Args:
            request_id: ID of the request
            status: New status
            progress: Optional progress (0.0 to 1.0)
            message: Optional status message
            content_id: Optional content ID
            
        Returns:
            True if update was successful, False otherwise
        """
        update_data = {
            "status": status,
            "updated_at": datetime.now().isoformat()
        }
        
        if progress is not None:
            update_data["progress"] = progress
        
        if message is not None:
            update_data["message"] = message
            
        if content_id is not None:
            update_data["content_id"] = content_id
        
        # Return success or failure
        result = await self.mongodb.update_content_status(request_id, update_data)
        return result