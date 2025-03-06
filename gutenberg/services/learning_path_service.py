"""
Learning Path Service
=================
Service for managing learning paths and related content generation.
"""

import logging
import time
import uuid
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from models.content import ContentRequest, ContentType, ContentDifficulty
from core.content_generator import ContentGenerator
from integrations.mongodb_service import MongoDBService
from integrations.ptolemy_client import PtolemyClient
from services.content_service import ContentService
from utils.error_handling import NotFoundError, ValidationError, PtolemyError
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class LearningPathService:
    """Service for managing learning paths and related content generation."""
    
    def __init__(self):
        """Initialize the learning path service."""
        self.mongodb = MongoDBService()
        self.ptolemy = PtolemyClient()
        self.content_generator = ContentGenerator()
        self.content_service = ContentService()
        logger.info("LearningPathService initialized")
    
    async def get_learning_path(self, path_id: str) -> Dict[str, Any]:
        """
        Get a learning path from Ptolemy.
        
        Args:
            path_id: ID of the learning path
            
        Returns:
            Learning path data
            
        Raises:
            NotFoundError: If path not found
            PtolemyError: If error communicating with Ptolemy
        """
        logger.info(f"Getting learning path: {path_id}")
        
        try:
            # Get path from Ptolemy
            path_data = await self.ptolemy.get_learning_path(path_id)
            
            if not path_data:
                logger.warning(f"Learning path not found: {path_id}")
                raise NotFoundError(
                    message="Learning path not found",
                    resource_type="learning_path",
                    resource_id=path_id
                )
            
            return path_data
            
        except PtolemyError as e:
            # Re-raise Ptolemy errors
            raise
            
        except Exception as e:
            logger.error(f"Error getting learning path: {str(e)}")
            raise PtolemyError(
                message=f"Error communicating with Ptolemy: {str(e)}",
                error_details=str(e)
            )
    
    async def generate_path_content(self, path_id: str, 
                                 age_range: str = "14-18",
                                 difficulty: Optional[str] = None,
                                 step_ids: List[str] = None) -> Dict[str, Any]:
        """
        Generate content for a learning path.
        
        Args:
            path_id: ID of the learning path
            age_range: Target age range
            difficulty: Optional difficulty level (defaults to path difficulty)
            step_ids: Optional list of specific step IDs to generate content for
            
        Returns:
            Dictionary with request IDs for each step
            
        Raises:
            NotFoundError: If path not found
            ValidationError: If parameters are invalid
            PtolemyError: If error communicating with Ptolemy
        """
        logger.info(f"Generating content for learning path: {path_id}")
        
        # Get path data
        path_data = await self.get_learning_path(path_id)
        
        # Get steps to generate content for
        steps = path_data.get("steps", [])
        if not steps:
            logger.warning(f"Learning path has no steps: {path_id}")
            raise ValidationError(
                message="Learning path has no steps",
                field_errors={"path_id": "Learning path has no steps"}
            )
        
        # Filter steps if step_ids provided
        if step_ids:
            steps = [step for step in steps if step.get("concept_id") in step_ids]
            if not steps:
                logger.warning(f"No matching steps found for provided step IDs")
                raise ValidationError(
                    message="No matching steps found for provided step IDs",
                    field_errors={"step_ids": "No matching steps found"}
                )
        
        # Determine difficulty level
        if difficulty:
            content_difficulty = ContentDifficulty(difficulty)
        else:
            # Use path difficulty
            path_difficulty = path_data.get("target_learner_level", "intermediate")
            content_difficulty = ContentDifficulty.from_learner_level(path_difficulty)
        
        # Create content requests for each step
        request_ids = {}
        for step in steps:
            concept_id = step.get("concept_id")
            if not concept_id:
                logger.warning(f"Step missing concept_id in path {path_id}")
                continue
            
            # Create content request
            request = ContentRequest(
                concept_id=concept_id,
                content_type=ContentType.LESSON,
                path_id=path_id,
                difficulty=content_difficulty,
                age_range=age_range,
                context={
                    "learning_path": path_data,
                    "step_data": step,
                    "step_order": step.get("order"),
                    "step_reason": step.get("reason"),
                    "learning_activities": step.get("learning_activities", []),
                    "estimated_time_minutes": step.get("estimated_time_minutes", 30)
                }
            )
            
            # Submit request
            try:
                result = await self.content_service.create_content_request(request)
                request_ids[concept_id] = result["request_id"]
            except Exception as e:
                logger.error(f"Error creating content request for concept {concept_id}: {str(e)}")
                request_ids[concept_id] = f"error: {str(e)}"
        
        # Prepare response
        result = {
            "path_id": path_id,
            "path_name": path_data.get("name"),
            "request_ids": request_ids,
            "step_count": len(steps),
            "age_range": age_range,
            "difficulty": content_difficulty.value,
            "message": f"Content generation started for {len(request_ids)} steps"
        }
        
        return result
    
    async def get_path_content_status(self, path_id: str, 
                                   request_ids: Dict[str, str]) -> Dict[str, Any]:
        """
        Get the status of content generation for a learning path.
        
        Args:
            path_id: ID of the learning path
            request_ids: Dictionary mapping concept IDs to request IDs
            
        Returns:
            Dictionary with status for each step
        """
        logger.info(f"Getting content status for learning path: {path_id}")
        
        # Get status for each request
        statuses = {}
        overall_progress = 0.0
        completed_count = 0
        failed_count = 0
        
        for concept_id, request_id in request_ids.items():
            if request_id.startswith("error:"):
                # This was an error during request creation
                statuses[concept_id] = {
                    "status": "failed",
                    "message": request_id,
                    "progress": 0.0
                }
                failed_count += 1
                continue
            
            try:
                status = await self.content_service.get_generation_status(request_id)
                statuses[concept_id] = status
                
                # Update counts
                if status.get("status") == "completed":
                    completed_count += 1
                    overall_progress += 1.0
                elif status.get("status") == "failed":
                    failed_count += 1
                    overall_progress += 1.0
                else:
                    overall_progress += status.get("progress", 0.0)
                    
            except Exception as e:
                logger.error(f"Error getting status for request {request_id}: {str(e)}")
                statuses[concept_id] = {
                    "request_id": request_id,
                    "status": "error",
                    "message": f"Error getting status: {str(e)}",
                    "progress": 0.0
                }
        
        # Calculate overall progress
        if request_ids:
            overall_progress /= len(request_ids)
        else:
            overall_progress = 0.0
        
        # Determine overall status
        if completed_count == len(request_ids):
            overall_status = "completed"
        elif failed_count == len(request_ids):
            overall_status = "failed"
        elif failed_count > 0:
            overall_status = "partially_failed"
        else:
            overall_status = "in_progress"
        
        # Prepare response
        result = {
            "path_id": path_id,
            "overall_status": overall_status,
            "overall_progress": overall_progress,
            "completed_count": completed_count,
            "failed_count": failed_count,
            "total_count": len(request_ids),
            "step_statuses": statuses
        }
        
        return result
    
    async def get_path_content(self, path_id: str, 
                            content_ids: Dict[str, str],
                            include_steps: bool = True) -> Dict[str, Any]:
        """
        Get content for a learning path.
        
        Args:
            path_id: ID of the learning path
            content_ids: Dictionary mapping concept IDs to content IDs
            include_steps: Whether to include full step content
            
        Returns:
            Dictionary with path data and content
        """
        logger.info(f"Getting content for learning path: {path_id}")
        
        # Get path data
        path_data = await self.get_learning_path(path_id)
        
        # Get content for each step
        step_content = {}
        for concept_id, content_id in content_ids.items():
            try:
                if include_steps:
                    content = await self.content_service.get_content(content_id)
                    step_content[concept_id] = content
                else:
                    # Just include content ID
                    step_content[concept_id] = {"content_id": content_id}
                    
            except Exception as e:
                logger.error(f"Error getting content {content_id}: {str(e)}")
                step_content[concept_id] = {
                    "error": f"Error getting content: {str(e)}"
                }
        
        # Prepare response
        result = {
            "path_id": path_id,
            "path_name": path_data.get("name"),
            "path_goal": path_data.get("goal"),
            "path_description": path_data.get("description"),
            "target_learner_level": path_data.get("target_learner_level"),
            "total_time_minutes": path_data.get("total_time_minutes"),
            "step_content": step_content
        }
        
        return result