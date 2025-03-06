"""
Gutenberg Content Generation System - Core Content Generator
==========================================================
Primary engine for generating educational content based on templates and knowledge structures.
Integrates with Ptolemy for concept data and relationships.
"""

import json
import logging
import time
import traceback
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import asyncio
import uuid

from pydantic import BaseModel, Field

from config.settings import get_config
from integrations.ptolemy_client import PtolemyClient
from integrations.mongodb_service import MongoDBService
from core.template_engine import TemplateEngine
from core.rag_processor import RAGProcessor
from models.content import (
    ContentRequest, 
    ContentResponse, 
    ContentType,
    ContentDifficulty,
    MediaType,
    TemplateVariable,
    ContentFeedback
)
from models.template import ContentTemplate, TemplateSection
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class ContentGenerator:
    """
    Core content generation engine for creating educational content
    based on templates and knowledge structures from Ptolemy.
    """
    
    def __init__(self):
        """Initialize the content generator with required services."""
        self.config = get_config()
        
        # Initialize services
        self.mongodb = MongoDBService()
        self.ptolemy = PtolemyClient()
        self.template_engine = TemplateEngine()
        self.rag_processor = RAGProcessor()
        
        # Initialize cache with expiration times
        self._concept_cache_ttl = {}  # Store expiration timestamps
        self._relationship_cache_ttl = {}
        
        # Cache configuration
        self.cache_ttl = self.config.get("ptolemy", {}).get("cache_ttl", 300)  # 5 minutes default
        
        # Cache for concept data to reduce API calls
        self._concept_cache = {}
        self._relationship_cache = {}
        
        logger.info("ContentGenerator initialized")
    
    async def generate_content(self, request: ContentRequest) -> ContentResponse:
        """
        Generate content based on the given request.
        
        This is the main entry point for content generation, which:
        1. Retrieves the template
        2. Gathers context from Ptolemy (concepts, relationships)
        3. Processes the template with the context
        4. Generates and returns the content
        
        Args:
            request: Content generation request with parameters
            
        Returns:
            Generated content with metadata
        """
        start_time = time.time()
        logger.info(f"Content generation request: type={request.content_type}, concept_id={request.concept_id}")
        
        try:
            # Generate a unique ID for this content
            content_id = str(uuid.uuid4())
            
            # Get template
            template = await self._get_template(request.template_id, request.content_type)
            if not template:
                raise ValueError(f"Template not found: {request.template_id}")
            
            # Get context from Ptolemy
            context = await self._gather_context(request)
            
            # Process template with context
            processed_content = await self.template_engine.process_template(
                template=template,
                context=context,
                content_type=request.content_type,
                difficulty=request.difficulty,
                age_range=request.age_range,
                rag_processor=self.rag_processor
            )
            
            # Create response
            response = ContentResponse(
                content_id=content_id,
                concept_id=request.concept_id,
                path_id=request.path_id,
                content=processed_content.content,
                sections=processed_content.sections,
                metadata={
                    "generator_version": self.config.version,
                    "generation_time": time.time() - start_time,
                    "source_template": request.template_id,
                    "content_type": request.content_type.value,
                    "difficulty": request.difficulty.value,
                    "generation_date": datetime.now().isoformat(),
                    "ptolemy_concepts": [c["id"] for c in context.get("concepts", [])],
                    "word_count": len(processed_content.content.split()),
                    "reading_time_minutes": len(processed_content.content.split()) / 200,  # Approximation
                    "age_range": request.age_range
                },
                media=processed_content.media,
                interactive_elements=processed_content.interactive_elements,
                assessment_items=processed_content.assessment_items,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # Save content to database
            await self._save_content(response)
            
            logger.info(f"Content generated successfully: {content_id} in {time.time() - start_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
    
    async def generate_path_content(self, path_id: str, 
                                    step_ids: Optional[List[str]] = None,
                                    difficulty: Optional[ContentDifficulty] = None,
                                    age_range: str = "14-18") -> Dict[str, Any]:
        """
        Generate content for an entire learning path or specific steps.
        
        Args:
            path_id: Ptolemy learning path ID
            step_ids: Optional list of specific step IDs to generate content for
            difficulty: Content difficulty level
            age_range: Target age range
            
        Returns:
            Dictionary with path metadata and generated content for each step
        """
        logger.info(f"Generating content for learning path: {path_id}")
        start_time = time.time()
        
        try:
            # Get learning path from Ptolemy
            path_data = await self.ptolemy.get_learning_path(path_id)
            if not path_data:
                raise ValueError(f"Learning path not found: {path_id}")
            
            # If no step_ids provided, generate for all steps
            if not step_ids:
                step_ids = [step.get("concept_id") for step in path_data.get("steps", [])]
            
            # Create a mapping of step concept IDs to step data
            step_map = {step.get("concept_id"): step for step in path_data.get("steps", [])}
            
            # Generate content for each step
            step_contents = {}
            for step_id in step_ids:
                # Get concept data for this step
                concept_id = step_id  # The step ID is the concept ID in Ptolemy
                
                # Get step metadata
                step_data = step_map.get(concept_id, {})
                step_difficulty = difficulty or ContentDifficulty.from_learner_level(path_data.get("target_learner_level", "beginner"))
                
                # Determine appropriate template based on step position
                template_id = self._select_template_for_step(
                    concept_id=concept_id,
                    step_data=step_data,
                    is_first=(step_id == step_ids[0]),
                    is_last=(step_id == step_ids[-1])
                )
                
                # Create content request
                request = ContentRequest(
                    concept_id=concept_id,
                    path_id=path_id,
                    content_type=ContentType.LESSON,
                    template_id=template_id,
                    difficulty=step_difficulty,
                    age_range=age_range,
                    context={
                        "learning_path": path_data,
                        "step_data": step_data,
                        "step_order": step_data.get("order"),
                        "step_reason": step_data.get("reason"),
                        "learning_activities": step_data.get("learning_activities", []),
                        "estimated_time_minutes": step_data.get("estimated_time_minutes", 30)
                    }
                )
                
                # Generate content for this step
                step_content = await self.generate_content(request)
                step_contents[concept_id] = step_content
            
            # Create the overall path content response
            path_content = {
                "path_id": path_id,
                "path_name": path_data.get("name"),
                "path_goal": path_data.get("goal"),
                "path_difficulty": path_data.get("target_learner_level"),
                "total_time_minutes": path_data.get("total_time_minutes"),
                "generated_steps": len(step_contents),
                "step_contents": step_contents,
                "generation_time": time.time() - start_time,
                "generated_at": datetime.now().isoformat()
            }
            
            logger.info(f"Generated content for {len(step_contents)} steps in path {path_id}")
            return path_content
            
        except Exception as e:
            logger.error(f"Error generating path content: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
    
    async def generate_concept_graph_content(self, 
                                      concept_id: str,
                                      depth: int = 1,
                                      relationship_types: Optional[List[str]] = None,
                                      difficulty: ContentDifficulty = ContentDifficulty.INTERMEDIATE,
                                      age_range: str = "14-18") -> Dict[str, Any]:
        """
        Generate content for a concept and its related concepts.
        
        Args:
            concept_id: Central concept ID
            depth: Relationship traversal depth
            relationship_types: Optional relationship types to follow
            difficulty: Content difficulty level
            age_range: Target age range
            
        Returns:
            Dictionary with graph structure and generated content
        """
        logger.info(f"Generating content for concept graph: {concept_id} with depth {depth}")
        start_time = time.time()
        
        try:
            # Get concept graph from Ptolemy
            graph_data = await self.ptolemy.get_concept_graph(
                concept_id=concept_id,
                depth=depth,
                relationship_types=relationship_types
            )
            
            if not graph_data or not graph_data.get("nodes"):
                raise ValueError(f"Concept graph not found or empty: {concept_id}")
            
            # Generate content for each node in the graph
            node_contents = {}
            for node in graph_data.get("nodes", []):
                node_id = node.get("id")
                
                # Select template based on relationship to central concept
                is_central = (node_id == concept_id)
                relationship_to_central = self._get_relationship_to_central(
                    node_id=node_id,
                    central_id=concept_id,
                    edges=graph_data.get("edges", [])
                )
                
                template_id = self._select_template_for_graph_node(
                    node_id=node_id,
                    is_central=is_central,
                    relationship=relationship_to_central
                )
                
                # Create content request
                request = ContentRequest(
                    concept_id=node_id,
                    content_type=ContentType.CONCEPT_EXPLANATION if is_central else ContentType.RELATED_CONCEPT,
                    template_id=template_id,
                    difficulty=difficulty,
                    age_range=age_range,
                    context={
                        "central_concept_id": concept_id,
                        "is_central": is_central,
                        "relationship_to_central": relationship_to_central,
                        "graph_depth": depth,
                        "node_data": node
                    }
                )
                
                # Generate content
                node_content = await self.generate_content(request)
                node_contents[node_id] = node_content
            
            # Create response with graph structure and content
            graph_content = {
                "central_concept_id": concept_id,
                "depth": depth,
                "graph_structure": {
                    "nodes": graph_data.get("nodes", []),
                    "edges": graph_data.get("edges", [])
                },
                "node_contents": node_contents,
                "generation_time": time.time() - start_time,
                "generated_at": datetime.now().isoformat()
            }
            
            logger.info(f"Generated content for {len(node_contents)} nodes in concept graph")
            return graph_content
            
        except Exception as e:
            logger.error(f"Error generating concept graph content: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
    
    async def process_content_feedback(self, feedback: ContentFeedback) -> Dict[str, Any]:
        """
        Process feedback on generated content.
        
        Args:
            feedback: Feedback data from users
            
        Returns:
            Processing result with acknowledgment
        """
        logger.info(f"Processing feedback for content: {feedback.content_id}")
        
        try:
            # Save feedback to database
            feedback_id = await self.mongodb.create_feedback(feedback)
            
            # If rating is below threshold, flag for review
            needs_improvement = False
            if feedback.rating and feedback.rating < self.config.content.feedback_threshold_for_regeneration:
                needs_improvement = True
                
            # Forward feedback to Socrates if enabled
            if self.config.feedback.forward_to_socrates:
                try:
                    # This would be implemented in socrates_client.py
                    # await self.socrates.send_feedback(feedback)
                    pass
                except Exception as e:
                    logger.warning(f"Failed to forward feedback to Socrates: {e}")
            
            return {
                "success": True,
                "feedback_id": feedback_id,
                "needs_improvement": needs_improvement,
                "processed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
    
    async def regenerate_content(self, content_id: str, modifications: Dict[str, Any] = None) -> ContentResponse:
        """
        Regenerate content with optional modifications.
        
        Args:
            content_id: ID of existing content to regenerate
            modifications: Optional parameters to modify for regeneration
            
        Returns:
            Newly generated content
        """
        logger.info(f"Regenerating content: {content_id}")
        
        try:
            # Get existing content
            existing_content = await self.mongodb.get_content(content_id)
            if not existing_content:
                raise ValueError(f"Content not found: {content_id}")
            
            # Create new request based on existing content with modifications
            request = ContentRequest(
                concept_id=existing_content.get("concept_id"),
                path_id=existing_content.get("path_id"),
                content_type=ContentType(existing_content.get("metadata", {}).get("content_type", "lesson")),
                template_id=existing_content.get("metadata", {}).get("source_template"),
                difficulty=ContentDifficulty(existing_content.get("metadata", {}).get("difficulty", "intermediate")),
                age_range=existing_content.get("metadata", {}).get("age_range", "14-18"),
                context=existing_content.get("metadata", {}).get("generation_context", {})
            )
            
            # Apply modifications
            if modifications:
                for key, value in modifications.items():
                    if hasattr(request, key):
                        setattr(request, key, value)
            
            # Generate new content
            new_content = await self.generate_content(request)
            
            # Add reference to original content
            new_content.metadata["regenerated_from"] = content_id
            
            # Save updated content
            await self._save_content(new_content)
            
            logger.info(f"Content regenerated successfully: {new_content.content_id}")
            return new_content
            
        except Exception as e:
            logger.error(f"Error regenerating content: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
    
    # -------- Private Helper Methods -------- #
    
    async def _get_template(self, template_id: Optional[str], content_type: ContentType) -> ContentTemplate:
        """Get template by ID or find appropriate default for content type."""
        if template_id:
            # Get specific template
            template = await self.mongodb.get_template(template_id)
            if not template:
                raise ValueError(f"Template not found: {template_id}")
            return ContentTemplate(**template)
        else:
            # Get default template for content type
            default_template_id = self._get_default_template_id(content_type)
            template = await self.mongodb.get_template(default_template_id)
            if not template:
                raise ValueError(f"Default template not found for {content_type}")
            return ContentTemplate(**template)
    
    def _get_default_template_id(self, content_type: ContentType) -> str:
        """Get default template ID for a content type."""
        content_type_map = {
            ContentType.LESSON: "default",  # Use the available default template
            ContentType.CONCEPT_EXPLANATION: "default",
            ContentType.EXERCISE: "default",
            ContentType.ASSESSMENT: "default",
            ContentType.SUMMARY: "default",
            ContentType.RELATED_CONCEPT: "default",
            ContentType.PREREQUISITE: "default"
        }
        template_id = content_type_map.get(content_type, "default")
        
        # Check if template exists in MongoDB
        async def _check_template():
            try:
                template = await self.mongodb.get_template(template_id)
                if template:
                    return template_id
                else:
                    return "default"  # Fall back to default if specialized template doesn't exist
            except Exception as e:
                logger.warning(f"Error checking template existence: {e}")
                return "default"
                
        # Run the async function in the current event loop or create a new one
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context
                return template_id  # Just return without checking to avoid nested awaits
            else:
                return loop.run_until_complete(_check_template())
        except Exception as e:
            logger.warning(f"Error in template selection: {e}")
            return "default"
    
    async def _gather_context(self, request: ContentRequest) -> Dict[str, Any]:
        """
        Gather context information from Ptolemy and other sources.
        
        This builds a rich context object containing:
        - Primary concept data
        - Related concept data based on relationships
        - Learning path data (if applicable)
        - User-provided context
        """
        context = {}
        
        # Start with user-provided context
        if request.context:
            context.update(request.context)
        
        # Get primary concept data
        if request.concept_id:
            primary_concept = await self._get_concept_with_details(request.concept_id)
            if primary_concept:
                context["concept"] = primary_concept
                
                # Get relationships and related concepts
                relationships = await self._get_concept_relationships(request.concept_id)
                context["relationships"] = relationships
                
                # Get related concepts data
                related_concepts = []
                for rel in relationships:
                    target_id = rel.get("target_id")
                    source_id = rel.get("source_id")
                    related_id = target_id if source_id == request.concept_id else source_id
                    related_concept = await self._get_concept(related_id)
                    if related_concept:
                        # Add relationship information to the concept
                        related_concept["relationship"] = {
                            "type": rel.get("relationship_type"),
                            "description": rel.get("description", ""),
                            "strength": rel.get("strength", 0.5),
                            "direction": "outgoing" if source_id == request.concept_id else "incoming"
                        }
                        related_concepts.append(related_concept)
                
                context["related_concepts"] = related_concepts
                context["prerequisite_concepts"] = [c for c in related_concepts 
                                                 if c.get("relationship", {}).get("type") == "prerequisite" 
                                                 and c.get("relationship", {}).get("direction") == "outgoing"]
                context["builds_on_concepts"] = [c for c in related_concepts 
                                              if c.get("relationship", {}).get("type") == "builds_on" 
                                              and c.get("relationship", {}).get("direction") == "outgoing"]
        
        # Get learning path data if applicable
        if request.path_id:
            path_data = await self.ptolemy.get_learning_path(request.path_id)
            if path_data:
                context["learning_path"] = path_data
                
                # Find current step in path if applicable
                if request.concept_id:
                    current_step = next((step for step in path_data.get("steps", []) 
                                      if step.get("concept_id") == request.concept_id), None)
                    if current_step:
                        context["current_step"] = current_step
                        
                        # Get previous and next steps
                        step_order = current_step.get("order", 0)
                        context["previous_step"] = next((step for step in path_data.get("steps", []) 
                                                     if step.get("order") == step_order - 1), None)
                        context["next_step"] = next((step for step in path_data.get("steps", []) 
                                                  if step.get("order") == step_order + 1), None)
        
        # Add all concepts to a list for easy access
        concepts = [context.get("concept")] if context.get("concept") else []
        concepts.extend(context.get("related_concepts", []))
        context["concepts"] = [c for c in concepts if c]  # Filter out None
        
        # Add metadata
        context["difficulty"] = request.difficulty.value
        context["age_range"] = request.age_range
        context["content_type"] = request.content_type.value
        
        return context
    
    async def _get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Get concept from cache or Ptolemy with TTL handling.
        
        Args:
            concept_id: ID of the concept to retrieve
            
        Returns:
            Concept data or None if not found
        """
        # Check if in cache and not expired
        current_time = time.time()
        if concept_id in self._concept_cache:
            # Check if cache entry is still valid
            if concept_id in self._concept_cache_ttl and self._concept_cache_ttl[concept_id] > current_time:
                logger.debug(f"Cache hit for concept: {concept_id}")
                return self._concept_cache[concept_id]
            # Remove expired entry
            elif concept_id in self._concept_cache_ttl:
                logger.debug(f"Cache expired for concept: {concept_id}")
                del self._concept_cache[concept_id]
                del self._concept_cache_ttl[concept_id]
        
        # Fetch from Ptolemy
        try:
            concept = await self.ptolemy.get_concept(concept_id)
            if concept:
                # Store in cache with expiration time
                self._concept_cache[concept_id] = concept
                self._concept_cache_ttl[concept_id] = current_time + self.cache_ttl
                logger.debug(f"Cached concept: {concept_id}, expires in {self.cache_ttl}s")
            return concept
        except Exception as e:
            logger.error(f"Error fetching concept {concept_id}: {str(e)}")
            return None
    
    async def _get_concept_with_details(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Get concept with detailed relationship data."""
        concept_with_rels = await self.ptolemy.get_concept_with_relationships(concept_id)
        if concept_with_rels and "concept" in concept_with_rels:
            self._concept_cache[concept_id] = concept_with_rels["concept"]
            return concept_with_rels["concept"]
        return None
    
    async def _get_concept_relationships(self, concept_id: str) -> List[Dict[str, Any]]:
        """
        Get concept relationships from cache or Ptolemy with TTL handling.
        
        Args:
            concept_id: ID of the concept to get relationships for
            
        Returns:
            List of relationship data
        """
        cache_key = f"rels_{concept_id}"
        current_time = time.time()
        
        # Check if in cache and not expired
        if cache_key in self._relationship_cache:
            # Check if cache entry is still valid
            if cache_key in self._relationship_cache_ttl and self._relationship_cache_ttl[cache_key] > current_time:
                logger.debug(f"Cache hit for relationships: {concept_id}")
                return self._relationship_cache[cache_key]
            # Remove expired entry
            elif cache_key in self._relationship_cache_ttl:
                logger.debug(f"Cache expired for relationships: {concept_id}")
                del self._relationship_cache[cache_key]
                del self._relationship_cache_ttl[cache_key]
        
        # Fetch from Ptolemy
        try:
            relationships = await self.ptolemy.get_concept_relationships(concept_id)
            if relationships:
                # Store in cache with expiration time
                self._relationship_cache[cache_key] = relationships
                self._relationship_cache_ttl[cache_key] = current_time + self.cache_ttl
                logger.debug(f"Cached relationships for concept: {concept_id}, expires in {self.cache_ttl}s")
            return relationships or []
        except Exception as e:
            logger.error(f"Error fetching relationships for concept {concept_id}: {str(e)}")
            return []
    
    async def _save_content(self, content: ContentResponse) -> str:
        """Save generated content to MongoDB."""
        return await self.mongodb.create_content(content.dict())
    
    def _select_template_for_step(self, concept_id: str, step_data: Dict[str, Any], 
                                is_first: bool, is_last: bool) -> str:
        """
        Select appropriate template based on step position in learning path.
        
        Args:
            concept_id: ID of the concept for this step
            step_data: Metadata about this step in the learning path
            is_first: Whether this is the first step in the path
            is_last: Whether this is the last step in the path
            
        Returns:
            Template ID to use for this step
        """
        # First, check if step_data has a specific template_id
        if step_data and "template_id" in step_data:
            return step_data["template_id"]
            
        # Next, determine based on position and content type
        if is_first:
            # The first step is usually an introduction/overview
            return "default_concept"
        elif is_last:
            # The last step often includes assessment
            return "default_assessment"
        else:
            # Middle steps use standard lesson template
            return "default_lesson"
    
    def _select_template_for_graph_node(self, node_id: str, is_central: bool, 
                                     relationship: Dict[str, Any]) -> str:
        """
        Select appropriate template based on node's position in concept graph.
        
        Args:
            node_id: ID of the node in the graph
            is_central: Whether this is the central concept in the graph
            relationship: Information about the relationship to the central concept
            
        Returns:
            Template ID to use for this node
        """
        if is_central:
            # The central concept gets the most detailed explanation
            return "default_concept"
        
        # Select based on relationship type
        if relationship:
            rel_type = relationship.get("type")
            if rel_type == "prerequisite":
                # Prerequisites need clear explanations but focused on supporting the central concept
                return "default_concept"
            elif rel_type == "builds_on":
                # Build-on relationships are extensions of the central concept
                return "default_lesson"
            elif rel_type == "example_of":
                # Examples are concrete applications of the central concept
                return "default_exercise"
            elif rel_type == "part_of":
                # Parts are components of the central concept
                return "default_concept"
            elif rel_type == "contrasts_with":
                # Contrasting concepts highlight differences
                return "default_concept"
        
        # Default for most related concepts
        return "default_concept"
    
    def _get_relationship_to_central(self, node_id: str, central_id: str, 
                                 edges: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find relationship between a node and the central concept."""
        for edge in edges:
            source_id = edge.get("source_id")
            target_id = edge.get("target_id")
            
            if (source_id == central_id and target_id == node_id) or \
               (source_id == node_id and target_id == central_id):
                return {
                    "type": edge.get("relationship_type"),
                    "description": edge.get("description", ""),
                    "strength": edge.get("strength", 0.5),
                    "direction": "outgoing" if source_id == central_id else "incoming"
                }
        
        return None