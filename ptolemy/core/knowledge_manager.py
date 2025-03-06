"""
Ptolemy Knowledge Map System - Knowledge Manager
============================================
Core service coordinating knowledge map operations across all services.
"""

import logging
import uuid
import time
import traceback
import threading
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from datetime import datetime
from functools import wraps
from collections import OrderedDict

from config import Config
from models import (
    Concept, ConceptCreate, ConceptUpdate, 
    Relationship, RelationshipCreate, RelationshipUpdate,
    LearningPath, LearningPathRequest, LearningPathStep,
    ValidationResult, KnowledgeGap, Activity, ActivityType,
    ConceptType, RelationshipType, DifficultyLevel, ValidationStatus,
    DomainStructureRequest, ConceptSimilarityResult, ConceptWithRelationships
)
from db.neo4j_service import Neo4jService
from db.mongodb_service import MongoService
from db.qdrant_service import QdrantService
from embeddings.embedding_service import EmbeddingService
from llm.generation import LLMService

# Configure module-level logger
logger = logging.getLogger("knowledge.manager")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# LRU Cache implementation for KnowledgeManager
class LRUCache:
    """LRU Cache implementation with max size and optional TTL."""
    
    def __init__(self, max_size=1000, ttl_seconds=3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
        self._lock = threading.RLock()  # Thread safety lock
    
    def get(self, key):
        """Get item from cache if it exists and hasn't expired."""
        with self._lock:
            now = time.time()
            if key in self.cache:
                # Check TTL
                if self.ttl_seconds > 0:
                    if now - self.timestamps[key] > self.ttl_seconds:
                        self.remove(key)
                        return None
                
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None
    
    def put(self, key, value):
        """Add item to cache, evicting oldest if at capacity."""
        with self._lock:
            # Remove if already exists
            if key in self.cache:
                self.remove(key)
            
            # Evict oldest if at capacity
            if len(self.cache) >= self.max_size:
                oldest = next(iter(self.cache))
                self.remove(oldest)
            
            # Add new item
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def remove(self, key):
        """Remove item from cache."""
        with self._lock:
            if key in self.cache:
                self.cache.pop(key)
                self.timestamps.pop(key, None)
    
    def clear(self):
        """Clear all items from cache."""
        with self._lock:
            self.cache.clear()
            self.timestamps.clear()


def transaction(method):
    """Decorator for implementing transaction management across services."""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        # Track changes for potential rollback
        changes = []
        
        try:
            # Start transaction
            logger.debug(f"Starting transaction for {method.__name__}")
            
            # Record original method call for potential rollback
            result = method(self, *args, **kwargs)
            
            logger.debug(f"Transaction completed successfully for {method.__name__}")
            return result
        except Exception as e:
            # Log the original error
            logger.error(f"Transaction failed in {method.__name__}: {e}")
            logger.debug(traceback.format_exc())
            
            # Attempt rollback of recorded changes
            for change in reversed(changes):
                try:
                    service, operation, entity_id = change
                    logger.debug(f"Rolling back {operation} on {entity_id} in {service}")
                    
                    if operation == "create":
                        if service == "mongodb":
                            self.mongo_service.delete_entity(entity_id)
                        elif service == "neo4j":
                            self.neo4j_service.delete_entity(entity_id)
                        elif service == "qdrant":
                            self.qdrant_service.delete_embedding(entity_id)
                    # Add other rollback operations as needed
                except Exception as rollback_error:
                    logger.error(f"Rollback failed: {rollback_error}")
            
            # Re-raise the original exception
            raise
    return wrapper


class KnowledgeManager:
    """Core service for managing the knowledge map system."""
    
    def __init__(self, config: Config):
        """Initialize the knowledge manager with all required services.
        
        Args:
            config: Configuration for all services
        """
        logger.info("Initializing Knowledge Manager")
        self.config = config
        
        # Initialize database services
        try:
            self.neo4j_service = Neo4jService(config.neo4j)
            logger.info("Neo4j service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j service: {e}")
            self.neo4j_service = None
        
        try:
            self.mongo_service = MongoService(config.mongo)
            logger.info("MongoDB service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB service: {e}")
            self.mongo_service = None
        
        try:
            self.qdrant_service = QdrantService(config.qdrant)
            logger.info("Qdrant service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant service: {e}")
            self.qdrant_service = None
        
        # Initialize processing services
        try:
            self.embedding_service = EmbeddingService(config.embeddings)
            logger.info("Embedding service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Embedding service: {e}")
            self.embedding_service = None
        
        try:
            self.llm_service = LLMService(config.llm, config.prompts)
            logger.info("LLM service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {e}")
            self.llm_service = None
        
        # Initialize caches with size limits
        self._concept_cache = LRUCache(
            max_size=config.cache.get("concept_cache_size", 1000),
            ttl_seconds=config.cache.get("concept_cache_ttl", 3600)
        )
        self._relationship_cache = LRUCache(
            max_size=config.cache.get("relationship_cache_size", 1000),
            ttl_seconds=config.cache.get("relationship_cache_ttl", 3600)
        )
        
        # Thread pool for parallel operations
        self._thread_pool = None
        if config.general.get("use_threading", False):
            try:
                from concurrent.futures import ThreadPoolExecutor
                self._thread_pool = ThreadPoolExecutor(
                    max_workers=config.general.get("max_workers", 10)
                )
                logger.info(f"Thread pool initialized with {config.general.get('max_workers', 10)} workers")
            except Exception as e:
                logger.error(f"Failed to initialize thread pool: {e}")
        
        logger.info("Knowledge Manager initialized successfully")
    
    def close(self):
        """Close all service connections."""
        logger.info("Closing Knowledge Manager connections")
        
        # Close thread pool if exists
        if self._thread_pool:
            try:
                self._thread_pool.shutdown(wait=False)
                logger.info("Thread pool shut down")
            except Exception as e:
                logger.error(f"Error shutting down thread pool: {e}")
        
        # Close each service
        if self.neo4j_service:
            try:
                self.neo4j_service.close()
                logger.info("Neo4j connection closed")
            except Exception as e:
                logger.error(f"Error closing Neo4j connection: {e}")
        
        if self.mongo_service:
            try:
                self.mongo_service.close()
                logger.info("MongoDB connection closed")
            except Exception as e:
                logger.error(f"Error closing MongoDB connection: {e}")
        
        if self.qdrant_service:
            try:
                self.qdrant_service.close()
                logger.info("Qdrant connection closed")
            except Exception as e:
                logger.error(f"Error closing Qdrant connection: {e}")
        
        # Clear caches
        self._concept_cache.clear()
        self._relationship_cache.clear()
        
        logger.info("Knowledge Manager: all connections closed")
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all services.
        
        Returns:
            Dictionary with health status of all services
        """
        logger.info("Performing health check on all services")
        
        # Check each service
        neo4j_health = {"status": "unavailable", "error": "Service not initialized"}
        if self.neo4j_service:
            try:
                neo4j_health = self.neo4j_service.health_check()
            except Exception as e:
                logger.error(f"Error checking Neo4j health: {e}")
                neo4j_health = {"status": "error", "error": str(e)}
        
        mongo_health = {"status": "unavailable", "error": "Service not initialized"}
        if self.mongo_service:
            try:
                mongo_health = self.mongo_service.health_check()
            except Exception as e:
                logger.error(f"Error checking MongoDB health: {e}")
                mongo_health = {"status": "error", "error": str(e)}
        
        qdrant_health = {"status": "unavailable", "error": "Service not initialized"}
        if self.qdrant_service:
            try:
                qdrant_health = self.qdrant_service.health_check()
            except Exception as e:
                logger.error(f"Error checking Qdrant health: {e}")
                qdrant_health = {"status": "error", "error": str(e)}
        
        embedding_health = {"status": "unavailable", "error": "Service not initialized"}
        if self.embedding_service:
            try:
                embedding_health = self.embedding_service.get_model_info()
            except Exception as e:
                logger.error(f"Error checking Embedding service health: {e}")
                embedding_health = {"status": "error", "error": str(e)}
        
        # Determine overall status
        all_healthy = (
            neo4j_health.get("status") == "connected" and
            mongo_health.get("status") == "connected" and
            qdrant_health.get("status") == "connected" and
            embedding_health.get("status") == "loaded"
        )
        
        # Return consolidated health status
        health_status = {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.now().isoformat(),
            "version": self.config.version,
            "services": {
                "neo4j": neo4j_health,
                "mongodb": mongo_health,
                "qdrant": qdrant_health,
                "embedding": embedding_health
            },
            "cache_stats": {
                "concept_cache": len(self._concept_cache.cache),
                "relationship_cache": len(self._relationship_cache.cache)
            }
        }
        
        logger.info(f"Health check completed with status: {health_status['status']}")
        return health_status
    
    #---------------------------
    # Concept Management
    #---------------------------
    
    @transaction
    def create_concept(self, concept_data: ConceptCreate, user_id: Optional[str] = None) -> Concept:
        """Create a new concept in the knowledge map.
        
        Args:
            concept_data: Data for the new concept
            user_id: Optional ID of the user creating the concept
            
        Returns:
            The created concept
            
        Raises:
            ValueError: If input validation fails
            Exception: If creation fails in the primary database
        """
        logger.info(f"Creating concept: {concept_data.name}")
        
        # Input validation
        self._validate_concept_create(concept_data)
        
        # Generate unique ID and timestamps
        concept_id = str(uuid.uuid4())
        now = datetime.now()
        
        # Create concept object
        concept = Concept(
            id=concept_id,
            name=concept_data.name,
            description=concept_data.description,
            concept_type=concept_data.concept_type,
            difficulty=concept_data.difficulty,
            parent_id=concept_data.parent_id,
            importance=concept_data.importance,
            complexity=concept_data.complexity or 0.5,
            keywords=concept_data.keywords or [],
            estimated_learning_time_minutes=concept_data.estimated_learning_time_minutes,
            taxonomies=concept_data.taxonomies,
            external_references=concept_data.external_references,
            metadata=concept_data.metadata,
            validation_status=ValidationStatus.PENDING,
            created_at=now,
            updated_at=now,
            version=1
        )
        
        # Check if MongoDB service is available
        if not self.mongo_service:
            raise Exception("MongoDB service not available")
        
        # Store in MongoDB (primary storage)
        start_time = time.time()
        if not self.mongo_service.create_concept(concept):
            raise Exception(f"Failed to create concept in MongoDB: {concept.name}")
        logger.debug(f"MongoDB create concept took {time.time() - start_time:.3f}s")
        
        # Store in Neo4j (graph storage)
        if self.neo4j_service:
            start_time = time.time()
            try:
                neo4j_id = self.neo4j_service.create_concept(concept)
                if not neo4j_id:
                    logger.warning(f"Failed to create concept in Neo4j: {concept.id} - continuing with MongoDB only")
                logger.debug(f"Neo4j create concept took {time.time() - start_time:.3f}s")
            except Exception as e:
                logger.error(f"Error creating concept in Neo4j: {e}")
                # Don't fail the entire operation if Neo4j fails
        
        # Generate and store embedding
        if self.embedding_service and self.qdrant_service:
            start_time = time.time()
            try:
                embedding_success = self._generate_and_store_embedding(concept)
                if not embedding_success:
                    logger.warning(f"Failed to generate/store embedding for concept: {concept.id}")
                logger.debug(f"Embedding generation took {time.time() - start_time:.3f}s")
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                # Don't fail the operation if embedding fails
        
        # Log activity
        if user_id:
            self._log_activity(
                ActivityType.CREATE,
                user_id,
                "concept",
                concept.id,
                {"name": concept.name, "concept_type": concept.concept_type.value}
            )
        
        # Add to cache
        self._concept_cache.put(concept.id, concept)
        
        logger.info(f"Concept created successfully: {concept.id} - {concept.name}")
        return concept
    
    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Get a concept by ID.
        
        Args:
            concept_id: ID of the concept to retrieve
            
        Returns:
            The concept if found, None otherwise
        """
        if not concept_id:
            return None
        
        logger.debug(f"Getting concept: {concept_id}")
            
        # Check cache first
        cached_concept = self._concept_cache.get(concept_id)
        if cached_concept:
            logger.debug(f"Concept found in cache: {concept_id}")
            return cached_concept
        
        # Check if MongoDB service is available
        if not self.mongo_service:
            logger.error("MongoDB service not available")
            return None
        
        # Get from MongoDB
        start_time = time.time()
        try:
            concept_data = self.mongo_service.get_concept(concept_id)
            logger.debug(f"MongoDB get concept took {time.time() - start_time:.3f}s")
        except Exception as e:
            logger.error(f"Error retrieving concept from MongoDB: {e}")
            return None
        
        if not concept_data:
            logger.debug(f"Concept not found: {concept_id}")
            return None
        
        try:
            # Convert to Concept object
            concept = Concept(**concept_data)
            
            # Update cache
            self._concept_cache.put(concept.id, concept)
            
            logger.debug(f"Concept retrieved: {concept.id} - {concept.name}")
            return concept
        except Exception as e:
            logger.error(f"Error parsing concept data: {e}")
            return None

    @transaction
    def update_concept(self, concept_id: str, updates: ConceptUpdate, 
                       user_id: Optional[str] = None) -> Optional[Concept]:
        """Update an existing concept.
        
        Args:
            concept_id: ID of the concept to update
            updates: Data to update
            user_id: Optional ID of the user performing the update
            
        Returns:
            Updated concept if successful, None otherwise
            
        Raises:
            ValueError: If input validation fails
            Exception: If update fails in the primary database
        """
        logger.info(f"Updating concept: {concept_id}")
        
        # Input validation
        self._validate_concept_update(concept_id, updates)
        
        # Get current concept
        current_concept = self.get_concept(concept_id)
        if not current_concept:
            logger.error(f"Concept not found: {concept_id}")
            return None
        
        # Prepare updates - using exclude_unset=True to only include fields actually provided
        update_dict = {k: v for k, v in updates.dict(exclude_unset=True).items() if v is not None}
        
        # Add metadata
        update_dict["updated_at"] = datetime.now()
        update_dict["version"] = current_concept.version + 1
        
        # Check if fields affecting semantic meaning have changed
        semantic_change = any(key in update_dict for key in [
            "name", "description", "keywords", "concept_type", "taxonomies"
        ])
        
        # Check if MongoDB service is available
        if not self.mongo_service:
            raise Exception("MongoDB service not available")
        
        # Update in MongoDB
        start_time = time.time()
        if not self.mongo_service.update_concept(concept_id, update_dict):
            raise Exception(f"Failed to update concept in MongoDB: {concept_id}")
        logger.debug(f"MongoDB update concept took {time.time() - start_time:.3f}s")
        
        # Update in Neo4j
        if self.neo4j_service:
            start_time = time.time()
            try:
                neo4j_success = self.neo4j_service.update_concept(concept_id, update_dict)
                if not neo4j_success:
                    logger.warning(f"Failed to update concept in Neo4j: {concept_id}")
                logger.debug(f"Neo4j update concept took {time.time() - start_time:.3f}s")
            except Exception as e:
                logger.error(f"Error updating concept in Neo4j: {e}")
                # Don't fail the operation if Neo4j fails
        
        # Clear cache entry to force refresh from database
        self._concept_cache.remove(concept_id)
        
        # Get updated concept
        updated_concept = self.get_concept(concept_id)
        if not updated_concept:
            raise Exception(f"Failed to retrieve updated concept: {concept_id}")
        
        # Update embedding if semantic fields changed
        if semantic_change and self.embedding_service and self.qdrant_service:
            start_time = time.time()
            try:
                self._generate_and_store_embedding(updated_concept)
                logger.debug(f"Embedding regeneration took {time.time() - start_time:.3f}s")
            except Exception as e:
                logger.error(f"Error regenerating embedding: {e}")
                # Don't fail the operation if embedding fails
        
        # Log activity
        if user_id:
            self._log_activity(
                ActivityType.UPDATE,
                user_id,
                "concept",
                concept_id,
                {"fields_updated": list(update_dict.keys())}
            )
        
        # Update cache with fresh data
        self._concept_cache.put(concept_id, updated_concept)
        
        logger.info(f"Concept updated successfully: {concept_id} - {updated_concept.name}")
        return updated_concept

    @transaction
    def delete_concept(self, concept_id: str, user_id: Optional[str] = None) -> bool:
        """Delete a concept and its relationships.
        
        Args:
            concept_id: ID of the concept to delete
            user_id: Optional ID of the user performing the deletion
            
        Returns:
            True if deletion was successful, False otherwise
            
        Raises:
            Exception: If deletion fails in the primary database
        """
        if not concept_id:
            return False
            
        logger.info(f"Deleting concept: {concept_id}")
        
        # Get concept for activity logging
        concept = self.get_concept(concept_id)
        if not concept:
            logger.error(f"Concept not found: {concept_id}")
            return False
        
        # Get the relationships to delete
        relationships = self.get_concept_relationships(concept_id)
        
        # Check if MongoDB service is available
        if not self.mongo_service:
            raise Exception("MongoDB service not available")
        
        # Delete from MongoDB
        start_time = time.time()
        try:
            if not self.mongo_service.delete_concept(concept_id):
                raise Exception(f"Failed to delete concept from MongoDB: {concept_id}")
            logger.debug(f"MongoDB delete concept took {time.time() - start_time:.3f}s")
        except Exception as e:
            logger.error(f"Error deleting concept from MongoDB: {e}")
            raise
        
        # Delete from Neo4j
        if self.neo4j_service:
            start_time = time.time()
            try:
                neo4j_success = self.neo4j_service.delete_concept(concept_id)
                if not neo4j_success:
                    logger.warning(f"Failed to delete concept from Neo4j: {concept_id}")
                logger.debug(f"Neo4j delete concept took {time.time() - start_time:.3f}s")
            except Exception as e:
                logger.error(f"Error deleting concept from Neo4j: {e}")
                # Don't fail the operation if Neo4j fails
        
        # Delete from Qdrant
        if self.qdrant_service:
            start_time = time.time()
            try:
                qdrant_success = self.qdrant_service.delete_embedding(concept_id)
                if not qdrant_success:
                    logger.warning(f"Failed to delete concept embedding from Qdrant: {concept_id}")
                logger.debug(f"Qdrant delete embedding took {time.time() - start_time:.3f}s")
            except Exception as e:
                logger.error(f"Error deleting concept embedding from Qdrant: {e}")
                # Don't fail the operation if Qdrant fails
        
        # Log activity
        if user_id:
            self._log_activity(
                ActivityType.DELETE,
                user_id,
                "concept",
                concept_id,
                {"name": concept.name, "concept_type": concept.concept_type.value}
            )
        
        # Clean caches
        self._concept_cache.remove(concept_id)
        
        # Clean relationship cache for related relationships
        for rel in relationships:
            if isinstance(rel, dict) and "id" in rel:
                self._relationship_cache.remove(rel["id"])
        
        logger.info(f"Concept deleted successfully: {concept_id} - {concept.name}")
        return True
    
    def list_concepts(self, skip: int = 0, limit: int = 100, 
                      filters: Optional[Dict[str, Any]] = None, 
                      sort_by: str = "created_at", 
                      sort_desc: bool = True) -> List[Concept]:
        """List concepts with filtering, pagination and sorting.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            filters: Optional filters to apply
            sort_by: Field to sort by
            sort_desc: Whether to sort in descending order
            
        Returns:
            List of matching concepts
        """
        logger.info(f"Listing concepts: skip={skip}, limit={limit}, filters={filters}")
        
        # Input validation
        if skip < 0:
            raise ValueError("Skip value cannot be negative")
        if limit < 1:
            raise ValueError("Limit value must be at least 1")
        if limit > 1000:
            raise ValueError("Limit value cannot exceed 1000")
        
        # Check if MongoDB service is available
        if not self.mongo_service:
            logger.error("MongoDB service not available")
            return []
        
        # Get from MongoDB
        start_time = time.time()
        try:
            concept_dicts = self.mongo_service.list_concepts(
                skip=skip,
                limit=limit,
                filters=filters,
                sort_by=sort_by,
                sort_desc=sort_desc
            )
            logger.debug(f"MongoDB list concepts took {time.time() - start_time:.3f}s")
        except Exception as e:
            logger.error(f"Error listing concepts from MongoDB: {e}")
            return []
        
        # Convert to Concept objects
        concepts = []
        for concept_dict in concept_dicts:
            try:
                concept = Concept(**concept_dict)
                concepts.append(concept)
                
                # Update cache
                self._concept_cache.put(concept.id, concept)
            except Exception as e:
                logger.error(f"Error parsing concept data: {e}")
        
        logger.info(f"Listed {len(concepts)} concepts")
        return concepts
    
    def search_concepts(self, query: str, limit: int = 10) -> List[Concept]:
        """Search concepts by text query.
        
        Args:
            query: Text to search for
            limit: Maximum number of results to return
            
        Returns:
            List of matching concepts
        """
        logger.info(f"Searching concepts: query='{query}', limit={limit}")
        
        if not query or not query.strip():
            return []
            
        if limit < 1:
            raise ValueError("Limit value must be at least 1")
        
        # Check if MongoDB service is available
        if not self.mongo_service:
            logger.error("MongoDB service not available")
            return []
        
        # Text search in MongoDB
        start_time = time.time()
        try:
            concept_dicts = self.mongo_service.search_concepts(query, limit)
            logger.debug(f"MongoDB search concepts took {time.time() - start_time:.3f}s")
        except Exception as e:
            logger.error(f"Error searching concepts in MongoDB: {e}")
            return []
        
        # Convert to Concept objects
        concepts = []
        for concept_dict in concept_dicts:
            try:
                concept = Concept(**concept_dict)
                concepts.append(concept)
                
                # Update cache
                self._concept_cache.put(concept.id, concept)
            except Exception as e:
                logger.error(f"Error parsing concept data: {e}")
        
        logger.info(f"Found {len(concepts)} concepts matching query '{query}'")
        return concepts

    def semantic_search(self, query: str, limit: int = 10, 
                        concept_types: Optional[List[ConceptType]] = None) -> List[ConceptSimilarityResult]:
        """Search concepts semantically using vector similarity.
        
        Args:
            query: Text query to search for
            limit: Maximum number of results to return
            concept_types: Optional list of concept types to filter by
            
        Returns:
            List of concept similarity results
        """
        logger.info(f"Semantic search: query='{query}', limit={limit}, concept_types={concept_types}")
        
        if not query or not query.strip():
            return []
            
        if limit < 1:
            raise ValueError("Limit value must be at least 1")
        
        # Check if required services are available
        if not self.embedding_service:
            logger.error("Embedding service not available")
            return []
            
        if not self.qdrant_service:
            logger.error("Qdrant service not available")
            return []
        
        # Generate query embedding
        start_time = time.time()
        try:
            query_embedding = self.embedding_service.generate_embedding(query)
            if not query_embedding:
                logger.error("Failed to generate embedding for query")
                return []
            logger.debug(f"Query embedding generation took {time.time() - start_time:.3f}s")
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return []
        
        # Search with Qdrant
        start_time = time.time()
        try:
            similarity_results = self.qdrant_service.search_similar(
                embedding=query_embedding,
                limit=limit,
                concept_types=concept_types
            )
            logger.debug(f"Qdrant search took {time.time() - start_time:.3f}s")
        except Exception as e:
            logger.error(f"Error performing semantic search in Qdrant: {e}")
            return []
        
        if not similarity_results:
            logger.info(f"No results found for semantic search: '{query}'")
            return []
        
        # Format and return results
        formatted_results = self._format_similarity_results(similarity_results)
        logger.info(f"Found {len(formatted_results)} results for semantic search: '{query}'")
        return formatted_results

    def get_concept_with_relationships(self, concept_id: str) -> ConceptWithRelationships:
        """Get a concept with all its relationships.
        
        Args:
            concept_id: ID of the concept
            
        Returns:
            ConceptWithRelationships object containing the concept and its relationships
            
        Raises:
            Exception: If the concept is not found
        """
        logger.info(f"Getting concept with relationships: {concept_id}")
        
        if not concept_id:
            raise ValueError("Concept ID cannot be empty")
            
        # Get the concept
        concept = self.get_concept(concept_id)
        if not concept:
            raise Exception(f"Concept not found: {concept_id}")
        
        # Check if Neo4j service is available
        if not self.neo4j_service:
            logger.error("Neo4j service not available")
            return ConceptWithRelationships(
                concept=concept,
                incoming_relationships=[],
                outgoing_relationships=[]
            )
        
        # Get incoming and outgoing relationships
        start_time = time.time()
        try:
            outgoing = self.neo4j_service.get_concept_relationships(concept_id, direction="outgoing")
            incoming = self.neo4j_service.get_concept_relationships(concept_id, direction="incoming")
            logger.debug(f"Neo4j get relationships took {time.time() - start_time:.3f}s")
        except Exception as e:
            logger.error(f"Error getting relationships from Neo4j: {e}")
            outgoing = []
            incoming = []
        
        # Ensure relationships have consistent format
        outgoing_relationships = self._normalize_relationships(outgoing)
        incoming_relationships = self._normalize_relationships(incoming)
        
        logger.info(f"Retrieved concept with {len(incoming_relationships)} incoming and {len(outgoing_relationships)} outgoing relationships")
        return ConceptWithRelationships(
            concept=concept,
            incoming_relationships=incoming_relationships,
            outgoing_relationships=outgoing_relationships
        )

    def get_concept_graph(self, concept_id: str, depth: int = 1, 
                          relationship_types: Optional[List[RelationshipType]] = None) -> Dict[str, Any]:
        """Get a subgraph centered on a concept.
        
        Args:
            concept_id: ID of the concept
            depth: Depth of traversal from the center concept
            relationship_types: Optional list of relationship types to include
            
        Returns:
            Dictionary with nodes and edges of the graph
            
        Raises:
            ValueError: If input validation fails
        """
        logger.info(f"Getting concept graph: concept_id={concept_id}, depth={depth}")
        
        if not concept_id:
            raise ValueError("Concept ID cannot be empty")
            
        if depth < 1:
            raise ValueError("Depth must be at least 1")
            
        if depth > 3:
            logger.warning(f"Large graph depth requested: {depth}. This may impact performance.")
        
        # Check if Neo4j service is available
        if not self.neo4j_service:
            logger.error("Neo4j service not available")
            return {"nodes": [], "edges": []}
        
        # Convert relationship_types to strings if provided
        relationship_type_strs = None
        if relationship_types:
            relationship_type_strs = [rt.value for rt in relationship_types]
        
        # Get graph from Neo4j
        start_time = time.time()
        try:
            graph_data = self.neo4j_service.get_concept_graph(
                concept_id=concept_id,
                depth=depth,
                relationship_types=relationship_type_strs
            )
            logger.debug(f"Neo4j get concept graph took {time.time() - start_time:.3f}s")
        except Exception as e:
            logger.error(f"Error getting concept graph from Neo4j: {e}")
            return {"nodes": [], "edges": []}
        
        # Ensure the response has the expected format with 'nodes' and 'edges'
        if not graph_data:
            return {"nodes": [], "edges": []}
            
        if "nodes" not in graph_data or "edges" not in graph_data:
            # Transform the data to the expected format if necessary
            nodes = graph_data.get("nodes", []) or graph_data.get("concepts", [])
            relationships = graph_data.get("relationships", []) or graph_data.get("edges", [])
            
            # Format data as expected by client
            formatted_graph = {
                "nodes": nodes,
                "edges": relationships
            }
            
            graph_data = formatted_graph
        
        # Convert nodes to proper Concept objects before caching
        for node in graph_data.get("nodes", []):
            try:
                if isinstance(node, dict) and "id" in node:
                    concept_id = node["id"]
                    # Only convert to Concept and cache if it's not already a proper Concept object
                    if not any(field not in node for field in ["name", "description", "concept_type"]):
                        # Ensure concept_type is an enum
                        if isinstance(node.get("concept_type"), str):
                            try:
                                node["concept_type"] = ConceptType(node["concept_type"])
                            except ValueError:
                                node["concept_type"] = ConceptType.TOPIC
                        
                        # Create proper Concept object
                        concept = Concept(**node)
                        self._concept_cache.put(concept_id, concept)
            except Exception as e:
                logger.error(f"Error processing graph node: {e}")
        
        logger.info(f"Retrieved concept graph with {len(graph_data.get('nodes', []))} nodes and {len(graph_data.get('edges', []))} edges")
        return graph_data
    
    #---------------------------
    # Relationship Management
    #---------------------------
    
    @transaction
    def create_relationship(self, relationship_data: RelationshipCreate, 
                            user_id: Optional[str] = None) -> Relationship:
        """Create a new relationship between concepts.
        
        Args:
            relationship_data: Data for the new relationship
            user_id: Optional ID of the user creating the relationship
            
        Returns:
            The created relationship
            
        Raises:
            ValueError: If validation fails
            Exception: If creation fails or concepts don't exist
        """
        logger.info(f"Creating relationship: {relationship_data.source_id} -> {relationship_data.target_id}")
        
        # Input validation
        self._validate_relationship_create(relationship_data)
        
        # Verify source and target concepts exist
        source = self.get_concept(relationship_data.source_id)
        target = self.get_concept(relationship_data.target_id)
        
        if not source:
            raise ValueError(f"Source concept not found: {relationship_data.source_id}")
        if not target:
            raise ValueError(f"Target concept not found: {relationship_data.target_id}")
        
        # Check for self-reference
        if source.id == target.id:
            raise ValueError("Source and target concepts cannot be the same")
        
        # Generate unique ID and timestamp
        relationship_id = str(uuid.uuid4())
        now = datetime.now()
        
        # Create relationship object
        relationship = Relationship(
            id=relationship_id,
            source_id=relationship_data.source_id,
            target_id=relationship_data.target_id,
            relationship_type=relationship_data.relationship_type,
            strength=relationship_data.strength,
            description=relationship_data.description,
            bidirectional=relationship_data.bidirectional,
            metadata=relationship_data.metadata,
            created_at=now,
            updated_at=now
        )
        
        # Check if MongoDB service is available
        if not self.mongo_service:
            raise Exception("MongoDB service not available")
        
        # Store in MongoDB
        start_time = time.time()
        try:
            if not self.mongo_service.create_relationship(relationship):
                raise Exception("Failed to create relationship in MongoDB")
            logger.debug(f"MongoDB create relationship took {time.time() - start_time:.3f}s")
        except Exception as e:
            logger.error(f"Error creating relationship in MongoDB: {e}")
            raise
        
        # Store in Neo4j
        if self.neo4j_service:
            start_time = time.time()
            try:
                neo4j_id = self.neo4j_service.create_relationship(relationship)
                if not neo4j_id:
                    logger.warning(f"Failed to create relationship in Neo4j: {relationship.id}")
                logger.debug(f"Neo4j create relationship took {time.time() - start_time:.3f}s")
            except Exception as e:
                logger.error(f"Error creating relationship in Neo4j: {e}")
                # Don't fail the operation if Neo4j fails
        
        # Log activity
        if user_id:
            self._log_activity(
                ActivityType.CREATE,
                user_id,
                "relationship",
                relationship.id,
                {
                    "source_id": relationship.source_id,
                    "target_id": relationship.target_id,
                    "relationship_type": relationship.relationship_type.value
                }
            )
        
        # Add to cache
        self._relationship_cache.put(relationship.id, relationship)
        
        logger.info(f"Relationship created successfully: {relationship.id} ({relationship.relationship_type.value})")
        return relationship
    
    def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """Get a relationship by ID.
        
        Args:
            relationship_id: ID of the relationship to retrieve
            
        Returns:
            The relationship if found, None otherwise
        """
        if not relationship_id:
            return None
            
        logger.debug(f"Getting relationship: {relationship_id}")
        
        # Check cache first
        cached_relationship = self._relationship_cache.get(relationship_id)
        if cached_relationship:
            logger.debug(f"Relationship found in cache: {relationship_id}")
            return cached_relationship
        
        # Check if MongoDB service is available
        if not self.mongo_service:
            logger.error("MongoDB service not available")
            return None
        
        # Get from MongoDB
        start_time = time.time()
        try:
            relationship_data = self.mongo_service.get_relationship(relationship_id)
            logger.debug(f"MongoDB get relationship took {time.time() - start_time:.3f}s")
        except Exception as e:
            logger.error(f"Error retrieving relationship from MongoDB: {e}")
            return None
        
        if not relationship_data:
            logger.debug(f"Relationship not found: {relationship_id}")
            return None
        
        try:
            # Convert to Relationship object
            relationship = Relationship(**relationship_data)
            
            # Update cache
            self._relationship_cache.put(relationship.id, relationship)
            
            logger.debug(f"Relationship retrieved: {relationship.id}")
            return relationship
        except Exception as e:
            logger.error(f"Error parsing relationship data: {e}")
            return None

    @transaction
    def update_relationship(self, relationship_id: str, updates: RelationshipUpdate,
                            user_id: Optional[str] = None) -> Optional[Relationship]:
        """Update an existing relationship.
        
        Args:
            relationship_id: ID of the relationship to update
            updates: Data to update
            user_id: Optional ID of the user performing the update
            
        Returns:
            Updated relationship if successful, None otherwise
            
        Raises:
            ValueError: If validation fails
            Exception: If update fails in the primary database
        """
        logger.info(f"Updating relationship: {relationship_id}")
        
        # Input validation
        self._validate_relationship_update(relationship_id, updates)
        
        # Get current relationship
        current_relationship = self.get_relationship(relationship_id)
        if not current_relationship:
            logger.error(f"Relationship not found: {relationship_id}")
            return None
        
        # Prepare updates - using exclude_unset=True to only include fields actually provided
        update_dict = {k: v for k, v in updates.dict(exclude_unset=True).items() if v is not None}
        
        # If source or target is being updated, verify they exist
        if "source_id" in update_dict:
            source = self.get_concept(update_dict["source_id"])
            if not source:
                raise ValueError(f"Source concept not found: {update_dict['source_id']}")
                
        if "target_id" in update_dict:
            target = self.get_concept(update_dict["target_id"])
            if not target:
                raise ValueError(f"Target concept not found: {update_dict['target_id']}")
        
        # Prevent self-reference
        source_id = update_dict.get("source_id", current_relationship.source_id)
        target_id = update_dict.get("target_id", current_relationship.target_id)
        if source_id == target_id:
            raise ValueError("Source and target concepts cannot be the same")
        
        # Add metadata
        update_dict["updated_at"] = datetime.now()
        
        # Check if MongoDB service is available
        if not self.mongo_service:
            raise Exception("MongoDB service not available")
        
        # Update in MongoDB
        start_time = time.time()
        try:
            if not self.mongo_service.update_relationship(relationship_id, update_dict):
                raise Exception(f"Failed to update relationship in MongoDB: {relationship_id}")
            logger.debug(f"MongoDB update relationship took {time.time() - start_time:.3f}s")
        except Exception as e:
            logger.error(f"Error updating relationship in MongoDB: {e}")
            raise
        
        # Update in Neo4j
        if self.neo4j_service:
            start_time = time.time()
            try:
                neo4j_success = self.neo4j_service.update_relationship(relationship_id, update_dict)
                if not neo4j_success:
                    logger.warning(f"Failed to update relationship in Neo4j: {relationship_id}")
                logger.debug(f"Neo4j update relationship took {time.time() - start_time:.3f}s")
            except Exception as e:
                logger.error(f"Error updating relationship in Neo4j: {e}")
                # Don't fail the operation if Neo4j fails
        
        # Clear cache entry to force refresh from database
        self._relationship_cache.remove(relationship_id)
        
        # Get updated relationship
        updated_relationship = self.get_relationship(relationship_id)
        if not updated_relationship:
            raise Exception(f"Failed to retrieve updated relationship: {relationship_id}")
        
        # Log activity
        if user_id:
            self._log_activity(
                ActivityType.UPDATE,
                user_id,
                "relationship",
                relationship_id,
                {"fields_updated": list(update_dict.keys())}
            )
        
        # Update cache with fresh data
        self._relationship_cache.put(relationship_id, updated_relationship)
        
        logger.info(f"Relationship updated successfully: {relationship_id}")
        return updated_relationship

    @transaction
    def delete_relationship(self, relationship_id: str, user_id: Optional[str] = None) -> bool:
        """Delete a relationship.
        
        Args:
            relationship_id: ID of the relationship to delete
            user_id: Optional ID of the user performing the deletion
            
        Returns:
            True if deletion was successful, False otherwise
            
        Raises:
            Exception: If deletion fails in the primary database
        """
        if not relationship_id:
            return False
            
        logger.info(f"Deleting relationship: {relationship_id}")
        
        # Get relationship for activity logging
        relationship = self.get_relationship(relationship_id)
        if not relationship:
            logger.error(f"Relationship not found: {relationship_id}")
            return False
        
        # Check if MongoDB service is available
        if not self.mongo_service:
            raise Exception("MongoDB service not available")
        
        # Delete from MongoDB
        start_time = time.time()
        try:
            if not self.mongo_service.delete_relationship(relationship_id):
                raise Exception(f"Failed to delete relationship from MongoDB: {relationship_id}")
            logger.debug(f"MongoDB delete relationship took {time.time() - start_time:.3f}s")
        except Exception as e:
            logger.error(f"Error deleting relationship from MongoDB: {e}")
            raise
        
        # Delete from Neo4j
        if self.neo4j_service:
            start_time = time.time()
            try:
                neo4j_success = self.neo4j_service.delete_relationship(relationship_id)
                if not neo4j_success:
                    logger.warning(f"Failed to delete relationship from Neo4j: {relationship_id}")
                logger.debug(f"Neo4j delete relationship took {time.time() - start_time:.3f}s")
            except Exception as e:
                logger.error(f"Error deleting relationship from Neo4j: {e}")
                # Don't fail the operation if Neo4j fails
        
        # Log activity
        if user_id:
            self._log_activity(
                ActivityType.DELETE,
                user_id,
                "relationship",
                relationship_id,
                {
                    "source_id": relationship.source_id,
                    "target_id": relationship.target_id,
                    "relationship_type": relationship.relationship_type.value
                }
            )
        
        # Remove from cache
        self._relationship_cache.remove(relationship_id)
        
        logger.info(f"Relationship deleted successfully: {relationship_id}")
        return True
    
    def get_concept_relationships(self, concept_id: str, direction: str = "both",
                                  relationship_types: Optional[List[RelationshipType]] = None) -> List[Dict[str, Any]]:
        """Get relationships for a concept with optional filtering.
        
        Args:
            concept_id: ID of the concept
            direction: One of "incoming", "outgoing", or "both"
            relationship_types: Optional list of relationship types to filter by
            
        Returns:
            List of relationships
            
        Raises:
            ValueError: If validation fails
        """
        logger.info(f"Getting relationships for concept: {concept_id}, direction={direction}")
        
        if not concept_id:
            raise ValueError("Concept ID cannot be empty")
            
        if direction not in ["incoming", "outgoing", "both"]:
            raise ValueError("Direction must be one of: incoming, outgoing, both")
        
        # Check if Neo4j service is available
        if not self.neo4j_service:
            logger.error("Neo4j service not available")
            return []
        
        # Convert relationship_types to strings if provided
        relationship_type_strs = None
        if relationship_types:
            relationship_type_strs = [rt.value for rt in relationship_types]
        
        # Get from Neo4j (more efficient for graph operations)
        start_time = time.time()
        try:
            relationships = self.neo4j_service.get_concept_relationships(
                concept_id=concept_id,
                direction=direction,
                relationship_types=relationship_type_strs
            )
            logger.debug(f"Neo4j get concept relationships took {time.time() - start_time:.3f}s")
        except Exception as e:
            logger.error(f"Error getting concept relationships from Neo4j: {e}")
            return []
        
        # Normalize relationship format
        normalized_relationships = self._normalize_relationships(relationships)
        logger.info(f"Retrieved {len(normalized_relationships)} relationships for concept {concept_id}")
        return normalized_relationships
    
    #---------------------------
    # Learning Path Management
    #---------------------------
    
    @transaction
    def create_learning_path(self, request: LearningPathRequest, 
                             user_id: Optional[str] = None) -> LearningPath:
        """Create a learning path based on a request.
        
        Args:
            request: Data for the learning path request
            user_id: Optional ID of the user creating the path
            
        Returns:
            Generated learning path
            
        Raises:
            ValueError: If validation fails
            Exception: If path creation fails
        """
        logger.info(f"Creating learning path for goal: {request.goal}")
        
        # Input validation
        self._validate_learning_path_request(request)
        
        # Collect concepts for the path
        concepts = []
        
        if request.concept_ids:
            # Use specified concepts
            for concept_id in request.concept_ids:
                concept = self.get_concept(concept_id)
                if concept:
                    concepts.append(concept.dict())
        elif request.domain_id:
            # Get all concepts in the domain
            domain_concepts = self.list_concepts(
                limit=100,
                filters={"parent_id": request.domain_id}
            )
            concepts = [c.dict() for c in domain_concepts]
        
        if not concepts:
            raise ValueError("No concepts available for learning path")
        
        # Check if LLM service is available
        if not self.llm_service:
            raise Exception("LLM service not available")
        
        # Generate path using LLM
        start_time = time.time()
        try:
            path_data = self.llm_service.generate_learning_path(
                goal=request.goal,
                concepts=concepts,
                level=request.learner_level.value
            )
            logger.debug(f"LLM path generation took {time.time() - start_time:.3f}s")
        except Exception as e:
            logger.error(f"Error generating learning path with LLM: {e}")
            raise Exception(f"Failed to generate learning path: {str(e)}")
        
        if not path_data or not isinstance(path_data, dict) or "path" not in path_data:
            raise Exception("Failed to generate valid learning path")
        
        # Validate the generated path
        if not path_data.get("path"):
            raise Exception("Generated learning path contains no steps")
        
        # Create learning path object
        path_id = str(uuid.uuid4())
        now = datetime.now()
        
        # Ensure each step has required fields
        validated_steps = []
        for step in path_data["path"]:
            if not isinstance(step, dict):
                continue
            
            # Ensure each step has a concept_id
            if "concept_id" not in step:
                continue
                
            # Validate concept exists
            concept = self.get_concept(step["concept_id"])
            if not concept:
                continue
                
            # Add concept name if missing
            if "concept_name" not in step:
                step["concept_name"] = concept.name
                
            validated_steps.append(step)
        
        # Create learning path with validated steps
        learning_path = LearningPath(
            id=path_id,
            name=f"Path: {request.goal}",
            description=f"Learning path for: {request.goal}",
            goal=request.goal,
            target_learner_level=request.learner_level,
            concepts=[step["concept_id"] for step in validated_steps],
            steps=validated_steps,
            total_time_minutes=path_data.get("total_time_minutes", 0),
            created_at=now,
            updated_at=now,
            metadata={
                "user_id": user_id,
                "max_time_minutes": request.max_time_minutes,
                "prior_knowledge": request.prior_knowledge
            }
        )
        
        # Check if MongoDB service is available
        if not self.mongo_service:
            raise Exception("MongoDB service not available")
        
        # Store in MongoDB
        start_time = time.time()
        try:
            if not self.mongo_service.create_learning_path(learning_path):
                raise Exception("Failed to create learning path in MongoDB")
            logger.debug(f"MongoDB create learning path took {time.time() - start_time:.3f}s")
        except Exception as e:
            logger.error(f"Error creating learning path in MongoDB: {e}")
            raise
        
        # Log activity
        if user_id:
            self._log_activity(
                ActivityType.CREATE,
                user_id,
                "learning_path",
                learning_path.id,
                {"goal": request.goal, "concepts": len(concepts)}
            )
        
        logger.info(f"Learning path created successfully: {learning_path.id} with {len(validated_steps)} steps")
        return learning_path
    
    def get_learning_path(self, path_id: str) -> Optional[LearningPath]:
        """Get a learning path by ID.
        
        Args:
            path_id: ID of the learning path to retrieve
            
        Returns:
            The learning path if found, None otherwise
        """
        if not path_id:
            return None
            
        logger.info(f"Getting learning path: {path_id}")
        
        # Check if MongoDB service is available
        if not self.mongo_service:
            logger.error("MongoDB service not available")
            return None
        
        start_time = time.time()
        try:
            path_data = self.mongo_service.get_learning_path(path_id)
            logger.debug(f"MongoDB get learning path took {time.time() - start_time:.3f}s")
        except Exception as e:
            logger.error(f"Error retrieving learning path from MongoDB: {e}")
            return None
        
        if not path_data:
            logger.debug(f"Learning path not found: {path_id}")
            return None
        
        try:
            learning_path = LearningPath(**path_data)
            logger.debug(f"Learning path retrieved: {path_id}")
            return learning_path
        except Exception as e:
            logger.error(f"Error parsing learning path data: {e}")
            return None
    
    def list_learning_paths(self, skip: int = 0, limit: int = 20,
                            filters: Optional[Dict[str, Any]] = None) -> List[LearningPath]:
        """List learning paths with filtering and pagination.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            filters: Optional filters to apply
            
        Returns:
            List of learning paths
        """
        logger.info(f"Listing learning paths: skip={skip}, limit={limit}, filters={filters}")
        
        # Input validation
        if skip < 0:
            raise ValueError("Skip value cannot be negative")
        if limit < 1:
            raise ValueError("Limit value must be at least 1")
        if limit > 100:
            raise ValueError("Limit value cannot exceed 100")
        
        # Check if MongoDB service is available
        if not self.mongo_service:
            logger.error("MongoDB service not available")
            return []
        
        start_time = time.time()
        try:
            path_dicts = self.mongo_service.list_learning_paths(
                skip=skip,
                limit=limit,
                filters=filters
            )
            logger.debug(f"MongoDB list learning paths took {time.time() - start_time:.3f}s")
        except Exception as e:
            logger.error(f"Error listing learning paths from MongoDB: {e}")
            return []
        
        paths = []
        for path_dict in path_dicts:
            try:
                path = LearningPath(**path_dict)
                paths.append(path)
            except Exception as e:
                logger.error(f"Error parsing learning path data: {e}")
        
        logger.info(f"Listed {len(paths)} learning paths")
        return paths
    
    #---------------------------
    # Domain Management
    #---------------------------

    @transaction
    def create_domain(self, request: DomainStructureRequest, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a complete domain structure with concepts and relationships.
        
        Args:
            request: Request containing domain details
            user_id: Optional ID of the user creating the domain
            
        Returns:
            Dictionary with creation results
            
        Raises:
            ValueError: If validation fails
        """
        logger.info(f"Creating domain structure: {request.domain_name}")
        
        # Input validation
        self._validate_domain_structure_request(request)
        
        # Check if LLM service is available
        if not self.llm_service:
            raise Exception("LLM service not available")
        
        try:
            # Use LLM service to generate domain structure.
            start_time = time.time()
            try:
                domain_data = self.llm_service.generate_complete_domain(request)
                logger.debug(f"LLM domain generation took {time.time() - start_time:.3f}s")
            except Exception as e:
                logger.error(f"Error generating domain with LLM: {e}")
                raise Exception(f"LLM domain generation failed: {str(e)}")
            
            if not domain_data or not isinstance(domain_data, dict):
                raise Exception("LLM returned invalid domain data")
                
            if not domain_data.get("success", False):
                # If LLM generation fails, raise an exception.
                raise Exception("LLM domain generation failed: " + domain_data.get("error", "Unknown error"))
            
            # Validate domain data structure
            if "concepts" not in domain_data or not domain_data["concepts"]:
                raise ValueError("Generated domain must contain concepts")
            
            # Use a transaction-like approach to ensure all-or-nothing
            created_domain = None
            created_concepts = []
            created_relationships = []
            concept_id_map = {}  # Map from generated ID (from LLM) to actual DB IDs.
            
            try:
                # Create the main domain concept.
                domain_concept = ConceptCreate(
                    name=request.domain_name,
                    description=request.domain_description,
                    concept_type=ConceptType.DOMAIN,
                    difficulty=request.difficulty_level or DifficultyLevel.INTERMEDIATE,
                    importance=1.0,
                    keywords=request.key_topics or [],
                    metadata=request.metadata
                )
                domain = self.create_concept(domain_concept, user_id)
                if not domain:
                    raise Exception("Failed to create domain concept")
                    
                created_domain = domain
                
                # Create child concepts.
                for cdata in domain_data.get("concepts", []):
                    # Validate concept data
                    if not isinstance(cdata, dict) or "name" not in cdata or "id" not in cdata:
                        logger.warning(f"Skipping invalid concept data: {cdata}")
                        continue
                    
                    try:
                        # Convert difficulty string to enum if needed
                        difficulty = cdata.get("difficulty", "intermediate")
                        if isinstance(difficulty, str):
                            try:
                                difficulty = DifficultyLevel(difficulty)
                            except ValueError:
                                difficulty = DifficultyLevel.INTERMEDIATE
                        
                        concept_create = ConceptCreate(
                            name=cdata["name"],
                            description=cdata.get("description", f"Concept for {cdata['name']}"),
                            concept_type=ConceptType.TOPIC,  # Default to topic.
                            difficulty=difficulty,
                            parent_id=domain.id,
                            importance=float(cdata.get("importance", 0.5)),
                            complexity=float(cdata.get("complexity", 0.5)),
                            keywords=cdata.get("keywords", []),
                            estimated_learning_time_minutes=int(cdata.get("estimated_learning_time_minutes", 30)),
                            metadata={
                                "original_id": cdata["id"],
                                "prerequisites": cdata.get("prerequisites", [])
                            }
                        )
                        created_concept = self.create_concept(concept_create, user_id)
                        concept_id_map[cdata["id"]] = created_concept.id
                        created_concepts.append(created_concept)
                    except Exception as ex:
                        logger.error(f"Error creating child concept '{cdata.get('name')}': {ex}")
                        raise  # Re-raise to trigger rollback
                
                # Create relationships between child concepts.
                for rel_data in domain_data.get("relationships", []):
                    if not isinstance(rel_data, dict) or "source_id" not in rel_data or "target_id" not in rel_data:
                        logger.warning(f"Skipping invalid relationship data: {rel_data}")
                        continue
                        
                    source_id = concept_id_map.get(rel_data["source_id"])
                    target_id = concept_id_map.get(rel_data["target_id"])
                    if not source_id or not target_id:
                        logger.warning("Skipping relationship: missing source or target ID")
                        continue
                        
                    # Skip self-relationships
                    if source_id == target_id:
                        logger.warning(f"Skipping self-relationship for concept: {source_id}")
                        continue
                        
                    try:
                        # Convert relationship_type string to enum
                        rel_type = rel_data.get("relationship_type", "related_to")
                        if isinstance(rel_type, str):
                            try:
                                rel_type = RelationshipType(rel_type)
                            except ValueError:
                                rel_type = RelationshipType.RELATED_TO
                                
                        rel_create = RelationshipCreate(
                            source_id=source_id,
                            target_id=target_id,
                            relationship_type=rel_type,
                            strength=float(rel_data.get("strength", 0.5)),
                            description=rel_data.get("description", ""),
                            bidirectional=bool(rel_data.get("bidirectional", False))
                        )
                        created_rel = self.create_relationship(rel_create, user_id)
                        created_relationships.append(created_rel)
                    except Exception as ex:
                        logger.error(f"Error creating relationship between {source_id} and {target_id}: {ex}")
                        raise  # Re-raise to trigger rollback
                
                # Create prerequisite relationships based on metadata.
                for concept in created_concepts:
                    prereqs = concept.metadata.get("prerequisites", []) if concept.metadata else []
                    for prereq_id in prereqs:
                        mapped_prereq_id = concept_id_map.get(prereq_id)
                        if mapped_prereq_id and mapped_prereq_id != concept.id:
                            # Check if relationship already exists
                            existing_rels = self.get_concept_relationships(
                                concept.id, 
                                direction="incoming",
                                relationship_types=[RelationshipType.PREREQUISITE]
                            )
                            existing_source_ids = [r.get("source_id") for r in existing_rels if r.get("source_id")]
                            if mapped_prereq_id not in existing_source_ids:
                                prereq_rel = RelationshipCreate(
                                    source_id=mapped_prereq_id,
                                    target_id=concept.id,
                                    relationship_type=RelationshipType.PREREQUISITE,
                                    strength=0.8
                                )
                                new_rel = self.create_relationship(prereq_rel, user_id)
                                created_relationships.append(new_rel)
                
                # Log activity
                if user_id:
                    self._log_activity(
                        ActivityType.CREATE,
                        user_id,
                        "domain",
                        domain.id,
                        {"name": domain.name, "concepts": len(created_concepts), "relationships": len(created_relationships)}
                    )
                
                logger.info(f"Domain created successfully: {domain.id} with {len(created_concepts)} concepts and {len(created_relationships)} relationships")
                return {
                    "success": True,
                    "domain_id": domain.id,
                    "concepts": len(created_concepts),
                    "relationships": len(created_relationships),
                    "domain": domain.dict()
                }
                
            except Exception as creation_error:
                # If we get here, something failed during creation
                # The @transaction decorator will handle rollback
                logger.error(f"Domain creation failed: {creation_error}")
                raise Exception(f"Domain creation failed during entity creation: {str(creation_error)}")
        
        except Exception as e:
            logger.error(f"Error creating domain: {e}")
            return {
                "success": False,
                "error": f"Domain creation failed: {str(e)}"
            }

    def get_domain_structure(self, domain_id: str, include_relationships: bool = True) -> Dict[str, Any]:
        """Get the structure of a domain including its concepts and their hierarchy.
        
        Args:
            domain_id: ID of the domain
            include_relationships: Whether to include relationships
            
        Returns:
            Dictionary with domain, concepts, and relationships
        """
        logger.info(f"Getting domain structure: {domain_id}")
        
        if not domain_id:
            raise ValueError("Domain ID cannot be empty")
            
        # Check if Neo4j service is available
        if not self.neo4j_service:
            logger.error("Neo4j service not available")
            # Fallback to MongoDB if Neo4j is not available
            return self._get_domain_structure_from_mongodb(domain_id, include_relationships)
        
        start_time = time.time()
        try:
            domain_structure = self.neo4j_service.get_domain_structure(
                domain_id=domain_id,
                include_relationships=include_relationships
            )
            logger.debug(f"Neo4j get domain structure took {time.time() - start_time:.3f}s")
        except Exception as e:
            logger.error(f"Error getting domain structure from Neo4j: {e}")
            # Fallback to MongoDB if Neo4j fails
            return self._get_domain_structure_from_mongodb(domain_id, include_relationships)
        
        # Ensure required keys exist in response
        if not domain_structure:
            return {"domain": None, "concepts": [], "relationships": []}
        
        if "domain" not in domain_structure:
            domain_concept = self.get_concept(domain_id)
            if domain_concept:
                domain_structure["domain"] = domain_concept.dict()
            else:
                domain_structure["domain"] = None
                
        if "concepts" not in domain_structure:
            domain_structure["concepts"] = []
            
        if "relationships" not in domain_structure:
            domain_structure["relationships"] = []
        
        # Add concepts to cache
        for concept_data in domain_structure.get("concepts", []):
            try:
                if concept_data and isinstance(concept_data, dict) and "id" in concept_data:
                    # Convert concept_type to enum if it's a string
                    if "concept_type" in concept_data and isinstance(concept_data["concept_type"], str):
                        try:
                            concept_data["concept_type"] = ConceptType(concept_data["concept_type"])
                        except ValueError:
                            concept_data["concept_type"] = ConceptType.TOPIC
                    
                    concept = Concept(**concept_data)
                    self._concept_cache.put(concept.id, concept)
            except Exception as e:
                logger.error(f"Error parsing concept data for caching: {e}")
        
        logger.info(f"Retrieved domain structure with {len(domain_structure.get('concepts', []))} concepts and {len(domain_structure.get('relationships', []))} relationships")
        return domain_structure
    
    def _get_domain_structure_from_mongodb(self, domain_id: str, include_relationships: bool = True) -> Dict[str, Any]:
        """Fallback method to get domain structure from MongoDB.
        
        Args:
            domain_id: ID of the domain
            include_relationships: Whether to include relationships
            
        Returns:
            Dictionary with domain, concepts, and relationships
        """
        logger.info(f"Getting domain structure from MongoDB: {domain_id}")
        
        # Check if MongoDB service is available
        if not self.mongo_service:
            logger.error("MongoDB service not available")
            return {"domain": None, "concepts": [], "relationships": []}
        
        # Get domain concept
        domain = self.get_concept(domain_id)
        if not domain:
            return {"domain": None, "concepts": [], "relationships": []}
        
        # Get child concepts
        concepts = self.list_concepts(
            limit=1000,
            filters={"parent_id": domain_id}
        )
        
        # Convert concepts to dicts
        concept_dicts = [c.dict() for c in concepts]
        
        # Add domain to concepts list
        concept_dicts.append(domain.dict())
        
        # Get relationships if needed
        relationships = []
        if include_relationships:
            # Get all concept IDs
            concept_ids = [domain_id] + [c.id for c in concepts]
            
            # Get relationships for each concept
            for cid in concept_ids:
                rel_data = self.get_concept_relationships(cid)
                if rel_data:
                    relationships.extend(rel_data)
            
            # Deduplicate relationships
            seen_rel_ids = set()
            unique_relationships = []
            for rel in relationships:
                rel_id = rel.get("id")
                if rel_id and rel_id not in seen_rel_ids:
                    seen_rel_ids.add(rel_id)
                    unique_relationships.append(rel)
            
            relationships = unique_relationships
        
        logger.info(f"Retrieved domain structure from MongoDB with {len(concept_dicts)} concepts and {len(relationships)} relationships")
        return {
            "domain": domain.dict(),
            "concepts": concept_dicts,
            "relationships": relationships
        }
    
    def validate_domain(self, domain_id: str) -> ValidationResult:
        """Validate a domain for consistency and quality issues.
        
        Args:
            domain_id: ID of the domain to validate
            
        Returns:
            ValidationResult with issues and warnings
            
        Raises:
            ValueError: If domain not found
            Exception: If validation fails
        """
        logger.info(f"Validating domain: {domain_id}")
        
        if not domain_id:
            raise ValueError("Domain ID cannot be empty")
            
        # Get domain structure
        domain_data = self.get_domain_structure(domain_id)
        
        if not domain_data or not domain_data.get("domain"):
            raise ValueError(f"Domain not found: {domain_id}")
        
        # Check if Neo4j service is available
        if not self.neo4j_service:
            logger.error("Neo4j service not available")
            return ValidationResult(
                valid=False,
                issues=[{
                    "issue_type": "service_unavailable",
                    "severity": "high",
                    "description": "Neo4j service is not available for validation",
                    "concepts_involved": []
                }],
                warnings=[],
                stats={"concepts": len(domain_data.get("concepts", [])), "relationships": len(domain_data.get("relationships", []))}
            )
        
        # Use Neo4j validation
        start_time = time.time()
        try:
            validation_result = self.neo4j_service.validate_knowledge_graph(domain_id)
            logger.debug(f"Neo4j validate domain took {time.time() - start_time:.3f}s")
        except Exception as e:
            logger.error(f"Error validating domain with Neo4j: {e}")
            # Provide a default validation result if Neo4j validation fails
            return ValidationResult(
                valid=False,
                issues=[{
                    "severity": "error", 
                    "message": f"Failed to validate domain: {str(e)}",
                    "issue_type": "validation_error",
                    "concepts_involved": []
                }],
                warnings=[],
                stats={"concepts": len(domain_data.get("concepts", [])), "relationships": len(domain_data.get("relationships", []))}
            )
        
        if not validation_result:
            # Provide a default validation result if none returned
            return ValidationResult(
                valid=False,
                issues=[{
                    "severity": "error", 
                    "message": "Failed to validate domain", 
                    "issue_type": "validation_error",
                    "concepts_involved": []
                }],
                warnings=[],
                stats={"concepts": len(domain_data.get("concepts", [])), "relationships": len(domain_data.get("relationships", []))}
            )
        
        logger.info(f"Domain validation completed: valid={validation_result.valid}, issues={len(validation_result.issues)}, warnings={len(validation_result.warnings)}")
        return validation_result
    
    #---------------------------
    # Helper Methods
    #---------------------------
    
    def _generate_and_store_embedding(self, concept: Concept) -> bool:
        """Generate and store embedding for a concept.
        
        Args:
            concept: The concept to generate embedding for
            
        Returns:
            True if successful, False otherwise
        """
        if not concept or not concept.id:
            return False
            
        logger.debug(f"Generating embedding for concept: {concept.id}")
        
        # Check if required services are available
        if not self.embedding_service:
            logger.error("Embedding service not available")
            return False
            
        if not self.qdrant_service:
            logger.error("Qdrant service not available")
            return False
        
        # Generate embedding
        try:
            embedding = self.embedding_service.generate_concept_embedding(concept)
            if not embedding:
                logger.error(f"Failed to generate embedding for concept: {concept.id}")
                return False
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return False
        
        # Prepare metadata for Qdrant
        metadata = {
            "name": concept.name,
            "concept_type": concept.concept_type.value,
            "difficulty": concept.difficulty.value,
            "importance": concept.importance,
            "description": concept.description[:200] if concept.description else ""  # Truncate description
        }
        
        # Store in Qdrant
        try:
            success = self.qdrant_service.store_embedding(concept.id, embedding, metadata)
            if not success:
                logger.error(f"Failed to store embedding for concept: {concept.id}")
                return False
        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
            return False
        
        # Update concept with embedding_id in MongoDB
        if self.mongo_service:
            try:
                self.mongo_service.update_concept(concept.id, {"embedding_id": concept.id})
            except Exception as e:
                logger.error(f"Error updating concept with embedding ID: {e}")
                # Don't fail the operation if this fails
        
        logger.debug(f"Embedding generated and stored for concept: {concept.id}")
        return True
    
    def _log_activity(self, activity_type: ActivityType, user_id: str, 
                      entity_type: str, entity_id: str, details: Dict[str, Any]) -> bool:
        """Log user activity.
        
        Args:
            activity_type: Type of activity
            user_id: ID of the user
            entity_type: Type of entity (concept, relationship, etc.)
            entity_id: ID of the entity
            details: Additional details about the activity
            
        Returns:
            True if successful, False otherwise
        """
        if not user_id or not entity_id:
            return False
            
        logger.debug(f"Logging activity: {activity_type.value} on {entity_type} {entity_id} by user {user_id}")
        
        # Check if MongoDB service is available
        if not self.mongo_service:
            logger.error("MongoDB service not available for activity logging")
            return False
        
        try:
            activity = Activity(
                id=str(uuid.uuid4()),
                activity_type=activity_type,
                user_id=user_id,
                entity_type=entity_type,
                entity_id=entity_id,
                details=details,
                timestamp=datetime.now()
            )
            
            return self.mongo_service.create_activity(activity)
        except Exception as e:
            logger.error(f"Error logging activity: {e}")
            return False
    
    def get_similar_concepts(self, concept_id: str, limit: int = 10) -> List[ConceptSimilarityResult]:
        """Find concepts similar to a given concept.
        
        Args:
            concept_id: ID of the concept to find similar concepts for
            limit: Maximum number of results to return
            
        Returns:
            List of similar concepts with similarity scores
            
        Raises:
            ValueError: If validation fails
        """
        logger.info(f"Getting similar concepts for: {concept_id}, limit={limit}")
        
        if not concept_id:
            raise ValueError("Concept ID cannot be empty")
            
        if limit < 1:
            raise ValueError("Limit must be at least 1")
        
        # Get concept
        concept = self.get_concept(concept_id)
        if not concept:
            logger.error(f"Concept not found: {concept_id}")
            return []
        
        # Check if required services are available
        if not self.qdrant_service:
            logger.error("Qdrant service not available")
            return []
            
        if not self.embedding_service:
            logger.error("Embedding service not available")
            return []
        
        # Get embedding
        start_time = time.time()
        try:
            embedding = self.qdrant_service.get_concept_vector(concept_id)
            if not embedding:
                # Generate new embedding if not found
                embedding = self.embedding_service.generate_concept_embedding(concept)
                if not embedding:
                    logger.error(f"Could not get or generate embedding for concept: {concept_id}")
                    return []
            logger.debug(f"Get concept vector took {time.time() - start_time:.3f}s")
        except Exception as e:
            logger.error(f"Error getting concept vector: {e}")
            return []
        
        # Search similar
        start_time = time.time()
        try:
            similarity_results = self.qdrant_service.get_nearest_neighbors(concept_id, limit)
            logger.debug(f"Qdrant nearest neighbors search took {time.time() - start_time:.3f}s")
        except Exception as e:
            logger.error(f"Error getting nearest neighbors: {e}")
            return []
        
        # Process and format results
        formatted_results = self._format_similarity_results(similarity_results)
        logger.info(f"Found {len(formatted_results)} similar concepts for {concept_id}")
        return formatted_results
        
    def _normalize_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize relationship data to ensure consistent format.
        
        Args:
            relationships: List of relationship dictionaries
            
        Returns:
            List of normalized relationship dictionaries
        """
        normalized = []
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
                
            # Ensure all required fields are present
            normalized_rel = {
                "id": rel.get("id", ""),
                "source_id": rel.get("source_id", ""),
                "target_id": rel.get("target_id", ""),
                "relationship_type": rel.get("relationship_type", RelationshipType.RELATED_TO.value)
            }
            
            # Convert relationship_type to string if it's an enum
            if isinstance(normalized_rel["relationship_type"], RelationshipType):
                normalized_rel["relationship_type"] = normalized_rel["relationship_type"].value
                
            # Add other fields if they exist
            if "strength" in rel:
                normalized_rel["strength"] = rel["strength"]
            if "description" in rel:
                normalized_rel["description"] = rel["description"]
            if "bidirectional" in rel:
                normalized_rel["bidirectional"] = rel["bidirectional"]
            if "created_at" in rel:
                normalized_rel["created_at"] = rel["created_at"]
                
            # Only add valid relationships (those with source and target)
            if normalized_rel["source_id"] and normalized_rel["target_id"]:
                normalized.append(normalized_rel)
                
                # Add to relationship cache if ID is present
                if normalized_rel["id"] and self.get_relationship(normalized_rel["id"]) is None:
                    try:
                        rel_obj = Relationship(**normalized_rel)
                        self._relationship_cache.put(rel_obj.id, rel_obj)
                    except Exception as e:
                        logger.error(f"Error caching relationship: {e}")
        
        return normalized
    
    def _format_similarity_results(self, results) -> List[ConceptSimilarityResult]:
        """Format similarity search results consistently.
        
        Args:
            results: Results from similarity search
            
        Returns:
            List of ConceptSimilarityResult objects
        """
        formatted_results = []
        
        if not results:
            return formatted_results
            
        for result in results:
            try:
                # Handle different possible formats
                if isinstance(result, dict):
                    concept_id = result.get("id", "") or result.get("concept_id", "")
                    concept_name = result.get("name", "Unknown") or result.get("concept_name", "Unknown")
                    similarity = float(result.get("score", 0.0) or result.get("similarity", 0.0))
                    
                    concept_type_val = result.get("concept_type", ConceptType.TOPIC.value)
                    if isinstance(concept_type_val, str):
                        try:
                            concept_type = ConceptType(concept_type_val)
                        except ValueError:
                            concept_type = ConceptType.TOPIC
                    else:
                        concept_type = concept_type_val
                        
                else:  # Assume it's an object with attributes
                    concept_id = getattr(result, "concept_id", "") or getattr(result, "id", "")
                    concept_name = getattr(result, "concept_name", "Unknown") or getattr(result, "name", "Unknown")
                    similarity = float(getattr(result, "similarity", 0.0) or getattr(result, "score", 0.0))
                    
                    concept_type_attr = getattr(result, "concept_type", ConceptType.TOPIC)
                    if isinstance(concept_type_attr, str):
                        try:
                            concept_type = ConceptType(concept_type_attr)
                        except ValueError:
                            concept_type = ConceptType.TOPIC
                    else:
                        concept_type = concept_type_attr
                
                # Skip results with invalid or empty IDs
                if not concept_id:
                    continue
                    
                # Add valid result to formatted results
                formatted_results.append(ConceptSimilarityResult(
                    concept_id=concept_id,
                    concept_name=concept_name,
                    similarity=similarity,
                    concept_type=concept_type
                ))
            except Exception as e:
                logger.error(f"Error formatting similarity result: {e}")
                continue
        
        return formatted_results
    
    #---------------------------
    # Validation Methods
    #---------------------------
    
    def _validate_concept_create(self, concept_data: ConceptCreate) -> None:
        """Validate concept creation data.
        
        Args:
            concept_data: Data for the concept to create
            
        Raises:
            ValueError: If validation fails
        """
        if not concept_data.name or not concept_data.name.strip():
            raise ValueError("Concept name cannot be empty")
            
        if len(concept_data.name) > 200:
            raise ValueError("Concept name cannot exceed 200 characters")
            
        if not concept_data.description or not concept_data.description.strip():
            raise ValueError("Concept description cannot be empty")
            
        # Check parent exists if provided
        if concept_data.parent_id:
            parent = self.get_concept(concept_data.parent_id)
            if not parent:
                raise ValueError(f"Parent concept not found: {concept_data.parent_id}")
                
        # Validate importance range
        if concept_data.importance < 0 or concept_data.importance > 1:
            raise ValueError("Importance must be between 0 and 1")
            
        # Validate complexity range if provided
        if concept_data.complexity is not None and (concept_data.complexity < 0 or concept_data.complexity > 1):
            raise ValueError("Complexity must be between 0 and 1")
            
        # Validate learning time if provided
        if concept_data.estimated_learning_time_minutes is not None and concept_data.estimated_learning_time_minutes < 0:
            raise ValueError("Estimated learning time cannot be negative")
    
    def _validate_concept_update(self, concept_id: str, updates: ConceptUpdate) -> None:
        """Validate concept update data.
        
        Args:
            concept_id: ID of the concept to update
            updates: Data for the concept update
            
        Raises:
            ValueError: If validation fails
        """
        if not concept_id:
            raise ValueError("Concept ID cannot be empty")
            
        # Check if any updates are provided
        update_dict = updates.dict(exclude_unset=True)
        if not update_dict:
            raise ValueError("No updates provided")
            
        # Validate name if provided
        if "name" in update_dict and (not update_dict["name"] or not update_dict["name"].strip()):
            raise ValueError("Concept name cannot be empty")
            
        if "name" in update_dict and len(update_dict["name"]) > 200:
            raise ValueError("Concept name cannot exceed 200 characters")
            
        # Validate description if provided
        if "description" in update_dict and (not update_dict["description"] or not update_dict["description"].strip()):
            raise ValueError("Concept description cannot be empty")
            
        # Check parent exists if provided
        if "parent_id" in update_dict and update_dict["parent_id"]:
            parent = self.get_concept(update_dict["parent_id"])
            if not parent:
                raise ValueError(f"Parent concept not found: {update_dict['parent_id']}")
                
            # Prevent circular parent reference
            if update_dict["parent_id"] == concept_id:
                raise ValueError("Concept cannot be its own parent")
                
        # Validate importance range if provided
        if "importance" in update_dict and (update_dict["importance"] < 0 or update_dict["importance"] > 1):
            raise ValueError("Importance must be between 0 and 1")
            
        # Validate complexity range if provided
        if "complexity" in update_dict and (update_dict["complexity"] < 0 or update_dict["complexity"] > 1):
            raise ValueError("Complexity must be between 0 and 1")
            
        # Validate learning time if provided
        if "estimated_learning_time_minutes" in update_dict and update_dict["estimated_learning_time_minutes"] < 0:
            raise ValueError("Estimated learning time cannot be negative")
    
    def _validate_relationship_create(self, relationship_data: RelationshipCreate) -> None:
        """Validate relationship creation data.
        
        Args:
            relationship_data: Data for the relationship to create
            
        Raises:
            ValueError: If validation fails
        """
        if not relationship_data.source_id:
            raise ValueError("Source concept ID cannot be empty")
            
        if not relationship_data.target_id:
            raise ValueError("Target concept ID cannot be empty")
            
        if relationship_data.source_id == relationship_data.target_id:
            raise ValueError("Source and target concepts cannot be the same")
            
        # Validate strength range
        if relationship_data.strength < 0 or relationship_data.strength > 1:
            raise ValueError("Strength must be between 0 and 1")
    
    def _validate_relationship_update(self, relationship_id: str, updates: RelationshipUpdate) -> None:
        """Validate relationship update data.
        
        Args:
            relationship_id: ID of the relationship to update
            updates: Data for the relationship update
            
        Raises:
            ValueError: If validation fails
        """
        if not relationship_id:
            raise ValueError("Relationship ID cannot be empty")
            
        # Check if any updates are provided
        update_dict = updates.dict(exclude_unset=True)
        if not update_dict:
            raise ValueError("No updates provided")
            
        # Validate source_id if provided
        if "source_id" in update_dict and not update_dict["source_id"]:
            raise ValueError("Source concept ID cannot be empty")
            
        # Validate target_id if provided
        if "target_id" in update_dict and not update_dict["target_id"]:
            raise ValueError("Target concept ID cannot be empty")
            
        # Check that source and target are not the same
        source_id = update_dict.get("source_id")
        target_id = update_dict.get("target_id")
        
        if source_id and target_id and source_id == target_id:
            raise ValueError("Source and target concepts cannot be the same")
            
        # Validate strength range if provided
        if "strength" in update_dict and (update_dict["strength"] < 0 or update_dict["strength"] > 1):
            raise ValueError("Strength must be between 0 and 1")
    
    def _validate_learning_path_request(self, request: LearningPathRequest) -> None:
        """Validate learning path request data.
        
        Args:
            request: Learning path request data
            
        Raises:
            ValueError: If validation fails
        """
        if not request.goal or not request.goal.strip():
            raise ValueError("Learning path goal cannot be empty")
            
        if len(request.goal) > 500:
            raise ValueError("Learning path goal cannot exceed 500 characters")
            
        # Must have either concept_ids or domain_id
        if not request.concept_ids and not request.domain_id:
            raise ValueError("Either concept_ids or domain_id must be provided")
            
        # Validate max_time_minutes if provided
        if request.max_time_minutes is not None and request.max_time_minutes <= 0:
            raise ValueError("Max time must be positive")
            
        # Validate concept_ids if provided
        if request.concept_ids and len(request.concept_ids) > 0:
            for concept_id in request.concept_ids:
                concept = self.get_concept(concept_id)
                if not concept:
                    raise ValueError(f"Concept not found: {concept_id}")
                    
        # Validate domain_id if provided
        if request.domain_id:
            domain = self.get_concept(request.domain_id)
            if not domain:
                raise ValueError(f"Domain not found: {request.domain_id}")
            
            # Check domain concept type
            if domain.concept_type != ConceptType.DOMAIN:
                logger.warning(f"Concept {request.domain_id} is not a domain type (is {domain.concept_type.value})")
    
    def _validate_domain_structure_request(self, request: DomainStructureRequest) -> None:
        """Validate domain structure request data.
        
        Args:
            request: Domain structure request data
            
        Raises:
            ValueError: If validation fails
        """
        if not request.domain_name or not request.domain_name.strip():
            raise ValueError("Domain name cannot be empty")
            
        if len(request.domain_name) > 200:
            raise ValueError("Domain name cannot exceed 200 characters")
            
        if not request.domain_description or not request.domain_description.strip():
            raise ValueError("Domain description cannot be empty")
            
        if len(request.domain_description) > 2000:
            raise ValueError("Domain description cannot exceed 2000 characters")
            
        if not request.key_topics or len(request.key_topics) == 0:
            raise ValueError("At least one key topic must be provided")
            
        if len(request.key_topics) > 50:
            raise ValueError("Cannot have more than 50 key topics")
            
        # Validate difficulty level if provided
        if request.difficulty_level and not isinstance(request.difficulty_level, DifficultyLevel):
            try:
                DifficultyLevel(request.difficulty_level)
            except ValueError:
                raise ValueError(f"Invalid difficulty level: {request.difficulty_level}")
                
        # Validate metadata if provided
        if request.metadata and not isinstance(request.metadata, dict):
            raise ValueError("Metadata must be a dictionary")