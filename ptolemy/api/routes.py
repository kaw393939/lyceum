"""
Ptolemy Knowledge Map System - API Routes
========================================
FastAPI routes for the knowledge map system.
"""

import logging
import uuid
import time
import traceback
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query, Body, Path, Header
from fastapi import status, Request, BackgroundTasks, File, Form, UploadFile
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import io
import csv
import json

import jwt

from config import Config, get_config
from models import (
    Concept, ConceptCreate, ConceptUpdate, 
    Relationship, RelationshipCreate, RelationshipUpdate,
    LearningPath, LearningPathRequest, LearningPathStep,
    ValidationResult, KnowledgeGap, Activity, ActivityType,
    ConceptType, RelationshipType, DifficultyLevel, ValidationStatus,
    DomainStructureRequest, ConceptSimilarityResult, ConceptWithRelationships,
    GraphExportRequest, ExportFormat
)
from core.knowledge_manager import KnowledgeManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api")

# Security scheme
security = HTTPBearer(auto_error=False)

# Create routers
router = APIRouter()
concepts_router = APIRouter(prefix="/concepts", tags=["concepts"])
relationships_router = APIRouter(prefix="/relationships", tags=["relationships"])
learning_paths_router = APIRouter(prefix="/learning-paths", tags=["learning paths"])
domains_router = APIRouter(prefix="/domains", tags=["domains"])
search_router = APIRouter(prefix="/search", tags=["search"])
admin_router = APIRouter(prefix="/admin", tags=["admin"])
analytics_router = APIRouter(prefix="/analytics", tags=["analytics"])

# Global manager instance cache to reduce initialization overhead
_manager_cache = {}

# Dependency to get knowledge manager with caching
def get_knowledge_manager(config: Config = Depends(get_config)) -> KnowledgeManager:
    """Dependency to get the knowledge manager instance with caching."""
    cache_key = id(config)  # Use config object ID as cache key
    
    # Check if we have a cached manager for this config
    if cache_key in _manager_cache:
        manager, timestamp = _manager_cache[cache_key]
        # Check if the cached manager is still fresh (less than 10 minutes old)
        if time.time() - timestamp < 600:  # 10 minutes
            return manager
        # Otherwise close the old manager and create a new one
        try:
            manager.close()
        except Exception as e:
            logger.warning(f"Error closing cached manager: {e}")
    
    # Create a new manager
    try:
        manager = KnowledgeManager(config)
        _manager_cache[cache_key] = (manager, time.time())
        return manager
    except Exception as e:
        logger.error(f"Failed to initialize knowledge manager: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Failed to initialize knowledge management system: {str(e)}"
        )

# Dependency to get current user ID (if authenticated)
def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    config: Config = Depends(get_config)
) -> Optional[str]:
    """Validate JWT token and return user ID if valid."""
    if not config.api.require_auth:
        return None  # Authentication not required
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    try:
        payload = jwt.decode(
            credentials.credentials, 
            key=config.api.jwt_secret,
            algorithms=[config.api.jwt_algorithm]
        )
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"}
            )
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"}
        )

# Dependency to enforce authentication for admin endpoints
def require_admin_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    config: Config = Depends(get_config)
) -> str:
    """Always require authentication for admin endpoints regardless of config."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin endpoints require authentication",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    try:
        payload = jwt.decode(
            credentials.credentials, 
            key=config.api.jwt_secret,
            algorithms=[config.api.jwt_algorithm]
        )
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"}
            )
            
        # Check if user has admin role
        roles = payload.get("roles", [])
        if "admin" not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin role required"
            )
            
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.InvalidTokenError as e:
        logger.error(f"JWT error in admin auth: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as e:
        # Catch any other exception and return a 401 instead of 500
        logger.error(f"Unexpected error in admin auth: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication error",
            headers={"WWW-Authenticate": "Bearer"}
        )

# Request logging middleware function (to be registered in main.py)
async def log_requests(request: Request, call_next):
    """Log all API requests with timing information."""
    start_time = time.time()
    
    # Generate request ID for tracking
    request_id = str(uuid.uuid4())
    
    # Log the request
    logger.info(f"Request {request_id}: {request.method} {request.url.path}")
    
    # Process the request
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        logger.info(f"Request {request_id} completed in {process_time:.3f}s - Status: {response.status_code}")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request {request_id} failed after {process_time:.3f}s: {str(e)}")
        logger.debug(traceback.format_exc())
        raise

# Improved error handler for handling expected errors gracefully
from functools import wraps

def handle_exceptions(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ValueError as e:
            logger.warning(f"Validation error in {func.__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except KeyError as e:
            logger.warning(f"Missing required data in {func.__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required data: {str(e)}"
            )
        except HTTPException:
            raise
        except Exception as e:
            error_str = str(e)
            if "MongoDB service not available" in error_str:
                logger.error(f"MongoDB service unavailable in {func.__name__}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Storage service unavailable. Please try again later."
                )
            # ... (handle other specific errors similarly)
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            logger.debug(traceback.format_exc())
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred: {str(e)}"
            )
    return wrapper

#---------------------------
# Health Endpoints
#---------------------------

@router.get("/health", tags=["health"])
@handle_exceptions
async def health_check(manager: KnowledgeManager = Depends(get_knowledge_manager)):
    """Check health of all services."""
    start_time = time.time()
    try:
        health_data = manager.health_check()
        logger.info(f"Health check completed in {time.time() - start_time:.3f}s")
        return health_data
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        logger.debug(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/", tags=["health"])
async def root():
    """Root endpoint with API information."""
    config = get_config()
    return {
        "name": "Ptolemy Knowledge Map API",
        "version": config.version,
        "status": "online",
        "docs_url": config.api.docs_url,
        "timestamp": datetime.now().isoformat()
    }

#---------------------------
# Concept Endpoints
#---------------------------

@concepts_router.post(
    "/", 
    response_model=Concept, 
    status_code=status.HTTP_201_CREATED,
    summary="Create a new concept",
    description="""
    Create a new educational concept in the knowledge graph.
    
    A concept represents a distinct educational element, such as a topic, term, or skill.
    Concepts form the building blocks of the knowledge map and can be linked via relationships.
    
    The concept will be assigned a unique ID automatically if not provided.
    """,
    response_description="The newly created concept with generated ID and timestamps",
    responses={
        201: {
            "description": "Concept created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "c9b5fdb2-3f46-48c0-8a47-80e6d7f1848b",
                        "name": "Stoic Virtue Ethics",
                        "description": "The ethical framework of Stoicism that emphasizes virtue as the only true good.",
                        "concept_type": "topic",
                        "difficulty": "intermediate",
                        "parent_id": "stoicism-philosophy",
                        "importance": 0.8,
                        "complexity": 0.7,
                        "keywords": ["virtue", "ethics", "stoicism", "eudaimonia"],
                        "created_at": "2025-03-04T12:34:56Z",
                        "updated_at": "2025-03-04T12:34:56Z"
                    }
                }
            }
        },
        400: {
            "description": "Invalid input data",
            "content": {
                "application/json": {
                    "example": {"detail": "A concept with this name already exists"}
                }
            }
        },
        401: {
            "description": "Authentication required"
        }
    }
)
@handle_exceptions
async def create_concept(
    concept: ConceptCreate = Body(..., 
        description="The concept data to create",
        example={
            "name": "Stoic Virtue Ethics",
            "description": "The ethical framework of Stoicism that emphasizes virtue as the only true good.",
            "concept_type": "topic",
            "difficulty": "intermediate",
            "parent_id": "stoicism-philosophy",
            "importance": 0.8,
            "complexity": 0.7,
            "keywords": ["virtue", "ethics", "stoicism", "eudaimonia"]
        }
    ),
    manager: KnowledgeManager = Depends(get_knowledge_manager),
    current_user: Optional[str] = Depends(get_current_user)
):
    """Create a new concept in the knowledge graph."""
    return manager.create_concept(concept, current_user)

@concepts_router.post("/bulk", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
@handle_exceptions
async def bulk_create_concepts(
    concepts: List[ConceptCreate],
    manager: KnowledgeManager = Depends(get_knowledge_manager),
    current_user: Optional[str] = Depends(get_current_user)
):
    """Create multiple concepts in a single request."""
    results = []
    errors = []
    
    for i, concept_data in enumerate(concepts):
        try:
            concept = manager.create_concept(concept_data, current_user)
            results.append(concept)
        except Exception as e:
            errors.append({
                "index": i,
                "error": str(e),
                "concept_name": concept_data.name
            })
    
    return {
        "success": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors
    }

@concepts_router.get("/{concept_id}", response_model=Concept)
@handle_exceptions
async def get_concept(
    concept_id: str = Path(..., description="The ID of the concept to retrieve"),
    manager: KnowledgeManager = Depends(get_knowledge_manager)
):
    """Get a concept by ID."""
    concept = manager.get_concept(concept_id)
    if not concept:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Concept not found: {concept_id}"
        )
    return concept

@concepts_router.put("/{concept_id}", response_model=Concept)
@handle_exceptions
async def update_concept(
    concept_id: str = Path(..., description="The ID of the concept to update"),
    updates: ConceptUpdate = Body(...),
    manager: KnowledgeManager = Depends(get_knowledge_manager),
    current_user: Optional[str] = Depends(get_current_user)
):
    """Update an existing concept."""
    updated_concept = manager.update_concept(concept_id, updates, current_user)
    if not updated_concept:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Concept not found: {concept_id}"
        )
    return updated_concept

@concepts_router.delete("/{concept_id}", status_code=status.HTTP_204_NO_CONTENT)
@handle_exceptions
async def delete_concept(
    concept_id: str = Path(..., description="The ID of the concept to delete"),
    manager: KnowledgeManager = Depends(get_knowledge_manager),
    current_user: Optional[str] = Depends(get_current_user)
):
    """Delete a concept and its relationships."""
    success = manager.delete_concept(concept_id, current_user)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Concept not found: {concept_id}"
        )
    return None

@concepts_router.get("/", response_model=Dict[str, Any])
@handle_exceptions
async def list_concepts(
    skip: int = Query(0, ge=0, description="Number of concepts to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of concepts to return"),
    concept_type: Optional[ConceptType] = Query(None, description="Filter by concept type"),
    difficulty: Optional[DifficultyLevel] = Query(None, description="Filter by difficulty level"),
    parent_id: Optional[str] = Query(None, description="Filter by parent concept ID"),
    sort_by: str = Query("created_at", description="Field to sort by"),
    sort_desc: bool = Query(True, description="Sort in descending order"),
    manager: KnowledgeManager = Depends(get_knowledge_manager)
):
    """List concepts with filtering and pagination."""
    # Build filters
    filters = {}
    if concept_type:
        filters["concept_type"] = concept_type.value
    if difficulty:
        filters["difficulty"] = difficulty.value
    if parent_id:
        filters["parent_id"] = parent_id
    
    concepts = manager.list_concepts(skip, limit, filters, sort_by, sort_desc)
    
    return {
        "items": concepts,
        "pagination": {
            "skip": skip, 
            "limit": limit,
            "count": len(concepts)
        }
    }

@concepts_router.get("/{concept_id}/relationships", response_model=List[Dict[str, Any]])
@handle_exceptions
async def get_concept_relationships(
    concept_id: str = Path(..., description="The ID of the concept"),
    direction: str = Query("both", description="Relationship direction: 'incoming', 'outgoing', or 'both'"),
    relationship_type: Optional[RelationshipType] = Query(None, description="Filter by relationship type"),
    manager: KnowledgeManager = Depends(get_knowledge_manager)
):
    """Get relationships for a concept."""
    if direction not in ["incoming", "outgoing", "both"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Direction must be one of: incoming, outgoing, both"
        )
    
    # Verify concept exists
    concept = manager.get_concept(concept_id)
    if not concept:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Concept not found: {concept_id}"
        )
    
    relationships = manager.get_concept_relationships(
        concept_id=concept_id,
        direction=direction,
        relationship_types=[relationship_type] if relationship_type else None
    )
    return relationships

@concepts_router.get("/{concept_id}/graph", response_model=Dict[str, Any])
@handle_exceptions
async def get_concept_graph(
    concept_id: str = Path(..., description="The ID of the concept"),
    depth: int = Query(1, ge=1, le=3, description="Graph depth (number of hops)"),
    relationship_type: Optional[RelationshipType] = Query(None, description="Filter by relationship type"),
    manager: KnowledgeManager = Depends(get_knowledge_manager)
):
    """Get a subgraph centered on a concept."""
    # Verify concept exists
    concept = manager.get_concept(concept_id)
    if not concept:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Concept not found: {concept_id}"
        )
    
    graph = manager.get_concept_graph(
        concept_id=concept_id,
        depth=depth,
        relationship_types=[relationship_type] if relationship_type else None
    )
    
    return graph

@concepts_router.get("/{concept_id}/with-relationships", response_model=ConceptWithRelationships)
@handle_exceptions
async def get_concept_with_relationships(
    concept_id: str = Path(..., description="The ID of the concept"),
    manager: KnowledgeManager = Depends(get_knowledge_manager)
):
    """Get a concept with its relationships in both directions."""
    try:
        concept_data = manager.get_concept_with_relationships(concept_id)
        return concept_data
    except Exception as e:
        if "Concept not found" in str(e):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Concept not found: {concept_id}"
            )
        raise

@concepts_router.get("/{concept_id}/similar", response_model=List[ConceptSimilarityResult])
@handle_exceptions
async def get_similar_concepts(
    concept_id: str = Path(..., description="The ID of the concept"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of similar concepts to return"),
    manager: KnowledgeManager = Depends(get_knowledge_manager)
):
    """Find concepts similar to the specified concept."""
    # Verify concept exists
    concept = manager.get_concept(concept_id)
    if not concept:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Concept not found: {concept_id}"
        )
    
    similar_concepts = manager.get_similar_concepts(concept_id, limit)
    return similar_concepts

#---------------------------
# Relationship Endpoints
#---------------------------

@relationships_router.post("/", response_model=Relationship, status_code=status.HTTP_201_CREATED)
@handle_exceptions
async def create_relationship(
    relationship: RelationshipCreate,
    manager: KnowledgeManager = Depends(get_knowledge_manager),
    current_user: Optional[str] = Depends(get_current_user)
):
    """Create a new relationship between concepts."""
    return manager.create_relationship(relationship, current_user)

@relationships_router.post("/bulk", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
@handle_exceptions
async def bulk_create_relationships(
    relationships: List[RelationshipCreate],
    manager: KnowledgeManager = Depends(get_knowledge_manager),
    current_user: Optional[str] = Depends(get_current_user)
):
    """Create multiple relationships in a single request."""
    results = []
    errors = []
    
    for i, relationship_data in enumerate(relationships):
        try:
            # Verify source and target exist
            source = manager.get_concept(relationship_data.source_id)
            if not source:
                raise ValueError(f"Source concept not found: {relationship_data.source_id}")
                
            target = manager.get_concept(relationship_data.target_id)
            if not target:
                raise ValueError(f"Target concept not found: {relationship_data.target_id}")
            
            # Check for self-reference
            if relationship_data.source_id == relationship_data.target_id:
                raise ValueError("Source and target cannot be the same concept")
                
            relationship = manager.create_relationship(relationship_data, current_user)
            results.append(relationship)
        except Exception as e:
            errors.append({
                "index": i,
                "error": str(e),
                "source_id": relationship_data.source_id,
                "target_id": relationship_data.target_id
            })
    
    return {
        "success": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors
    }

@relationships_router.get("/{relationship_id}", response_model=Relationship)
@handle_exceptions
async def get_relationship(
    relationship_id: str = Path(..., description="The ID of the relationship to retrieve"),
    manager: KnowledgeManager = Depends(get_knowledge_manager)
):
    """Get a relationship by ID."""
    relationship = manager.get_relationship(relationship_id)
    if not relationship:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Relationship not found: {relationship_id}"
        )
    return relationship

@relationships_router.put("/{relationship_id}", response_model=Relationship)
@handle_exceptions
async def update_relationship(
    relationship_id: str = Path(..., description="The ID of the relationship to update"),
    updates: RelationshipUpdate = Body(...),
    manager: KnowledgeManager = Depends(get_knowledge_manager),
    current_user: Optional[str] = Depends(get_current_user)
):
    """Update an existing relationship."""
    updated_relationship = manager.update_relationship(relationship_id, updates, current_user)
    if not updated_relationship:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Relationship not found: {relationship_id}"
        )
    return updated_relationship

@relationships_router.delete("/{relationship_id}", status_code=status.HTTP_204_NO_CONTENT)
@handle_exceptions
async def delete_relationship(
    relationship_id: str = Path(..., description="The ID of the relationship to delete"),
    manager: KnowledgeManager = Depends(get_knowledge_manager),
    current_user: Optional[str] = Depends(get_current_user)
):
    """Delete a relationship."""
    success = manager.delete_relationship(relationship_id, current_user)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Relationship not found: {relationship_id}"
        )
    return None

#---------------------------
# Learning Path Endpoints
#---------------------------

@learning_paths_router.post("/", response_model=LearningPath, status_code=status.HTTP_201_CREATED)
@handle_exceptions
async def create_learning_path(
    request: LearningPathRequest,
    manager: KnowledgeManager = Depends(get_knowledge_manager),
    current_user: Optional[str] = Depends(get_current_user)
):
    """Create a learning path based on concepts and goal."""
    return manager.create_learning_path(request, current_user)

@learning_paths_router.get("/{path_id}", response_model=LearningPath)
@handle_exceptions
async def get_learning_path(
    path_id: str = Path(..., description="The ID of the learning path to retrieve"),
    manager: KnowledgeManager = Depends(get_knowledge_manager)
):
    """Get a learning path by ID."""
    path = manager.get_learning_path(path_id)
    if not path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Learning path not found: {path_id}"
        )
    return path

@learning_paths_router.get("/", response_model=Dict[str, Any])
@handle_exceptions
async def list_learning_paths(
    skip: int = Query(0, ge=0, description="Number of paths to skip"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of paths to return"),
    learner_level: Optional[str] = Query(None, description="Filter by learner level"),
    goal_keyword: Optional[str] = Query(None, description="Filter by keyword in goal"),
    manager: KnowledgeManager = Depends(get_knowledge_manager)
):
    """List learning paths with filtering and pagination."""
    # Build filters
    filters = {}
    if learner_level:
        filters["target_learner_level"] = learner_level
    if goal_keyword:
        filters["goal"] = {"$regex": goal_keyword, "$options": "i"}
    
    paths = manager.list_learning_paths(skip, limit, filters)
    
    return {
        "items": paths,
        "pagination": {
            "skip": skip,
            "limit": limit,
            "count": len(paths)
        }
    }

#---------------------------
# Domain Endpoints
#---------------------------

@domains_router.post("/", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
@handle_exceptions
async def create_domain(
    background_tasks: BackgroundTasks,
    request: DomainStructureRequest,
    manager: KnowledgeManager = Depends(get_knowledge_manager),
    current_user: Optional[str] = Depends(get_current_user)
):
    """Create a complete domain structure with concepts and relationships."""
    # For long-running generation, run as background task
    if request.concept_count and request.concept_count > 15:
        # First create a placeholder domain concept to return to the client
        domain_concept = ConceptCreate(
            name=request.domain_name,
            description=f"{request.domain_description} (Generation in progress...)",
            concept_type=ConceptType.DOMAIN,
            difficulty=request.difficulty_level or DifficultyLevel.INTERMEDIATE,
            importance=1.0,
            keywords=request.key_topics or [],
            metadata={"generation_status": "in_progress"}
        )
        domain = manager.create_concept(domain_concept, current_user)
        
        # Start generation in background
        background_tasks.add_task(
            _generate_domain_in_background, 
            manager, 
            request, 
            domain.id, 
            current_user
        )
        
        return {
            "status": "processing",
            "message": "Domain generation started. This may take some time.",
            "domain_id": domain.id,
            "domain": domain.dict()
        }
    else:
        # For smaller domain generation, do it synchronously
        result = manager.create_domain(request, current_user)
        if not result.get("success"):
            # If creation failed, return a 500 error with details
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Domain creation failed")
            )
        return result

async def _generate_domain_in_background(
    manager: KnowledgeManager,
    request: DomainStructureRequest,
    domain_id: str,
    user_id: Optional[str]
):
    """Background task to generate a domain."""
    try:
        logger.info(f"Starting background domain generation for {domain_id}")
        
        # Set the parent_id in the request to use the placeholder domain
        request.parent_id = domain_id
        
        # Generate the domain using the actual domain ID
        result = manager.create_domain(request, user_id)
        
        # Update the domain with generation status
        if result.get("success"):
            manager.update_concept(
                domain_id,
                ConceptUpdate(
                    description=request.domain_description,
                    metadata={"generation_status": "complete", "generated_at": datetime.now().isoformat()}
                ),
                user_id
            )
            logger.info(f"Background domain generation completed successfully for {domain_id}")
        else:
            # Update domain with error information
            manager.update_concept(
                domain_id,
                ConceptUpdate(
                    description=f"{request.domain_description} (Generation failed)",
                    metadata={
                        "generation_status": "failed", 
                        "error": result.get("error", "Unknown error"),
                        "failed_at": datetime.now().isoformat()
                    }
                ),
                user_id
            )
            logger.error(f"Background domain generation failed for {domain_id}: {result.get('error')}")
    except Exception as e:
        logger.error(f"Error in background domain generation for {domain_id}: {e}")
        logger.debug(traceback.format_exc())
        try:
            # Update domain with error information
            manager.update_concept(
                domain_id,
                ConceptUpdate(
                    description=f"{request.domain_description} (Generation failed)",
                    metadata={
                        "generation_status": "failed", 
                        "error": str(e),
                        "failed_at": datetime.now().isoformat()
                    }
                ),
                user_id
            )
        except Exception as update_error:
            logger.error(f"Failed to update domain error status: {update_error}")
            logger.debug(traceback.format_exc())

@domains_router.get("/{domain_id}/structure", response_model=Dict[str, Any])
@handle_exceptions
async def get_domain_structure(
    domain_id: str = Path(..., description="The ID of the domain"),
    include_relationships: bool = Query(True, description="Whether to include relationships"),
    manager: KnowledgeManager = Depends(get_knowledge_manager)
):
    """Get the structure of a domain including its concepts and relationships."""
    # Verify domain exists
    domain = manager.get_concept(domain_id)
    if not domain:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Domain not found: {domain_id}"
        )
    
    domain_structure = manager.get_domain_structure(domain_id, include_relationships)
    if not domain_structure.get("domain"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Domain not found: {domain_id}"
        )
    return domain_structure

@domains_router.get("/{domain_id}/validate", response_model=ValidationResult)
@handle_exceptions
async def validate_domain(
    domain_id: str = Path(..., description="The ID of the domain to validate"),
    manager: KnowledgeManager = Depends(get_knowledge_manager)
):
    """Validate a domain for consistency and quality issues."""
    # Verify domain exists
    domain = manager.get_concept(domain_id)
    if not domain:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Domain not found: {domain_id}"
        )
    
    validation_result = manager.validate_domain(domain_id)
    return validation_result

@domains_router.get("/{domain_id}/status", response_model=Dict[str, Any])
@handle_exceptions
async def get_domain_generation_status(
    domain_id: str = Path(..., description="The ID of the domain"),
    manager: KnowledgeManager = Depends(get_knowledge_manager)
):
    """Get the generation status of a domain."""
    domain = manager.get_concept(domain_id)
    if not domain:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Domain not found: {domain_id}"
        )
    
    # Get generation status from metadata
    metadata = domain.metadata or {}
    generation_status = metadata.get("generation_status", "unknown")
    
    # Count child concepts
    try:
        children = manager.list_concepts(
            limit=1000,
            filters={"parent_id": domain_id}
        )
        
        return {
            "domain_id": domain_id,
            "domain_name": domain.name,
            "status": generation_status,
            "child_count": len(children),
            "created_at": domain.created_at.isoformat() if domain.created_at else None,
            "updated_at": domain.updated_at.isoformat() if domain.updated_at else None,
            "metadata": metadata
        }
    except Exception as e:
        # If we can't get children, return basic info
        logger.warning(f"Error getting domain children count: {e}")
        return {
            "domain_id": domain_id,
            "domain_name": domain.name,
            "status": generation_status,
            "child_count": "unknown",
            "created_at": domain.created_at.isoformat() if domain.created_at else None,
            "updated_at": domain.updated_at.isoformat() if domain.updated_at else None,
            "metadata": metadata
        }

@domains_router.get("/{domain_id}/gaps", response_model=List[Dict[str, Any]])
@handle_exceptions
async def identify_knowledge_gaps(
    domain_id: str = Path(..., description="The ID of the domain"),
    manager: KnowledgeManager = Depends(get_knowledge_manager)
):
    """Identify knowledge gaps in a domain."""
    # Verify domain exists
    domain = manager.get_concept(domain_id)
    if not domain:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Domain not found: {domain_id}"
        )
    
    # This would call into a method to identify gaps - using the validation for now
    validation = manager.validate_domain(domain_id)
    
    # Convert validation issues to knowledge gaps
    gaps = []
    for issue in validation.issues:
        if issue.severity in ["high", "medium"]:
            gaps.append({
                "id": str(uuid.uuid4()),
                "domain_id": domain_id,
                "title": f"Knowledge Gap: {issue.issue_type}",
                "description": issue.description,
                "importance": 0.8 if issue.severity == "high" else 0.5,
                "related_concepts": issue.concepts_involved,
                "suggested_content": issue.recommendation,
                "status": "identified",
                "created_at": datetime.now().isoformat()
            })
    
    return gaps

#---------------------------
# Search Endpoints
#---------------------------

@search_router.get("/text", response_model=List[Concept])
@handle_exceptions
async def search_by_text(
    query: str = Query(..., min_length=1, description="Text to search for"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    concept_type: Optional[ConceptType] = Query(None, description="Filter by concept type"),
    manager: KnowledgeManager = Depends(get_knowledge_manager)
):
    """Search concepts by text query."""
    # First try to get exact match if query looks like an ID
    if len(query) > 20 and "-" in query:
        try:
            concept = manager.get_concept(query)
            if concept:
                return [concept]
        except:
            pass
    
    # Perform text search
    concepts = manager.search_concepts(query, limit)
    
    # Apply filters if needed
    if concept_type:
        concepts = [c for c in concepts if c.concept_type == concept_type]
    
    return concepts

@search_router.get("/semantic", response_model=List[ConceptSimilarityResult])
@handle_exceptions
async def search_by_semantic(
    query: str = Query(..., min_length=1, description="Text to search for semantically"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    concept_type: Optional[ConceptType] = Query(None, description="Filter by concept type"),
    manager: KnowledgeManager = Depends(get_knowledge_manager)
):
    """Search concepts semantically using vector similarity."""
    results = manager.semantic_search(
        query=query,
        limit=limit,
        concept_types=[concept_type] if concept_type else None
    )
    
    return results

#---------------------------
# Analytics Endpoints
#---------------------------

@analytics_router.get("/concept-counts", response_model=Dict[str, Any])
@handle_exceptions
async def get_concept_counts(
    manager: KnowledgeManager = Depends(get_knowledge_manager)
):
    """Get counts of concepts by various dimensions."""
    try:
        # Count by concept type
        concept_type_counts = {}
        for concept_type in ConceptType:
            # Fix: Changed limit=0 to limit=1000 to avoid validation error
            concepts = manager.list_concepts(
                limit=1000,  # Changed from 0 to avoid "Limit value must be at least 1" error
                filters={"concept_type": concept_type.value}
            )
            concept_type_counts[concept_type.value] = len(concepts)
        
        # Count by difficulty
        difficulty_counts = {}
        for difficulty in DifficultyLevel:
            # Fix: Changed limit=0 to limit=1000 to avoid validation error
            concepts = manager.list_concepts(
                limit=1000,  # Changed from 0 to avoid "Limit value must be at least 1" error
                filters={"difficulty": difficulty.value}
            )
            difficulty_counts[difficulty.value] = len(concepts)
        
        # Get total count
        # Fix: Changed limit=0 to limit=1000 to avoid validation error
        all_concepts = manager.list_concepts(limit=1000)  # Changed from 0 to avoid validation error
        
        return {
            "total": len(all_concepts),
            "by_concept_type": concept_type_counts,
            "by_difficulty": difficulty_counts,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting concept counts: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get concept counts: {str(e)}"
        )

@analytics_router.get("/relationship-stats", response_model=Dict[str, Any])
@handle_exceptions
async def get_relationship_stats(
    manager: KnowledgeManager = Depends(get_knowledge_manager)
):
    """Get statistics about relationships in the knowledge graph."""
    try:
        # This requires Neo4j service, so handle service unavailability
        return {
            "message": "Relationship analytics not yet implemented",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting relationship stats: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get relationship stats: {str(e)}"
        )

#---------------------------
# Admin Endpoints
#---------------------------

@admin_router.get("/stats", response_model=Dict[str, Any])
@handle_exceptions
async def get_system_stats(
    manager: KnowledgeManager = Depends(get_knowledge_manager),
    current_user: str = Depends(require_admin_auth)
):
    """Get system statistics (admin only)."""
    try:
        # Get health status
        health = manager.health_check()
        
        # Count total concepts
        total_concepts = "unavailable"
        try:
            # Fix: Changed limit=0 to limit=1000 to avoid validation error
            concepts = manager.list_concepts(limit=1000)  # Changed from 0
            total_concepts = len(concepts)
        except Exception as e:
            logger.warning(f"Error getting total concepts count: {e}")
        
        # Count concepts by type
        concept_counts = {}
        for concept_type in ConceptType:
            try:
                # Fix: Changed limit=0 to limit=1000 to avoid validation error
                concepts = manager.list_concepts(
                    limit=1000,  # Changed from 0
                    filters={"concept_type": concept_type.value}
                )
                concept_counts[concept_type.value] = len(concepts)
            except Exception as e:
                logger.warning(f"Error getting count for concept type {concept_type.value}: {e}")
                concept_counts[concept_type.value] = "unavailable"
        
        return {
            "health": health,
            "stats": {
                "total_concepts": total_concepts,
                "concept_counts": concept_counts
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system stats: {str(e)}"
        )
    
@admin_router.post("/export", response_model=Dict[str, Any])
@handle_exceptions
async def export_graph(
    background_tasks: BackgroundTasks,
    request: GraphExportRequest = Body(...),
    manager: KnowledgeManager = Depends(get_knowledge_manager),
    current_user: str = Depends(require_admin_auth)
):
    """Export part of the knowledge graph in various formats (admin only)."""
    # Start the export in a background task for larger exports
    export_id = str(uuid.uuid4())
    
    # This is a placeholder - actual implementation would use a KnowledgeManager export method
    background_tasks.add_task(
        _process_graph_export,
        manager,
        request,
        export_id,
        current_user
    )
    
    return {
        "status": "processing",
        "export_id": export_id,
        "message": "Graph export started. This operation will run in the background."
    }

async def _process_graph_export(
    manager: KnowledgeManager,
    request: GraphExportRequest,
    export_id: str,
    user_id: str
):
    """Background task to process graph export."""
    try:
        logger.info(f"Starting background export {export_id} for user {user_id}")
        
        # This is a placeholder - actual implementation would use a KnowledgeManager export method
        # For now, just log the completion
        logger.info(f"Export {export_id} completed successfully")
        
        # Log activity
        if hasattr(manager, "_log_activity"):
            manager._log_activity(
                ActivityType.EXPORT,
                user_id,
                "graph",
                export_id,
                {"format": request.format.value, "root_concept": request.root_concept_id}
            )
    except Exception as e:
        logger.error(f"Export {export_id} failed: {e}")
        logger.debug(traceback.format_exc())

@admin_router.post("/import", response_model=Dict[str, Any])
@handle_exceptions
async def import_data(
    file: UploadFile = File(...),
    import_type: str = Form(..., description="Type of import (concepts, relationships, domain)"),
    parent_id: Optional[str] = Form(None, description="Optional parent ID for imported concepts"),
    manager: KnowledgeManager = Depends(get_knowledge_manager),
    current_user: str = Depends(require_admin_auth)
):
    """Import data into the knowledge graph (admin only)."""
    # Validate import type
    if import_type not in ["concepts", "relationships", "domain"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid import type. Must be one of: concepts, relationships, domain"
        )
    
    # Validate file extension
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in ["json", "csv"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Supported formats: JSON, CSV"
        )
    
    # Read file content
    content = await file.read()
    
    # This is a placeholder since KnowledgeManager doesn't yet have these methods
    # This would need to be implemented with appropriate import methods
    return {
        "status": "not_implemented",
        "message": "Data import functionality is not yet implemented",
        "details": {
            "import_type": import_type,
            "file_name": file.filename,
            "file_size": len(content),
            "parent_id": parent_id
        }
    }

@admin_router.get("/cache/stats", response_model=Dict[str, Any])
@handle_exceptions
async def get_cache_stats(
    manager: KnowledgeManager = Depends(get_knowledge_manager),
    current_user: str = Depends(require_admin_auth)
):
    """Get statistics about the cache (admin only)."""
    try:
        # Access the cache from the manager
        concept_cache_size = len(manager._concept_cache.cache) if hasattr(manager, "_concept_cache") else 0
        relationship_cache_size = len(manager._relationship_cache.cache) if hasattr(manager, "_relationship_cache") else 0
        
        return {
            "concept_cache_size": concept_cache_size,
            "relationship_cache_size": relationship_cache_size,
            "manager_cache_size": len(_manager_cache),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache stats: {str(e)}"
        )

@admin_router.post("/cache/clear", response_model=Dict[str, Any])
@handle_exceptions
async def clear_cache(
    manager: KnowledgeManager = Depends(get_knowledge_manager),
    current_user: str = Depends(require_admin_auth)
):
    """Clear the cache (admin only)."""
    try:
        # Clear the manager's caches
        if hasattr(manager, "_concept_cache"):
            manager._concept_cache.clear()
        if hasattr(manager, "_relationship_cache"):
            manager._relationship_cache.clear()
        
        # Clear the manager cache
        global _manager_cache
        for mgr, _ in _manager_cache.values():
            try:
                mgr.close()
            except Exception as e:
                logger.warning(f"Error closing manager: {e}")
        _manager_cache.clear()
        
        return {
            "status": "success",
            "message": "Cache cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )

@admin_router.get("/activities", response_model=List[Dict[str, Any]])
@handle_exceptions
async def get_user_activities(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    manager: KnowledgeManager = Depends(get_knowledge_manager),
    current_user: str = Depends(require_admin_auth)
):
    """Get user activities (admin only)."""
    # This is a placeholder since KnowledgeManager doesn't have this method
    # It would need to be implemented using MongoDB
    return []

# Register all routers
router.include_router(concepts_router)
router.include_router(relationships_router)
router.include_router(learning_paths_router)
router.include_router(domains_router)
router.include_router(search_router)
router.include_router(analytics_router)
router.include_router(admin_router)