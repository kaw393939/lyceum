import logging
import time
import uuid
import traceback
import functools
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta

from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
from pymongo.errors import DuplicateKeyError, ConnectionFailure, OperationFailure
from pymongo.collection import Collection
from bson.objectid import ObjectId

from config import MongoConfig
from models import (
    Concept, Relationship, LearningPath, KnowledgeGap, Activity,
    ConceptType, RelationshipType, ValidationStatus
)

# Define decorators
def retry_on_exception(max_retries=3):
    """Decorator to retry a function on exception."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise
                    logging.warning(f"Retrying {func.__name__} after error: {e}, attempt {attempt + 1}/{max_retries}")
                    time.sleep(2 ** attempt)  # Exponential backoff
        return wrapper
    return decorator

def log_execution_time(func):
    """Decorator to log function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        logging.debug(f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result
    return wrapper

class MongoService:
    """Service for document storage operations using MongoDB."""
    
    def __init__(self, config: MongoConfig):
        self.config = config
        self.client: Optional[MongoClient] = None
        self.db = None
        self.concepts: Optional[Collection] = None
        self.domains: Optional[Collection] = None
        self.relationships: Optional[Collection] = None
        self.learning_paths: Optional[Collection] = None
        self.analytics: Optional[Collection] = None
        self.activity: Optional[Collection] = None
        self.users: Optional[Collection] = None
        self.cache: Optional[Collection] = None
        
        # Initialize metrics
        self._metrics = {
            "operations_executed": 0,
            "errors": 0,
            "last_operation_time": None,
            "last_error_time": None
        }
        self._last_error = None
        
        self.connect()
        self._setup_indexes()
    
    def connect(self) -> bool:
        """Establish connection to MongoDB with retry logic."""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                self.client = MongoClient(self.config.uri)
                self.db = self.client[self.config.database]
                
                # Initialize collections
                self.concepts = self.db[self.config.concepts_collection]
                self.domains = self.db[self.config.domains_collection]
                self.relationships = self.db[self.config.relationships_collection]
                self.learning_paths = self.db[self.config.learning_paths_collection]
                self.analytics = self.db[self.config.analytics_collection]
                self.users = self.db[self.config.users_collection]
                self.activity = self.db[self.config.activity_collection]
                self.cache = self.db[self.config.cache_collection]
                
                # Test connection
                self.client.admin.command('ping')
                logging.info("Connected to MongoDB successfully")
                return True
            except ConnectionFailure as e:
                logging.error(f"MongoDB connection attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logging.error("Failed to connect to MongoDB after multiple attempts")
                    self.client = None
                    self.db = None
                    self.concepts = None
                    self.relationships = None
                    self.learning_paths = None
                    return False
    
    def _check_connection(self) -> bool:
        """Check if MongoDB connection is available."""
        if self.client is None or self.db is None:
            logging.error("MongoDB connection not available")
            return False
        try:
            self.client.admin.command('ping')
            return True
        except Exception as e:
            logging.error(f"MongoDB connection check failed: {e}")
            return False
    
    def _convert_enum_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert any enum values in the data dictionary to their string representation."""
        result = {}
        for key, value in data.items():
            if hasattr(value, 'value'):  # Check if it's an enum
                result[key] = value.value
            elif isinstance(value, dict):
                result[key] = self._convert_enum_values(value)
            elif isinstance(value, list):
                result[key] = [
                    self._convert_enum_values(item) if isinstance(item, dict) 
                    else (item.value if hasattr(item, 'value') else item)
                    for item in value
                ]
            else:
                result[key] = value
        return result
    
    def _setup_indexes(self):
        """Set up MongoDB indexes for optimal query performance."""
        if self.db is None:
            logging.error("Cannot set up MongoDB indexes: No connection")
            return

        try:
            # Concepts collection indexes
            if self.concepts is not None:
                self.concepts.create_index([("id", ASCENDING)], unique=True)
                self.concepts.create_index([("name", ASCENDING)])
                self.concepts.create_index([("concept_type", ASCENDING)])
                self.concepts.create_index([("parent_id", ASCENDING)])
                self.concepts.create_index([("difficulty", ASCENDING)])
                self.concepts.create_index([("importance", DESCENDING)])
                self.concepts.create_index([("created_at", DESCENDING)])
                self.concepts.create_index([("keywords", ASCENDING)])
                self.concepts.create_index([("name", TEXT), ("description", TEXT), ("keywords", TEXT)])
            
            # Relationships collection indexes
            if self.relationships is not None:
                self.relationships.create_index([("id", ASCENDING)], unique=True)
                self.relationships.create_index([("source_id", ASCENDING)])
                self.relationships.create_index([("target_id", ASCENDING)])
                self.relationships.create_index([("relationship_type", ASCENDING)])
                self.relationships.create_index([("created_at", DESCENDING)])
                self.relationships.create_index([("source_id", ASCENDING), ("target_id", ASCENDING)])
            
            # Learning paths collection indexes
            if self.learning_paths is not None:
                self.learning_paths.create_index([("id", ASCENDING)], unique=True)
                self.learning_paths.create_index([("target_learner_level", ASCENDING)])
                self.learning_paths.create_index([("concepts", ASCENDING)])
                self.learning_paths.create_index([("created_at", DESCENDING)])
            
            # Activity collection indexes
            if self.activity is not None:
                self.activity.create_index([("timestamp", DESCENDING)])
                self.activity.create_index([("user_id", ASCENDING)])
                self.activity.create_index([("entity_id", ASCENDING)])
                self.activity.create_index([("activity_type", ASCENDING)])
            
            # Cache collection indexes
            if self.cache is not None:
                self.cache.create_index([("key", ASCENDING)], unique=True)
                self.cache.create_index([("expires_at", ASCENDING)], expireAfterSeconds=0)
            
            logging.info("MongoDB indexes setup complete")
        except Exception as e:
            logging.error(f"Failed to set up MongoDB indexes: {e}")
    
    def close(self):
        """Close the MongoDB database connection."""
        if self.client is not None:
            self.client.close()
            self.client = None
            self.db = None
            self.concepts = None
            self.domains = None
            self.relationships = None
            self.learning_paths = None
            self.analytics = None
            self.activity = None
            self.users = None
            self.cache = None
            logging.info("MongoDB connection closed")
    
    def health_check(self) -> Dict[str, Any]:
        """Check MongoDB connection health and return metrics."""
        if self.client is None:
            return {"status": "disconnected", "error": "No connection to MongoDB"}
        
        try:
            # Basic connectivity check
            self.client.admin.command('ping')
            
            # Get collection stats
            stats = {
                "concepts": self.concepts.count_documents({}) if self.concepts is not None else 0,
                "relationships": self.relationships.count_documents({}) if self.relationships is not None else 0,
                "learning_paths": self.learning_paths.count_documents({}) if self.learning_paths is not None else 0,
                "activities": self.activity.count_documents({}) if self.activity is not None else 0
            }
            
            # Database info
            server_info = self.client.server_info()
            
            return {
                "status": "connected",
                "version": server_info.get("version", "unknown"),
                "collections": stats,
                "total_documents": sum(stats.values())
            }
        except Exception as e:
            logging.error(f"MongoDB health check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    # Concept Operations
    
    def create_concept(self, concept: Concept) -> bool:
        """Create a concept document in MongoDB."""
        if self.concepts is None:
            logging.error("MongoDB connection not available")
            return False
        
        try:
            concept_dict = concept.dict()
            result = self.concepts.insert_one(concept_dict)
            return result.acknowledged
        except DuplicateKeyError:
            logging.error(f"Concept with ID {concept.id} already exists")
            return False
        except Exception as e:
            logging.error(f"MongoDB error creating concept: {e}")
            return False
    
    def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a concept from MongoDB by ID."""
        if self.concepts is None:
            logging.error("MongoDB connection not available")
            return None
        
        try:
            result = self.concepts.find_one({"id": concept_id})
            return result
        except Exception as e:
            logging.error(f"MongoDB error retrieving concept: {e}")
            return None
    
    def update_concept(self, concept_id: str, updates: Dict[str, Any]) -> bool:
        """Update a concept in MongoDB."""
        if self.concepts is None:
            logging.error("MongoDB connection not available")
            return False
        
        try:
            if "updated_at" not in updates:
                updates["updated_at"] = datetime.now()
            
            result = self.concepts.update_one(
                {"id": concept_id},
                {"$set": updates}
            )
            return result.modified_count > 0
        except Exception as e:
            logging.error(f"MongoDB error updating concept: {e}")
            return False
    
    def delete_concept(self, concept_id: str) -> bool:
        """Delete a concept from MongoDB along with its related relationships."""
        if self.concepts is None or self.relationships is None:
            logging.error("MongoDB connection not available")
            return False
        
        try:
            result = self.concepts.delete_one({"id": concept_id})
            
            # Delete related relationships
            self.relationships.delete_many({
                "$or": [
                    {"source_id": concept_id},
                    {"target_id": concept_id}
                ]
            })
            
            return result.deleted_count > 0
        except Exception as e:
            logging.error(f"MongoDB error deleting concept: {e}")
            return False
    
    def list_concepts(self, skip: int = 0, limit: int = 100, 
                      filters: Optional[Dict[str, Any]] = None, 
                      sort_by: str = "created_at", 
                      sort_desc: bool = True) -> List[Dict[str, Any]]:
        """List concepts with pagination, filtering and sorting."""
        if self.concepts is None:
            logging.error("MongoDB connection not available")
            return []
        
        try:
            query = filters or {}
            sort_direction = DESCENDING if sort_desc else ASCENDING
            cursor = self.concepts.find(query).sort(sort_by, sort_direction).skip(skip).limit(limit)
            return list(cursor)
        except Exception as e:
            logging.error(f"MongoDB error listing concepts: {e}")
            return []
    
    def count_concepts(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count concepts matching the given filters."""
        if self.concepts is None:
            logging.error("MongoDB connection not available")
            return 0
        
        try:
            query = filters or {}
            return self.concepts.count_documents(query)
        except Exception as e:
            logging.error(f"MongoDB error counting concepts: {e}")
            return 0
    
    def search_concepts(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search concepts using text search."""
        if self.concepts is None:
            logging.error("MongoDB connection not available")
            return []
        
        try:
            results = self.concepts.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)
            return list(results)
        except Exception as e:
            logging.error(f"MongoDB error searching concepts: {e}")
            return []
            
    def search_concepts_by_name(self, name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for concepts by exact or partial name match.
        
        Args:
            name: Name to search for (exact or partial)
            limit: Maximum number of results to return
            
        Returns:
            List of matching concepts
        """
        if self.concepts is None:
            logging.error("MongoDB connection not available")
            return []
            
        try:
            # First try exact match (case-insensitive)
            exact_query = {"name": {"$regex": f"^{name}$", "$options": "i"}}
            exact_matches = list(self.concepts.find(exact_query).limit(limit))
            
            if exact_matches:
                logging.info(f"MongoDB: found {len(exact_matches)} concepts with exact name match '{name}'")
                return exact_matches
                
            # If no exact match, try partial match
            partial_query = {"name": {"$regex": name, "$options": "i"}}
            cursor = self.concepts.find(partial_query).limit(limit)
            results = list(cursor)
            logging.info(f"MongoDB: found {len(results)} concepts with partial name match '{name}'")
            return results
        except Exception as e:
            logging.error(f"MongoDB error searching concepts by name: {e}")
            return []
    
    # Relationship Operations
    
    def create_relationship(self, relationship: Relationship) -> bool:
        """Create a relationship document in MongoDB."""
        if self.relationships is None:
            logging.error("MongoDB connection not available")
            return False
        
        try:
            relationship_dict = relationship.dict()
            result = self.relationships.insert_one(relationship_dict)
            return result.acknowledged
        except DuplicateKeyError:
            logging.error(f"Relationship with ID {relationship.id} already exists")
            return False
        except Exception as e:
            logging.error(f"MongoDB error creating relationship: {e}")
            return False
    
    def get_relationship(self, relationship_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a relationship from MongoDB by ID."""
        if self.relationships is None:
            logging.error("MongoDB connection not available")
            return None
        
        try:
            result = self.relationships.find_one({"id": relationship_id})
            return result
        except Exception as e:
            logging.error(f"MongoDB error retrieving relationship: {e}")
            return None

    def update_relationship(self, relationship_id: str, updates: Dict[str, Any]) -> bool:
        """Update a relationship in MongoDB."""
        if self.relationships is None:
            logging.error("MongoDB connection not available")
            return False

        try:
            if "updated_at" not in updates:
                updates["updated_at"] = datetime.now()
            result = self.relationships.update_one(
                {"id": relationship_id},
                {"$set": updates}
            )
            return result.modified_count > 0
        except Exception as e:
            logging.error(f"MongoDB error updating relationship: {e}")
            return False

    def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship from MongoDB."""
        if self.relationships is None:
            logging.error("MongoDB connection not available")
            return False

        try:
            result = self.relationships.delete_one({"id": relationship_id})
            return result.deleted_count > 0
        except Exception as e:
            logging.error(f"MongoDB error deleting relationship: {e}")
            return False

    def list_relationships(self, skip: int = 0, limit: int = 100,
                           filters: Optional[Dict[str, Any]] = None,
                           sort_by: str = "created_at", sort_desc: bool = True) -> List[Dict[str, Any]]:
        """List relationships with pagination, filtering and sorting."""
        if self.relationships is None:
            logging.error("MongoDB connection not available")
            return []

        try:
            query = filters or {}
            sort_direction = DESCENDING if sort_desc else ASCENDING
            cursor = self.relationships.find(query).sort(sort_by, sort_direction).skip(skip).limit(limit)
            return list(cursor)
        except Exception as e:
            logging.error(f"MongoDB error listing relationships: {e}")
            return []
    
    def count_relationships(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count relationships matching the given filters."""
        if self.relationships is None:
            logging.error("MongoDB connection not available")
            return 0

        try:
            query = filters or {}
            return self.relationships.count_documents(query)
        except Exception as e:
            logging.error(f"MongoDB error counting relationships: {e}")
            return 0
    
    # Learning Path Operations
    
    def create_learning_path(self, learning_path: LearningPath) -> bool:
        """Create a learning path document in MongoDB."""
        if self.learning_paths is None:
            logging.error("MongoDB connection not available")
            return False

        try:
            lp_dict = learning_path.dict()
            result = self.learning_paths.insert_one(lp_dict)
            return result.acknowledged
        except DuplicateKeyError:
            logging.error(f"Learning path with ID {learning_path.id} already exists")
            return False
        except Exception as e:
            logging.error(f"MongoDB error creating learning path: {e}")
            return False

    def get_learning_path(self, learning_path_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a learning path from MongoDB by ID."""
        if self.learning_paths is None:
            logging.error("MongoDB connection not available")
            return None

        try:
            result = self.learning_paths.find_one({"id": learning_path_id})
            return result
        except Exception as e:
            logging.error(f"MongoDB error retrieving learning path: {e}")
            return None

    def update_learning_path(self, learning_path_id: str, updates: Dict[str, Any]) -> bool:
        """Update a learning path in MongoDB."""
        if self.learning_paths is None:
            logging.error("MongoDB connection not available")
            return False

        try:
            if "updated_at" not in updates:
                updates["updated_at"] = datetime.now()
            result = self.learning_paths.update_one(
                {"id": learning_path_id},
                {"$set": updates}
            )
            return result.modified_count > 0
        except Exception as e:
            logging.error(f"MongoDB error updating learning path: {e}")
            return False

    def delete_learning_path(self, learning_path_id: str) -> bool:
        """Delete a learning path from MongoDB."""
        if self.learning_paths is None:
            logging.error("MongoDB connection not available")
            return False

        try:
            result = self.learning_paths.delete_one({"id": learning_path_id})
            return result.deleted_count > 0
        except Exception as e:
            logging.error(f"MongoDB error deleting learning path: {e}")
            return False

    def list_learning_paths(self, skip: int = 0, limit: int = 100,
                            filters: Optional[Dict[str, Any]] = None,
                            sort_by: str = "created_at", sort_desc: bool = True) -> List[Dict[str, Any]]:
        """List learning paths with pagination, filtering and sorting."""
        if self.learning_paths is None:
            logging.error("MongoDB connection not available")
            return []

        try:
            query = filters or {}
            sort_direction = DESCENDING if sort_desc else ASCENDING
            cursor = self.learning_paths.find(query).sort(sort_by, sort_direction).skip(skip).limit(limit)
            return list(cursor)
        except Exception as e:
            logging.error(f"MongoDB error listing learning paths: {e}")
            return []
    
    def count_learning_paths(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count learning paths matching the given filters."""
        if self.learning_paths is None:
            logging.error("MongoDB connection not available")
            return 0

        try:
            query = filters or {}
            return self.learning_paths.count_documents(query)
        except Exception as e:
            logging.error(f"MongoDB error counting learning paths: {e}")
            return 0
    
    # Activity Operations
    
    def create_activity(self, activity: Activity) -> bool:
        """Create an activity document in MongoDB."""
        if self.activity is None:
            logging.error("MongoDB connection not available")
            return False

        try:
            activity_dict = activity.dict()
            result = self.activity.insert_one(activity_dict)
            return result.acknowledged
        except Exception as e:
            logging.error(f"MongoDB error creating activity: {e}")
            return False

    def list_activities(self, skip: int = 0, limit: int = 100,
                        filters: Optional[Dict[str, Any]] = None,
                        sort_by: str = "timestamp", sort_desc: bool = True) -> List[Dict[str, Any]]:
        """List activities with pagination, filtering and sorting."""
        if self.activity is None:
            logging.error("MongoDB connection not available")
            return []

        try:
            query = filters or {}
            sort_direction = DESCENDING if sort_desc else ASCENDING
            cursor = self.activity.find(query).sort(sort_by, sort_direction).skip(skip).limit(limit)
            return list(cursor)
        except Exception as e:
            logging.error(f"MongoDB error listing activities: {e}")
            return []

    def count_activities(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count activities matching the given filters."""
        if self.activity is None:
            logging.error("MongoDB connection not available")
            return 0

        try:
            query = filters or {}
            return self.activity.count_documents(query)
        except Exception as e:
            logging.error(f"MongoDB error counting activities: {e}")
            return 0
    
    # Cache Operations
    
    def set_cache(self, key: str, value: Any, expires_at: datetime) -> bool:
        """Set a cache entry in MongoDB."""
        if self.cache is None:
            logging.error("MongoDB connection not available")
            return False

        try:
            cache_doc = {
                "key": key,
                "value": value,
                "expires_at": expires_at,
                "created_at": datetime.now()
            }
            result = self.cache.update_one(
                {"key": key},
                {"$set": cache_doc},
                upsert=True
            )
            return result.acknowledged
        except Exception as e:
            logging.error(f"MongoDB error setting cache: {e}")
            return False

    def get_cache(self, key: str) -> Optional[Any]:
        """Retrieve a cache entry from MongoDB."""
        if self.cache is None:
            logging.error("MongoDB connection not available")
            return None

        try:
            doc = self.cache.find_one({"key": key})
            return doc.get("value") if doc else None
        except Exception as e:
            logging.error(f"MongoDB error retrieving cache: {e}")
            return None

    def delete_cache(self, key: str) -> bool:
        """Delete a cache entry from MongoDB."""
        if self.cache is None:
            logging.error("MongoDB connection not available")
            return False

        try:
            result = self.cache.delete_one({"key": key})
            return result.deleted_count > 0
        except Exception as e:
            logging.error(f"MongoDB error deleting cache: {e}")
            return False
    
    # Domain Operations
    
    def create_domain(self, domain: Dict[str, Any]) -> bool:
        """Create a domain document in MongoDB."""
        if self.domains is None:
            logging.error("MongoDB connection not available")
            return False

        try:
            result = self.domains.insert_one(domain)
            return result.acknowledged
        except DuplicateKeyError:
            logging.error("Domain already exists")
            return False
        except Exception as e:
            logging.error(f"MongoDB error creating domain: {e}")
            return False

    def get_domain(self, domain_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a domain document from MongoDB by ID."""
        if self.domains is None:
            logging.error("MongoDB connection not available")
            return None

        try:
            result = self.domains.find_one({"id": domain_id})
            return result
        except Exception as e:
            logging.error(f"MongoDB error retrieving domain: {e}")
            return None

    def update_domain(self, domain_id: str, updates: Dict[str, Any]) -> bool:
        """Update a domain document in MongoDB."""
        if self.domains is None:
            logging.error("MongoDB connection not available")
            return False

        try:
            if "updated_at" not in updates:
                updates["updated_at"] = datetime.now()
            result = self.domains.update_one(
                {"id": domain_id},
                {"$set": updates}
            )
            return result.modified_count > 0
        except Exception as e:
            logging.error(f"MongoDB error updating domain: {e}")
            return False

    @retry_on_exception(max_retries=2)
    @log_execution_time
    def delete_domain(self, domain_id: str) -> bool:
        """Delete a domain document from MongoDB.
        
        Args:
            domain_id: ID of the domain to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if not self._check_connection() or self.domains is None:
            logging.error("MongoDB connection not available")
            return False
            
        if not domain_id:
            logging.error("Cannot delete domain: domain_id is None or empty")
            return False

        try:
            result = self.domains.delete_one({"id": domain_id})
            
            # Update metrics
            self._metrics["operations_executed"] += 1
            self._metrics["last_operation_time"] = datetime.now().isoformat()
            
            return result.deleted_count > 0
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logging.error(f"MongoDB error deleting domain {domain_id}: {e}")
            logging.debug(traceback.format_exc())
            return False

    @retry_on_exception(max_retries=1)
    @log_execution_time
    def list_domains(self, skip: int = 0, limit: int = 100,
                     filters: Optional[Dict[str, Any]] = None,
                     sort_by: str = "created_at", sort_desc: bool = True) -> List[Dict[str, Any]]:
        """List domains with pagination, filtering and sorting.
        
        Args:
            skip: Number of domains to skip
            limit: Maximum number of domains to return
            filters: Optional filters to apply
            sort_by: Field to sort by
            sort_desc: Whether to sort in descending order
            
        Returns:
            List of domain dictionaries
        """
        if not self._check_connection() or self.domains is None:
            logging.error("MongoDB connection not available")
            return []

        try:
            # Sanitize inputs
            skip = max(0, skip)
            limit = max(1, min(100, limit))  # Limit between 1 and 100
            
            # Prepare query
            query = filters or {}
            
            # Convert any enum values in filters
            if filters:
                query = self._convert_enum_values(query)
            
            # Set sort direction
            sort_direction = DESCENDING if sort_desc else ASCENDING
            
            # Validate the sort field exists in the schema
            allowed_sort_fields = {"id", "name", "created_at", "updated_at"}
            if sort_by not in allowed_sort_fields:
                logging.warning(f"Invalid sort field: {sort_by}, defaulting to created_at")
                sort_by = "created_at"
            
            # Execute the query
            cursor = self.domains.find(query).sort(sort_by, sort_direction).skip(skip).limit(limit)
            
            # Convert cursor to list
            results = list(cursor)
            
            # Update metrics
            self._metrics["operations_executed"] += 1
            self._metrics["last_operation_time"] = datetime.now().isoformat()
            
            logging.debug(f"Listed {len(results)} domains with skip={skip}, limit={limit}")
            return results
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logging.error(f"MongoDB error listing domains: {e}")
            logging.debug(traceback.format_exc())
            return []

    @retry_on_exception(max_retries=1)
    @log_execution_time
    def count_domains(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count domains matching the given filters.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            Number of matching domains
        """
        if not self._check_connection() or self.domains is None:
            logging.error("MongoDB connection not available")
            return 0

        try:
            # Prepare query
            query = filters or {}
            
            # Convert any enum values in filters
            if filters:
                query = self._convert_enum_values(query)
            
            count = self.domains.count_documents(query)
            
            # Update metrics
            self._metrics["operations_executed"] += 1
            self._metrics["last_operation_time"] = datetime.now().isoformat()
            
            return count
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logging.error(f"MongoDB error counting domains: {e}")
            logging.debug(traceback.format_exc())
            return 0

    @retry_on_exception(max_retries=1)
    @log_execution_time
    def search_domains(self, query: str, limit: int = 20,
                       filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search domains using text search.
        
        Args:
            query: Text to search for
            limit: Maximum number of results to return
            filters: Optional filters to apply
            
        Returns:
            List of matching domain dictionaries
        """
        if not self._check_connection() or self.domains is None:
            logging.error("MongoDB connection not available")
            return []
            
        if not query or not query.strip():
            logging.warning("Empty search query provided")
            return []

        try:
            # Sanitize inputs
            limit = max(1, min(100, limit))  # Limit between 1 and 100
            
            # Prepare search query
            search_query = {"$text": {"$search": query}}
            
            # Add additional filters if provided
            if filters:
                # Convert any enum values in filters
                processed_filters = self._convert_enum_values(filters)
                
                # Combine with text search query
                for key, value in processed_filters.items():
                    search_query[key] = value
            
            # Execute the query with text score for sorting
            results = self.domains.find(
                search_query,
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)
            
            # Convert cursor to list
            result_list = list(results)
            
            # Update metrics
            self._metrics["operations_executed"] += 1
            self._metrics["last_operation_time"] = datetime.now().isoformat()
            
            logging.debug(f"Text search for '{query}' found {len(result_list)} domains")
            return result_list
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logging.error(f"MongoDB error searching domains: {e}")
            logging.debug(traceback.format_exc())
            return []
    
    # Analytics Operations
    
    @retry_on_exception(max_retries=1)
    @log_execution_time
    def store_analytics(self, analytics_data: Dict[str, Any]) -> bool:
        """Store analytics data in MongoDB.
        
        Args:
            analytics_data: Analytics data to store
            
        Returns:
            True if storage was successful, False otherwise
        """
        if not self._check_connection() or self.analytics is None:
            logging.error("MongoDB connection not available")
            return False

        try:
            # Add timestamp if not present
            if "timestamp" not in analytics_data:
                analytics_data["timestamp"] = datetime.now()
                
            # Add unique ID if not present
            if "id" not in analytics_data:
                analytics_data["id"] = str(uuid.uuid4())
            
            result = self.analytics.insert_one(analytics_data)
            
            # Update metrics
            self._metrics["operations_executed"] += 1
            self._metrics["last_operation_time"] = datetime.now().isoformat()
            
            return result.acknowledged
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logging.error(f"MongoDB error storing analytics: {e}")
            logging.debug(traceback.format_exc())
            return False

    @retry_on_exception(max_retries=1)
    @log_execution_time
    def query_analytics(self, query: Dict[str, Any], 
                       aggregate: bool = False,
                       pipeline: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Query analytics data from MongoDB.
        
        Args:
            query: Query filter to apply
            aggregate: Whether to use aggregation framework
            pipeline: Optional aggregation pipeline (used if aggregate is True)
            
        Returns:
            List of matching analytics documents or aggregation results
        """
        if not self._check_connection() or self.analytics is None:
            logging.error("MongoDB connection not available")
            return []

        try:
            if aggregate and pipeline:
                # Use aggregation framework
                results = list(self.analytics.aggregate(pipeline))
            else:
                # Use regular find
                results = list(self.analytics.find(query).sort("timestamp", DESCENDING).limit(1000))
            
            # Update metrics
            self._metrics["operations_executed"] += 1
            self._metrics["last_operation_time"] = datetime.now().isoformat()
            
            return results
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logging.error(f"MongoDB error querying analytics: {e}")
            logging.debug(traceback.format_exc())
            return []

    @retry_on_exception(max_retries=1)
    @log_execution_time
    def get_analytics_summary(self, timeframe_days: int = 30) -> Dict[str, Any]:
        """Get summarized analytics data.
        
        Args:
            timeframe_days: Number of days to include in the summary
            
        Returns:
            Dictionary with summary statistics
        """
        if not self._check_connection() or self.analytics is None:
            logging.error("MongoDB connection not available")
            return {}

        try:
            # Calculate start date for timeframe
            start_date = datetime.now() - timedelta(days=timeframe_days)
            
            # Aggregation pipeline for summary stats
            pipeline = [
                {"$match": {"timestamp": {"$gte": start_date}}},
                {"$group": {
                    "_id": "$event_type",
                    "count": {"$sum": 1},
                    "avg_duration": {"$avg": "$duration"},
                    "users": {"$addToSet": "$user_id"}
                }},
                {"$project": {
                    "count": 1,
                    "avg_duration": 1,
                    "unique_users": {"$size": "$users"}
                }},
                {"$sort": {"count": -1}}
            ]
            
            results = list(self.analytics.aggregate(pipeline))
            
            # Format the results
            summary = {
                "timeframe_days": timeframe_days,
                "timestamp": datetime.now().isoformat(),
                "events": {}
            }
            
            for result in results:
                event_type = result["_id"]
                if event_type:
                    summary["events"][event_type] = {
                        "count": result["count"],
                        "avg_duration_ms": result.get("avg_duration", 0),
                        "unique_users": result.get("unique_users", 0)
                    }
            
            # Update metrics
            self._metrics["operations_executed"] += 1
            self._metrics["last_operation_time"] = datetime.now().isoformat()
            
            return summary
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logging.error(f"MongoDB error getting analytics summary: {e}")
            logging.debug(traceback.format_exc())
            return {}
    
    # Generic Entity Operations
    
    @retry_on_exception(max_retries=2)
    @log_execution_time
    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity from MongoDB by ID, checking all collections.
        
        Args:
            entity_id: ID of the entity to delete
            
        Returns:
            True if any deletion was successful, False otherwise
        """
        if not self._check_connection():
            logging.error("MongoDB connection not available")
            return False
            
        if not entity_id:
            logging.error("Cannot delete entity: entity_id is None or empty")
            return False

        try:
            deleted = False
            
            # Try to delete from each collection
            collections = [
                self.concepts, self.relationships, self.learning_paths, 
                self.domains, self.activity, self.cache
            ]
            
            for collection in collections:
                if collection is not None:
                    result = collection.delete_one({"id": entity_id})
                    if result.deleted_count > 0:
                        deleted = True
                        logging.debug(f"Deleted entity {entity_id} from {collection.name}")
            
            # Update metrics
            self._metrics["operations_executed"] += 1
            self._metrics["last_operation_time"] = datetime.now().isoformat()
            
            return deleted
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logging.error(f"MongoDB error deleting entity {entity_id}: {e}")
            logging.debug(traceback.format_exc())
            return False

    @retry_on_exception(max_retries=1)
    @log_execution_time
    def find_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Find an entity by ID across all collections.
        
        Args:
            entity_id: ID of the entity to find
            
        Returns:
            Entity data and collection name, or None if not found
        """
        if not self._check_connection():
            logging.error("MongoDB connection not available")
            return None
            
        if not entity_id:
            logging.error("Cannot find entity: entity_id is None or empty")
            return None

        try:
            collections = [
                self.concepts, self.relationships, self.learning_paths, 
                self.domains, self.activity, self.cache
            ]
            
            for collection in collections:
                if collection is not None:
                    result = collection.find_one({"id": entity_id})
                    if result:
                        result["_collection"] = collection.name
                        
                        # Update metrics
                        self._metrics["operations_executed"] += 1
                        self._metrics["last_operation_time"] = datetime.now().isoformat()
                        
                        return result
            
            return None
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logging.error(f"MongoDB error finding entity {entity_id}: {e}")
            logging.debug(traceback.format_exc())
            return None

    @retry_on_exception(max_retries=1)
    @log_execution_time
    def user_entities(self, user_id: str) -> Dict[str, List[str]]:
        """Find all entities associated with a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Dictionary of entity types and their IDs
        """
        if not self._check_connection():
            logging.error("MongoDB connection not available")
            return {}
            
        if not user_id:
            logging.error("Cannot get user entities: user_id is None or empty")
            return {}

        try:
            # Check activities for created entities
            pipeline = [
                {"$match": {
                    "user_id": user_id,
                    "activity_type": "create"
                }},
                {"$group": {
                    "_id": "$entity_type",
                    "entity_ids": {"$addToSet": "$entity_id"}
                }}
            ]
            
            results = {}
            if self.activity is not None:
                activity_results = list(self.activity.aggregate(pipeline))
                for result in activity_results:
                    entity_type = result["_id"]
                    if entity_type:
                        results[entity_type] = result["entity_ids"]
            
            # Update metrics
            self._metrics["operations_executed"] += 1
            self._metrics["last_operation_time"] = datetime.now().isoformat()
            
            return results
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logging.error(f"MongoDB error getting user entities: {e}")
            logging.debug(traceback.format_exc())
            return {}
    
    # User Management Operations
    
    @retry_on_exception(max_retries=2)
    @log_execution_time
    def create_user(self, user_data: Dict[str, Any]) -> bool:
        """Create a user document in MongoDB.
        
        Args:
            user_data: User data to create
            
        Returns:
            True if creation was successful, False otherwise
        """
        if not self._check_connection() or self.users is None:
            logging.error("MongoDB connection not available")
            return False
            
        if not user_data.get("id"):
            logging.error("Cannot create user: missing required id field")
            return False

        try:
            # Set timestamps if not present
            if "created_at" not in user_data:
                user_data["created_at"] = datetime.now()
            if "updated_at" not in user_data:
                user_data["updated_at"] = user_data["created_at"]
            
            result = self.users.insert_one(user_data)
            
            # Update metrics
            self._metrics["operations_executed"] += 1
            self._metrics["last_operation_time"] = datetime.now().isoformat()
            
            return result.acknowledged
        except DuplicateKeyError:
            logging.error(f"User with ID {user_data.get('id')} already exists")
            return False
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logging.error(f"MongoDB error creating user: {e}")
            logging.debug(traceback.format_exc())
            return False

    @retry_on_exception(max_retries=1)
    @log_execution_time
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a user document from MongoDB by ID.
        
        Args:
            user_id: ID of the user to retrieve
            
        Returns:
            Dictionary representing the user or None if not found
        """
        if not self._check_connection() or self.users is None:
            logging.error("MongoDB connection not available")
            return None
            
        if not user_id:
            logging.error("Cannot get user: user_id is None or empty")
            return None

        try:
            result = self.users.find_one({"id": user_id})
            
            # Update metrics
            self._metrics["operations_executed"] += 1
            self._metrics["last_operation_time"] = datetime.now().isoformat()
            
            return result
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logging.error(f"MongoDB error retrieving user {user_id}: {e}")
            logging.debug(traceback.format_exc())
            return None

    @retry_on_exception(max_retries=2)
    @log_execution_time
    def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update a user document in MongoDB.
        
        Args:
            user_id: ID of the user to update
            updates: Dictionary of updates to apply
            
        Returns:
            True if update was successful, False otherwise
        """
        if not self._check_connection() or self.users is None:
            logging.error("MongoDB connection not available")
            return False
            
        if not user_id:
            logging.error("Cannot update user: user_id is None or empty")
            return False

        try:
            if "updated_at" not in updates:
                updates["updated_at"] = datetime.now()
                
            result = self.users.update_one(
                {"id": user_id},
                {"$set": updates}
            )
            
            # Update metrics
            self._metrics["operations_executed"] += 1
            self._metrics["last_operation_time"] = datetime.now().isoformat()
            
            return result.modified_count > 0
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logging.error(f"MongoDB error updating user {user_id}: {e}")
            logging.debug(traceback.format_exc())
            return False

    @retry_on_exception(max_retries=2)
    @log_execution_time
    def delete_user(self, user_id: str) -> bool:
        """Delete a user document from MongoDB.
        
        Args:
            user_id: ID of the user to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if not self._check_connection() or self.users is None:
            logging.error("MongoDB connection not available")
            return False
            
        if not user_id:
            logging.error("Cannot delete user: user_id is None or empty")
            return False

        try:
            result = self.users.delete_one({"id": user_id})
            
            # Update metrics
            self._metrics["operations_executed"] += 1
            self._metrics["last_operation_time"] = datetime.now().isoformat()
            
            return result.deleted_count > 0
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logging.error(f"MongoDB error deleting user {user_id}: {e}")
            logging.debug(traceback.format_exc())
            return False

    @retry_on_exception(max_retries=1)
    @log_execution_time
    def list_users(self, skip: int = 0, limit: int = 100,
                  filters: Optional[Dict[str, Any]] = None,
                  sort_by: str = "created_at", sort_desc: bool = True) -> List[Dict[str, Any]]:
        """List users with pagination, filtering and sorting.
        
        Args:
            skip: Number of users to skip
            limit: Maximum number of users to return
            filters: Optional filters to apply
            sort_by: Field to sort by
            sort_desc: Whether to sort in descending order
            
        Returns:
            List of user dictionaries
        """
        if not self._check_connection() or self.users is None:
            logging.error("MongoDB connection not available")
            return []

        try:
            # Sanitize inputs
            skip = max(0, skip)
            limit = max(1, min(100, limit))  # Limit between 1 and 100
            
            # Prepare query
            query = filters or {}
            
            # Set sort direction
            sort_direction = DESCENDING if sort_desc else ASCENDING
            
            # Validate the sort field exists in the schema
            allowed_sort_fields = {"id", "username", "email", "created_at", "updated_at"}
            if sort_by not in allowed_sort_fields:
                logging.warning(f"Invalid sort field: {sort_by}, defaulting to created_at")
                sort_by = "created_at"
            
            # Execute the query
            cursor = self.users.find(query).sort(sort_by, sort_direction).skip(skip).limit(limit)
            
            # Convert cursor to list
            results = list(cursor)
            
            # Update metrics
            self._metrics["operations_executed"] += 1
            self._metrics["last_operation_time"] = datetime.now().isoformat()
            
            logging.debug(f"Listed {len(results)} users with skip={skip}, limit={limit}")
            return results
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logging.error(f"MongoDB error listing users: {e}")
            logging.debug(traceback.format_exc())
            return []

    @retry_on_exception(max_retries=1)
    @log_execution_time
    def find_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Find a user by email address.
        
        Args:
            email: Email address to search for
            
        Returns:
            User document or None if not found
        """
        if not self._check_connection() or self.users is None:
            logging.error("MongoDB connection not available")
            return None
            
        if not email:
            logging.error("Cannot find user: email is None or empty")
            return None

        try:
            # Case-insensitive search
            result = self.users.find_one({"email": {"$regex": f"^{email}$", "$options": "i"}})
            
            # Update metrics
            self._metrics["operations_executed"] += 1
            self._metrics["last_operation_time"] = datetime.now().isoformat()
            
            return result
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logging.error(f"MongoDB error finding user by email {email}: {e}")
            logging.debug(traceback.format_exc())
            return None

    @retry_on_exception(max_retries=1)
    @log_execution_time
    def find_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Find a user by username.
        
        Args:
            username: Username to search for
            
        Returns:
            User document or None if not found
        """
        if not self._check_connection() or self.users is None:
            logging.error("MongoDB connection not available")
            return None
            
        if not username:
            logging.error("Cannot find user: username is None or empty")
            return None

        try:
            # Case-insensitive search
            result = self.users.find_one({"username": {"$regex": f"^{username}$", "$options": "i"}})
            
            # Update metrics
            self._metrics["operations_executed"] += 1
            self._metrics["last_operation_time"] = datetime.now().isoformat()
            
            return result
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logging.error(f"MongoDB error finding user by username {username}: {e}")
            logging.debug(traceback.format_exc())
            return None
    
    # Backup and Restore Operations
    
    @retry_on_exception(max_retries=1)
    @log_execution_time
    def export_collection(self, collection_name: str) -> List[Dict[str, Any]]:
        """Export all documents from a collection.
        
        Args:
            collection_name: Name of the collection to export
            
        Returns:
            List of documents in the collection
        """
        if not self._check_connection() or not self.db:
            logging.error("MongoDB connection not available")
            return []

        try:
            # Get the collection
            collection = self.db[collection_name]
            
            # Export all documents
            documents = list(collection.find({}))
            
            # Update metrics
            self._metrics["operations_executed"] += 1
            self._metrics["last_operation_time"] = datetime.now().isoformat()
            
            logging.info(f"Exported {len(documents)} documents from {collection_name}")
            return documents
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logging.error(f"MongoDB error exporting collection {collection_name}: {e}")
            logging.debug(traceback.format_exc())
            return []

    @retry_on_exception(max_retries=2)
    @log_execution_time
    def import_collection(self, collection_name: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Import documents into a collection.
        
        Args:
            collection_name: Name of the collection to import into
            documents: List of documents to import
            
        Returns:
            Dictionary with import results
        """
        if not self._check_connection() or not self.db:
            logging.error("MongoDB connection not available")
            return {"success": False, "inserted": 0, "errors": 1}
            
        if not documents:
            return {"success": True, "inserted": 0, "errors": 0}

        try:
            # Get the collection
            collection = self.db[collection_name]
            
            # Remove _id fields if present to avoid duplicate key errors
            for doc in documents:
                if "_id" in doc:
                    del doc["_id"]
            
            # Use insert_many with ordered=False to continue on error
            result = collection.insert_many(documents, ordered=False)
            
            # Update metrics
            self._metrics["operations_executed"] += 1
            self._metrics["last_operation_time"] = datetime.now().isoformat()
            
            logging.info(f"Imported {len(result.inserted_ids)}/{len(documents)} documents into {collection_name}")
            return {
                "success": True,
                "inserted": len(result.inserted_ids),
                "errors": len(documents) - len(result.inserted_ids)
            }
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logging.error(f"MongoDB error importing collection {collection_name}: {e}")
            logging.debug(traceback.format_exc())
            return {"success": False, "inserted": 0, "errors": len(documents)}