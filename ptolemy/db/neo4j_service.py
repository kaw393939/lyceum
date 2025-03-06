"""
Ptolemy Knowledge Map System - Neo4j Database Service
====================================================
Provides graph database operations for the knowledge map using Neo4j. 
"""

import logging
import time
import uuid
import traceback
import threading
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable, TypeVar
from datetime import datetime
from contextlib import contextmanager
from functools import wraps

from neo4j import GraphDatabase, Driver, Session, Transaction, Result
from neo4j.exceptions import ServiceUnavailable, ClientError, DatabaseError, TransientError

from config import Neo4jConfig
from models import (
    Concept, Relationship, ValidationResult, ValidationIssue,
    ConceptType, RelationshipType, DifficultyLevel
)

# Configure module-level logger
logger = logging.getLogger("neo4j.service")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Type hint for reusable decorator
F = TypeVar('F', bound=Callable[..., Any])

def retry_on_exception(max_retries: int = 3, initial_delay: float = 2.0,
                      backoff_factor: float = 2.0, 
                      exceptions: Tuple = (ServiceUnavailable, TransientError)) -> Callable[[F], F]:
    """Decorator to retry functions on exception with exponential backoff."""
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = initial_delay
            
            for attempt in range(max_retries + 1):
                try:
                    if attempt > 0:
                        logger.debug(f"Retry attempt {attempt} for {func.__name__}")
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        # Log retry attempt
                        logger.warning(
                            f"Attempt {attempt+1}/{max_retries+1} for {func.__name__} failed with: {str(e)}. "
                            f"Retrying in {delay:.2f}s"
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        # Log final failure
                        logger.error(
                            f"All {max_retries+1} attempts for {func.__name__} failed. "
                            f"Last error: {str(e)}"
                        )
            
            # If we're here, all retries failed
            if last_exception:
                raise last_exception
        return wrapper
    return decorator

def log_execution_time(func: F) -> F:
    """Decorator to log execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Only log if execution takes more than 100ms
            if execution_time > 0.1:
                logger.debug(f"{func.__name__} executed in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f}s with error: {str(e)}")
            raise
    return wrapper

class Neo4jService:
    """Service for graph database operations using Neo4j."""
    
    def __init__(self, config: Neo4jConfig):
        """Initialize the Neo4j service with the given configuration.
        
        Args:
            config: Configuration for Neo4j connection and database
        """
        self.config = config
        self.driver = None
        self._is_connected = False
        self._lock = threading.RLock()
        
        # Service state
        self._schema_initialized = False
        self._last_error = None
        
        # Metrics for monitoring
        self._metrics = {
            "connection_attempts": 0,
            "successful_connections": 0,
            "queries_executed": 0,
            "errors": 0,
            "last_error_time": None,
            "last_operation_time": datetime.now().isoformat()
        }
        
        # Connect to Neo4j
        self.connect()
        if self._is_connected:
            self._setup_schema()
    
    @staticmethod
    def _format_datetime(dt: Optional[datetime]) -> Optional[str]:
        """Convert a datetime object to an ISO string.
        
        Args:
            dt: Datetime object to format
            
        Returns:
            ISO-formatted string or None
        """
        return dt.isoformat() if dt else None

    @staticmethod
    def _parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
        """Convert an ISO datetime string back to a datetime object.
        
        Args:
            dt_str: ISO-formatted datetime string
            
        Returns:
            Datetime object or None
        """
        if not dt_str:
            return None
            
        try:
            return datetime.fromisoformat(dt_str)
        except (ValueError, TypeError):
            try:
                # Try alternative formats if standard fromisoformat fails
                import dateutil.parser
                return dateutil.parser.parse(dt_str)
            except (ImportError, ValueError, TypeError):
                logger.warning(f"Could not parse datetime string: {dt_str}")
                return None

    @retry_on_exception(max_retries=3, initial_delay=2.0)
    def connect(self) -> bool:
        """Establish connection to Neo4j database with retry logic.
        
        Returns:
            bool: True if connected successfully, False otherwise
        """
        with self._lock:
            try:
                logger.info(f"Connecting to Neo4j at {self.config.uri}")
                self._metrics["connection_attempts"] += 1
                
                # Close existing driver if there is one
                if self.driver:
                    try:
                        self.driver.close()
                    except Exception as e:
                        logger.warning(f"Error closing existing driver: {e}")
                
                # Create new driver
                self.driver = GraphDatabase.driver(
                    self.config.uri,
                    auth=(self.config.user, self.config.password),
                    connection_timeout=self.config.connection_timeout,
                    max_connection_lifetime=self.config.max_connection_lifetime,
                    max_connection_pool_size=self.config.max_connection_pool_size,
                    connection_acquisition_timeout=self.config.connection_acquisition_timeout,
                    keep_alive=True
                )
                
                # Verify connection
                with self.driver.session(database=self.config.database) as session:
                    result = session.run("RETURN 1 as test")
                    result.single()
                
                self._is_connected = True
                self._metrics["successful_connections"] += 1
                self._last_error = None
                logger.info("Connected to Neo4j successfully")
                return True
            except Exception as e:
                self._is_connected = False
                self._last_error = str(e)
                self._metrics["errors"] += 1
                self._metrics["last_error_time"] = datetime.now().isoformat()
                logger.error(f"Neo4j connection failed: {e}")
                logger.debug(traceback.format_exc())
                self.driver = None
                return False

    @retry_on_exception(max_retries=1)
    @log_execution_time
    def _setup_schema(self) -> bool:
        """Set up Neo4j schema, constraints and indexes.
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        if not self.driver or not self._is_connected:
            logger.error("Cannot set up Neo4j schema: No connection")
            return False
        
        if self._schema_initialized:
            return True

        try:
            with self._get_session() as session:
                # First check Neo4j version to determine constraint syntax
                version_query = "CALL dbms.components() YIELD name, versions RETURN versions[0] as version"
                try:
                    version_result = session.run(version_query).single()
                    version = version_result["version"] if version_result else "4.0.0"  # Default to Neo4j 4.x
                    neo4j_4x = version.startswith("4.")
                    logger.info(f"Detected Neo4j version: {version}")
                except:
                    # If version check fails, assume Neo4j 4.x
                    neo4j_4x = True
                    logger.warning("Could not detect Neo4j version, assuming 4.x")
                
                # Set up constraints with appropriate syntax
                if neo4j_4x:
                    # Neo4j 4.x syntax
                    constraints = [
                        "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
                        "CREATE CONSTRAINT relationship_id IF NOT EXISTS FOR ()-[r:RELATES_TO]-() REQUIRE r.id IS UNIQUE"
                    ]
                else:
                    # Neo4j 5.x+ syntax
                    constraints = [
                        "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
                        "CREATE CONSTRAINT relationship_id IF NOT EXISTS FOR ()-[r:RELATES_TO]-() REQUIRE r.id IS UNIQUE"
                    ]
                
                # Create indexes for performance
                indexes = [
                    "CREATE INDEX concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name)",
                    "CREATE INDEX concept_type IF NOT EXISTS FOR (c:Concept) ON (c.concept_type)",
                    "CREATE INDEX concept_difficulty IF NOT EXISTS FOR (c:Concept) ON (c.difficulty)",
                    "CREATE INDEX concept_parent IF NOT EXISTS FOR (c:Concept) ON (c.parent_id)",
                    "CREATE INDEX relationship_type IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.relationship_type)"
                ]
                
                # Apply constraints and indexes
                for constraint in constraints:
                    try:
                        session.run(constraint)
                    except ClientError as e:
                        # Skip if constraint already exists or similar error
                        logger.warning(f"Constraint creation issue: {e}")
                        if "already exists" not in str(e):
                            # Try alternative syntax if not a "already exists" error
                            try:
                                alternative = constraint.replace("CREATE CONSTRAINT", "CREATE CONSTRAINT").replace("REQUIRE", "REQUIRE")
                                session.run(alternative)
                                logger.info(f"Created constraint with alternative syntax: {alternative}")
                            except Exception as alt_e:
                                logger.warning(f"Alternative constraint creation also failed: {alt_e}")
                
                for index in indexes:
                    try:
                        session.run(index)
                    except ClientError as e:
                        # Skip if index already exists or similar error
                        logger.warning(f"Index creation issue: {e}")
                
                self._schema_initialized = True
                logger.info("Neo4j schema setup complete")
                return True
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Failed to set up Neo4j schema: {e}")
            logger.debug(traceback.format_exc())
            return False

    def close(self) -> None:
        """Close the Neo4j database connection."""
        with self._lock:
            if self.driver:
                try:
                    self.driver.close()
                    logger.info("Neo4j connection closed")
                except Exception as e:
                    logger.error(f"Error closing Neo4j connection: {e}")
                finally:
                    self.driver = None
                    self._is_connected = False

    @retry_on_exception(max_retries=1)
    @log_execution_time
    def health_check(self) -> Dict[str, Any]:
        """Check Neo4j connection health and return metrics.
        
        Returns:
            Dict containing health status and metrics
        """
        if not self.driver or not self._is_connected:
            # Try to reconnect
            if not self.connect():
                return {
                    "status": "disconnected", 
                    "error": self._last_error or "No connection to Neo4j",
                    "metrics": self._metrics
                }
        
        try:
            with self._get_session() as session:
                # Basic connectivity check
                session.run("RETURN 1").single()
                
                # Get database stats
                concept_count = session.run(
                    "MATCH (c:Concept) RETURN count(c) as count"
                ).single()["count"]
                
                relationship_count = session.run(
                    "MATCH ()-[r:RELATES_TO]->() RETURN count(r) as count"
                ).single()["count"]
                
                # Get database information
                try:
                    db_info = session.run(
                        "CALL dbms.components() YIELD name, versions, edition RETURN name, versions, edition"
                    ).single()
                    
                    version = db_info["versions"][0] if db_info and db_info["versions"] else "unknown"
                    edition = db_info["edition"] if db_info else "unknown"
                except Exception as e:
                    logger.warning(f"Could not get Neo4j version information: {e}")
                    version = "unknown"
                    edition = "unknown"
                
                return {
                    "status": "connected",
                    "version": version,
                    "edition": edition,
                    "concept_count": concept_count,
                    "relationship_count": relationship_count,
                    "schema_initialized": self._schema_initialized,
                    "database": self.config.database,
                    "metrics": self._metrics
                }
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Neo4j health check failed: {e}")
            logger.debug(traceback.format_exc())
            
            # Try to reconnect
            self.connect()
            
            return {
                "status": "error", 
                "error": str(e),
                "metrics": self._metrics
            }

    @contextmanager
    def _get_session(self):
        """Context manager for Neo4j session handling with error management.
        
        Yields:
            Active Neo4j session
            
        Raises:
            Exception: If no connection is available or session creation fails
        """
        if not self.driver or not self._is_connected:
            # Try to reconnect
            if not self.connect():
                raise Exception("No connection to Neo4j database")
        
        session = None
        try:
            session = self.driver.session(database=self.config.database)
            yield session
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable: {e}")
            
            # Attempt to reconnect
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            self.connect()
            
            raise
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Neo4j session error: {e}")
            logger.debug(traceback.format_exc())
            raise
        finally:
            if session:
                try:
                    session.close()
                except Exception as e:
                    logger.warning(f"Error closing Neo4j session: {e}")

    def _run_query(self, query: str, params: Dict[str, Any] = None, database: str = None) -> Result:
        """Run a Neo4j query with error handling and metrics.
        
        Args:
            query: Cypher query to run
            params: Query parameters
            database: Optional database name override
            
        Returns:
            Query result
            
        Raises:
            Exception: If query execution fails
        """
        if not self.driver or not self._is_connected:
            # Try to reconnect
            if not self.connect():
                raise Exception("No connection to Neo4j database")
        
        params = params or {}
        db_name = database or self.config.database
        
        try:
            with self.driver.session(database=db_name) as session:
                start_time = time.time()
                result = session.run(query, params)
                execution_time = time.time() - start_time
                
                self._metrics["queries_executed"] += 1
                self._metrics["last_operation_time"] = datetime.now().isoformat()
                
                # Log slow queries
                if execution_time > 1.0:
                    truncated_query = query[:200] + "..." if len(query) > 200 else query
                    logger.warning(f"Slow query ({execution_time:.2f}s): {truncated_query}")
                
                return result
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Neo4j query error: {e}")
            logger.debug(f"Query: {query}")
            logger.debug(f"Params: {params}")
            logger.debug(traceback.format_exc())
            raise

    # Concept Operations

    @retry_on_exception(max_retries=2)
    @log_execution_time
    def create_concept(self, concept: Concept) -> Optional[str]:
        """Create a concept node in Neo4j.
        
        Args:
            concept: Concept to create
            
        Returns:
            ID of created concept or None if creation failed
        """
        if not self.driver or not self._is_connected:
            if not self.connect():
                logger.error("Neo4j connection not available")
                return None
        
        try:
            with self._get_session() as session:
                result = session.execute_write(self._create_concept_tx, concept)
                return result
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Neo4j error creating concept: {e}")
            logger.debug(traceback.format_exc())
            return None

    def _create_concept_tx(self, tx: Transaction, concept: Concept) -> Optional[str]:
        """Transaction function for creating a concept node.
        
        Args:
            tx: Neo4j transaction
            concept: Concept to create
            
        Returns:
            ID of created concept or None if creation failed
        """
        # Convert concept to dictionary and prepare data
        concept_dict = concept.dict() if hasattr(concept, "dict") else concept
        
        # Format datetime values
        concept_dict["created_at"] = self._format_datetime(concept_dict.get("created_at"))
        concept_dict["updated_at"] = self._format_datetime(concept_dict.get("updated_at"))
        
        # Ensure concept_type is stored as string
        if "concept_type" in concept_dict and hasattr(concept_dict["concept_type"], "value"):
            concept_dict["concept_type"] = concept_dict["concept_type"].value
            
        # Ensure difficulty is stored as string
        if "difficulty" in concept_dict and hasattr(concept_dict["difficulty"], "value"):
            concept_dict["difficulty"] = concept_dict["difficulty"].value
        
        query = """
        CREATE (c:Concept {
            id: $id,
            name: $name,
            description: $description,
            concept_type: $concept_type,
            difficulty: $difficulty,
            parent_id: $parent_id,
            importance: $importance,
            complexity: $complexity,
            keywords: $keywords,
            estimated_learning_time_minutes: $estimated_learning_time_minutes,
            created_at: $created_at,
            updated_at: $updated_at,
            version: $version
        })
        RETURN c.id as id
        """
        
        try:
            result = tx.run(query, concept_dict).single()
            
            # Handle parent relationship if specified
            parent_id = concept_dict.get("parent_id")
            if parent_id:
                parent_query = """
                MATCH (c:Concept {id: $concept_id})
                MATCH (p:Concept {id: $parent_id})
                CREATE (c)-[r:RELATES_TO {
                    id: $rel_id,
                    relationship_type: 'part_of',
                    strength: 1.0,
                    created_at: $created_at
                }]->(p)
                """
                try:
                    tx.run(parent_query, {
                        "concept_id": concept_dict["id"],
                        "parent_id": parent_id,
                        "rel_id": str(uuid.uuid4()),
                        "created_at": concept_dict["created_at"]
                    })
                except Exception as parent_error:
                    logger.warning(f"Error creating parent relationship: {parent_error}")
            
            return result["id"] if result else None
        except Exception as e:
            logger.error(f"Transaction error creating concept: {e}")
            raise

    @retry_on_exception(max_retries=2)
    @log_execution_time
    def update_concept(self, concept_id: str, updates: Dict[str, Any]) -> bool:
        """Update a concept node in Neo4j.
        
        Args:
            concept_id: ID of the concept to update
            updates: Dictionary of updates to apply
            
        Returns:
            True if update was successful, False otherwise
        """
        if not self.driver or not self._is_connected:
            if not self.connect():
                logger.error("Neo4j connection not available")
                return False
        
        try:
            with self._get_session() as session:
                result = session.execute_write(self._update_concept_tx, concept_id, updates)
                return result
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Neo4j error updating concept: {e}")
            logger.debug(traceback.format_exc())
            return False

    def _update_concept_tx(self, tx: Transaction, concept_id: str, updates: Dict[str, Any]) -> bool:
        """Transaction function for updating a concept node.
        
        Args:
            tx: Neo4j transaction
            concept_id: ID of the concept to update
            updates: Dictionary of updates to apply
            
        Returns:
            True if update was successful, False otherwise
        """
        # Format updated_at if present
        if "updated_at" in updates:
            updates["updated_at"] = self._format_datetime(updates["updated_at"])
        
        # Ensure enum values are converted to strings
        for key, value in updates.items():
            if hasattr(value, "value"):
                updates[key] = value.value
        
        # Generate dynamic update clauses (skip parent_id which is handled separately)
        update_clauses = []
        for key, value in updates.items():
            if key != "parent_id":
                update_clauses.append(f"c.{key} = ${key}")
        
        if update_clauses:
            query = f"""
            MATCH (c:Concept {{id: $concept_id}})
            SET {', '.join(update_clauses)}
            RETURN count(c) as count
            """
            params = {"concept_id": concept_id, **updates}
            result = tx.run(query, params).single()
        else:
            result = {"count": 1}  # Default to success if no updates to apply
        
        # Handle parent relationship update if included
        if "parent_id" in updates:
            # Remove old parent relationship
            tx.run("""
            MATCH (c:Concept {id: $concept_id})-[r:RELATES_TO {relationship_type: 'part_of'}]->(:Concept)
            DELETE r
            """, {"concept_id": concept_id})
            
            # Create new parent relationship if not None
            if updates["parent_id"]:
                tx.run("""
                MATCH (c:Concept {id: $concept_id})
                MATCH (p:Concept {id: $parent_id})
                CREATE (c)-[r:RELATES_TO {
                    id: $rel_id,
                    relationship_type: 'part_of',
                    strength: 1.0,
                    created_at: $updated_at
                }]->(p)
                """, {
                    "concept_id": concept_id,
                    "parent_id": updates["parent_id"],
                    "rel_id": str(uuid.uuid4()),
                    "updated_at": updates.get("updated_at", self._format_datetime(datetime.now()))
                })
        
        return result and result["count"] > 0

    @retry_on_exception(max_retries=1)
    @log_execution_time
    def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a concept from Neo4j by ID.
        
        Args:
            concept_id: ID of the concept to retrieve
            
        Returns:
            Dictionary representing the concept or None if not found
        """
        if not self.driver or not self._is_connected:
            if not self.connect():
                logger.error("Neo4j connection not available")
                return None
                
        if not concept_id:
            logger.error("Cannot get concept: concept_id is None or empty")
            return None
        
        try:
            with self._get_session() as session:
                query = """
                MATCH (c:Concept {id: $concept_id})
                RETURN c
                """
                result = session.run(query, {"concept_id": concept_id}).single()
                
                if result:
                    concept_data = dict(result["c"])
                    
                    # Convert string dates back to datetime objects
                    concept_data["created_at"] = self._parse_datetime(concept_data.get("created_at"))
                    concept_data["updated_at"] = self._parse_datetime(concept_data.get("updated_at"))
                    
                    return concept_data
                return None
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Neo4j error retrieving concept {concept_id}: {e}")
            logger.debug(traceback.format_exc())
            return None

    @retry_on_exception(max_retries=2)
    @log_execution_time
    def delete_concept(self, concept_id: str) -> bool:
        """Delete a concept and its relationships from Neo4j.
        
        Args:
            concept_id: ID of the concept to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if not self.driver or not self._is_connected:
            if not self.connect():
                logger.error("Neo4j connection not available")
                return False
                
        if not concept_id:
            logger.error("Cannot delete concept: concept_id is None or empty")
            return False
        
        try:
            with self._get_session() as session:
                result = session.execute_write(self._delete_concept_tx, concept_id)
                return result
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Neo4j error deleting concept {concept_id}: {e}")
            logger.debug(traceback.format_exc())
            return False

    def _delete_concept_tx(self, tx: Transaction, concept_id: str) -> bool:
        """Transaction function for deleting a concept.
        
        Args:
            tx: Neo4j transaction
            concept_id: ID of the concept to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        # Delete all outgoing relationships
        tx.run("""
        MATCH (c:Concept {id: $concept_id})-[r]->()
        DELETE r
        """, {"concept_id": concept_id})
        
        # Delete all incoming relationships
        tx.run("""
        MATCH ()-[r]->(c:Concept {id: $concept_id})
        DELETE r
        """, {"concept_id": concept_id})
        
        # Then delete the concept node
        query = """
        MATCH (c:Concept {id: $concept_id})
        DELETE c
        RETURN count(c) as count
        """
        result = tx.run(query, {"concept_id": concept_id}).single()
        
        return result and result["count"] > 0

    @retry_on_exception(max_retries=2)
    @log_execution_time
    def bulk_create_concepts(self, concepts: List[Concept]) -> Dict[str, Any]:
        """Create multiple concepts in a single transaction.
        
        Args:
            concepts: List of concepts to create
            
        Returns:
            Dictionary with success status and created IDs
        """
        if not self.driver or not self._is_connected:
            if not self.connect():
                logger.error("Neo4j connection not available")
                return {"success": False, "created": [], "message": "No connection to Neo4j"}
                
        if not concepts:
            return {"success": True, "created": [], "message": "No concepts to create"}
        
        try:
            with self._get_session() as session:
                result = session.execute_write(self._bulk_create_concepts_tx, concepts)
                return result
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Neo4j error bulk creating concepts: {e}")
            logger.debug(traceback.format_exc())
            return {"success": False, "created": [], "message": str(e)}

    def _bulk_create_concepts_tx(self, tx: Transaction, concepts: List[Concept]) -> Dict[str, Any]:
        """Transaction function for bulk creating concepts.
        
        Args:
            tx: Neo4j transaction
            concepts: List of concepts to create
            
        Returns:
            Dictionary with success status and created IDs
        """
        created_ids = []
        errors = []
        
        for concept in concepts:
            try:
                # Convert concept to dictionary and prepare data
                concept_dict = concept.dict() if hasattr(concept, "dict") else concept
                
                # Format datetime values
                concept_dict["created_at"] = self._format_datetime(concept_dict.get("created_at"))
                concept_dict["updated_at"] = self._format_datetime(concept_dict.get("updated_at"))
                
                # Ensure concept_type is stored as string
                if "concept_type" in concept_dict and hasattr(concept_dict["concept_type"], "value"):
                    concept_dict["concept_type"] = concept_dict["concept_type"].value
                    
                # Ensure difficulty is stored as string
                if "difficulty" in concept_dict and hasattr(concept_dict["difficulty"], "value"):
                    concept_dict["difficulty"] = concept_dict["difficulty"].value
                
                query = """
                CREATE (c:Concept {
                    id: $id,
                    name: $name,
                    description: $description,
                    concept_type: $concept_type,
                    difficulty: $difficulty,
                    parent_id: $parent_id,
                    importance: $importance,
                    complexity: $complexity,
                    keywords: $keywords,
                    estimated_learning_time_minutes: $estimated_learning_time_minutes,
                    created_at: $created_at,
                    updated_at: $updated_at,
                    version: $version
                })
                RETURN c.id as id
                """
                
                result = tx.run(query, concept_dict).single()
                
                # Handle parent relationship if specified
                parent_id = concept_dict.get("parent_id")
                if parent_id:
                    parent_query = """
                    MATCH (c:Concept {id: $concept_id})
                    MATCH (p:Concept {id: $parent_id})
                    CREATE (c)-[r:RELATES_TO {
                        id: $rel_id,
                        relationship_type: 'part_of',
                        strength: 1.0,
                        created_at: $created_at
                    }]->(p)
                    """
                    try:
                        tx.run(parent_query, {
                            "concept_id": concept_dict["id"],
                            "parent_id": parent_id,
                            "rel_id": str(uuid.uuid4()),
                            "created_at": concept_dict["created_at"]
                        })
                    except Exception as parent_error:
                        logger.warning(f"Error creating parent relationship: {parent_error}")
                
                created_ids.append(concept_dict["id"])
                
            except Exception as e:
                errors.append({"id": concept_dict.get("id"), "error": str(e)})
                logger.error(f"Error creating concept in bulk operation: {e}")
        
        success = len(errors) == 0
        return {
            "success": success, 
            "created": created_ids,
            "errors": errors,
            "message": f"Created {len(created_ids)} concepts with {len(errors)} errors"
        }

    # Relationship Operations

    @retry_on_exception(max_retries=2)
    @log_execution_time
    def create_relationship(self, relationship: Relationship) -> Optional[str]:
        """Create a relationship between concepts in Neo4j.
        
        Args:
            relationship: Relationship to create
            
        Returns:
            ID of created relationship or None if creation failed
        """
        if not self.driver or not self._is_connected:
            if not self.connect():
                logger.error("Neo4j connection not available")
                return None
        
        try:
            with self._get_session() as session:
                result = session.execute_write(self._create_relationship_tx, relationship)
                return result
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Neo4j error creating relationship: {e}")
            logger.debug(traceback.format_exc())
            return None

    def _create_relationship_tx(self, tx: Transaction, relationship: Relationship) -> Optional[str]:
        """Transaction function for creating a relationship.
        
        Args:
            tx: Neo4j transaction
            relationship: Relationship to create
            
        Returns:
            ID of created relationship or None if creation failed
        """
        # Convert relationship to dictionary
        relationship_dict = relationship.dict() if hasattr(relationship, "dict") else relationship
        
        # Check if both concepts exist
        check_query = """
        MATCH (source:Concept {id: $source_id})
        MATCH (target:Concept {id: $target_id})
        RETURN count(source) + count(target) as count
        """
        check_result = tx.run(check_query, {
            'source_id': relationship_dict["source_id"],
            'target_id': relationship_dict["target_id"]
        }).single()
        
        if not check_result or check_result["count"] != 2:
            logger.error("One or both concepts do not exist")
            return None
        
        # Convert datetime objects to strings
        relationship_dict["created_at"] = self._format_datetime(relationship_dict.get("created_at"))
        
        # Ensure relationship_type is stored as string
        if "relationship_type" in relationship_dict and hasattr(relationship_dict["relationship_type"], "value"):
            relationship_dict["relationship_type"] = relationship_dict["relationship_type"].value
        
        query = """
        MATCH (source:Concept {id: $source_id})
        MATCH (target:Concept {id: $target_id})
        CREATE (source)-[r:RELATES_TO {
            id: $id,
            relationship_type: $relationship_type,
            strength: $strength,
            description: $description,
            bidirectional: $bidirectional,
            created_at: $created_at
        }]->(target)
        RETURN r.id as id
        """
        
        result = tx.run(query, relationship_dict).single()
        
        # Create reverse relationship if bidirectional
        if relationship_dict.get("bidirectional", False):
            reverse_id = str(uuid.uuid4())
            reverse_query = """
            MATCH (source:Concept {id: $target_id})
            MATCH (target:Concept {id: $source_id})
            CREATE (source)-[r:RELATES_TO {
                id: $reverse_id,
                relationship_type: $relationship_type,
                strength: $strength,
                description: $description,
                bidirectional: $bidirectional,
                created_at: $created_at,
                reverse_of: $id
            }]->(target)
            """
            tx.run(reverse_query, {
                **relationship_dict,
                "reverse_id": reverse_id
            })
        
        return result["id"] if result else None

    @retry_on_exception(max_retries=2)
    @log_execution_time
    def update_relationship(self, relationship_id: str, updates: Dict[str, Any]) -> bool:
        """Update a relationship in Neo4j.
        
        Args:
            relationship_id: ID of the relationship to update
            updates: Dictionary of updates to apply
            
        Returns:
            True if update was successful, False otherwise
        """
        if not self.driver or not self._is_connected:
            if not self.connect():
                logger.error("Neo4j connection not available")
                return False
                
        if not relationship_id:
            logger.error("Cannot update relationship: relationship_id is None or empty")
            return False
        
        try:
            with self._get_session() as session:
                result = session.execute_write(self._update_relationship_tx, relationship_id, updates)
                return result
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Neo4j error updating relationship {relationship_id}: {e}")
            logger.debug(traceback.format_exc())
            return False

    def _update_relationship_tx(self, tx: Transaction, relationship_id: str, updates: Dict[str, Any]) -> bool:
        """Transaction function for updating a relationship.
        
        Args:
            tx: Neo4j transaction
            relationship_id: ID of the relationship to update
            updates: Dictionary of updates to apply
            
        Returns:
            True if update was successful, False otherwise
        """
        # Format updated_at if present
        if "updated_at" in updates:
            updates["updated_at"] = self._format_datetime(updates["updated_at"])
        
        # Ensure enum values are converted to strings
        for key, value in updates.items():
            if hasattr(value, "value"):
                updates[key] = value.value
        
        # Generate dynamic update clauses
        update_clauses = []
        for key, value in updates.items():
            update_clauses.append(f"r.{key} = ${key}")
        
        if update_clauses:
            query = f"""
            MATCH ()-[r:RELATES_TO {{id: $relationship_id}}]->()
            SET {', '.join(update_clauses)}
            RETURN count(r) as count
            """
            params = {"relationship_id": relationship_id, **updates}
            result = tx.run(query, params).single()
        else:
            result = {"count": 1}  # Default to success if no updates to apply
        
        # Update bidirectional relationship if needed
        if "bidirectional" in updates:
            if updates["bidirectional"]:
                # Check if reverse relationship exists
                check_query = """
                MATCH (a)-[r:RELATES_TO {id: $relationship_id}]->(b)
                MATCH (b)-[reverse:RELATES_TO]->(a)
                WHERE reverse.reverse_of = $relationship_id
                RETURN count(reverse) as count
                """
                check_result = tx.run(check_query, {"relationship_id": relationship_id}).single()
                
                if check_result and check_result["count"] == 0:
                    # Create reverse relationship
                    create_reverse_query = """
                    MATCH (a)-[r:RELATES_TO {id: $relationship_id}]->(b)
                    CREATE (b)-[reverse:RELATES_TO {
                        id: $reverse_id,
                        relationship_type: r.relationship_type,
                        strength: r.strength,
                        description: r.description,
                        bidirectional: true,
                        created_at: $created_at,
                        reverse_of: $relationship_id
                    }]->(a)
                    """
                    tx.run(create_reverse_query, {
                        "relationship_id": relationship_id,
                        "reverse_id": str(uuid.uuid4()),
                        "created_at": self._format_datetime(datetime.now())
                    })
            else:
                # Remove reverse relationship if bidirectional is set to false
                delete_reverse_query = """
                MATCH ()-[r:RELATES_TO]->() 
                WHERE r.reverse_of = $relationship_id
                DELETE r
                """
                tx.run(delete_reverse_query, {"relationship_id": relationship_id})
        
        # Update related reverse relationship if it exists and bidirectional remains true
        if result and result["count"] > 0 and updates.get("bidirectional", True):
            update_reverse_query = f"""
            MATCH ()-[r:RELATES_TO]->() 
            WHERE r.reverse_of = $relationship_id
            SET {', '.join(['r.' + key + ' = $' + key for key in updates.keys() if key != 'bidirectional'])}
            """
            tx.run(update_reverse_query, {"relationship_id": relationship_id, **updates})
        
        return result and result["count"] > 0

    @retry_on_exception(max_retries=2)
    @log_execution_time
    def delete_relationship(self, relationship_id: str) -> bool:
        """Delete a relationship from Neo4j.
        
        Args:
            relationship_id: ID of the relationship to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if not self.driver or not self._is_connected:
            if not self.connect():
                logger.error("Neo4j connection not available")
                return False
                
        if not relationship_id:
            logger.error("Cannot delete relationship: relationship_id is None or empty")
            return False
        
        try:
            with self._get_session() as session:
                result = session.execute_write(self._delete_relationship_tx, relationship_id)
                return result
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Neo4j error deleting relationship {relationship_id}: {e}")
            logger.debug(traceback.format_exc())
            return False

    def _delete_relationship_tx(self, tx: Transaction, relationship_id: str) -> bool:
        """Transaction function for deleting a relationship.
        
        Args:
            tx: Neo4j transaction
            relationship_id: ID of the relationship to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        # Check if relationship is bidirectional
        check_query = """
        MATCH ()-[r:RELATES_TO {id: $relationship_id}]->()
        RETURN r.bidirectional as bidirectional
        """
        check_result = tx.run(check_query, {"relationship_id": relationship_id}).single()
        
        if check_result and check_result.get("bidirectional"):
            # Delete reverse relationship if bidirectional
            reverse_query = """
            MATCH ()-[r:RELATES_TO]->() 
            WHERE r.reverse_of = $relationship_id
            DELETE r
            """
            tx.run(reverse_query, {"relationship_id": relationship_id})
        
        # Delete main relationship
        query = """
        MATCH ()-[r:RELATES_TO {id: $relationship_id}]->()
        DELETE r
        RETURN count(r) as count
        """
        result = tx.run(query, {"relationship_id": relationship_id}).single()
        
        return result and result["count"] > 0

    @retry_on_exception(max_retries=1)
    @log_execution_time
    def get_relationship(self, relationship_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a relationship from Neo4j by ID.
        
        Args:
            relationship_id: ID of the relationship to retrieve
            
        Returns:
            Dictionary representing the relationship or None if not found
        """
        if not self.driver or not self._is_connected:
            if not self.connect():
                logger.error("Neo4j connection not available")
                return None
                
        if not relationship_id:
            logger.error("Cannot get relationship: relationship_id is None or empty")
            return None
        
        try:
            with self._get_session() as session:
                query = """
                MATCH (source)-[r:RELATES_TO {id: $relationship_id}]->(target)
                RETURN r, source.id as source_id, source.name as source_name, 
                       target.id as target_id, target.name as target_name
                """
                result = session.run(query, {"relationship_id": relationship_id}).single()
                
                if result:
                    rel_data = dict(result["r"])
                    
                    # Add source and target info
                    rel_data["source_id"] = result["source_id"]
                    rel_data["source_name"] = result["source_name"]
                    rel_data["target_id"] = result["target_id"]
                    rel_data["target_name"] = result["target_name"]
                    
                    # Convert string dates back to datetime objects
                    rel_data["created_at"] = self._parse_datetime(rel_data.get("created_at"))
                    rel_data["updated_at"] = self._parse_datetime(rel_data.get("updated_at"))
                    
                    return rel_data
                return None
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Neo4j error retrieving relationship {relationship_id}: {e}")
            logger.debug(traceback.format_exc())
            return None

    @retry_on_exception(max_retries=1)
    @log_execution_time
    def get_concept_relationships(self, concept_id: str, direction: str = "both", 
                                  relationship_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get relationships for a concept, optionally filtered by type and direction.
        
        Args:
            concept_id: ID of the concept
            direction: Relationship direction ("incoming", "outgoing", or "both")
            relationship_types: Optional list of relationship types to filter by
            
        Returns:
            List of relationship dictionaries
        """
        if not self.driver or not self._is_connected:
            if not self.connect():
                logger.error("Neo4j connection not available")
                return []
                
        if not concept_id:
            logger.error("Cannot get concept relationships: concept_id is None or empty")
            return []
        
        try:
            with self._get_session() as session:
                params = {"concept_id": concept_id}
                type_filter = ""
                
                # Handle relationship type filtering
                if relationship_types:
                    # Convert enum values to strings if needed
                    types_str = []
                    for rt in relationship_types:
                        if hasattr(rt, "value"):
                            types_str.append(rt.value)
                        else:
                            types_str.append(str(rt))
                    
                    type_filter = "AND r.relationship_type IN $relationship_types"
                    params["relationship_types"] = types_str
                
                # Build query based on direction
                if direction == "outgoing":
                    query = f"""
                    MATCH (c:Concept {{id: $concept_id}})-[r:RELATES_TO]->(related:Concept)
                    WHERE 1=1 {type_filter}
                    RETURN r, c.id as source_id, related.id as target_id, 
                           related.id as related_id, related.name as related_name, 
                           related.concept_type as related_type, 'outgoing' as direction
                    """
                elif direction == "incoming":
                    query = f"""
                    MATCH (c:Concept {{id: $concept_id}})<-[r:RELATES_TO]-(related:Concept)
                    WHERE 1=1 {type_filter}
                    RETURN r, related.id as source_id, c.id as target_id, 
                           related.id as related_id, related.name as related_name, 
                           related.concept_type as related_type, 'incoming' as direction
                    """
                else:  # both
                    query = f"""
                    MATCH (c:Concept {{id: $concept_id}})-[r:RELATES_TO]->(related:Concept)
                    WHERE 1=1 {type_filter}
                    RETURN r, c.id as source_id, related.id as target_id, 
                           related.id as related_id, related.name as related_name, 
                           related.concept_type as related_type, 'outgoing' as direction
                    UNION
                    MATCH (c:Concept {{id: $concept_id}})<-[r:RELATES_TO]-(related:Concept)
                    WHERE 1=1 {type_filter}
                    RETURN r, related.id as source_id, c.id as target_id, 
                           related.id as related_id, related.name as related_name, 
                           related.concept_type as related_type, 'incoming' as direction
                    """
                
                results = session.run(query, params)
                relationships = []
                
                for record in results:
                    rel_data = dict(record["r"])
                    
                    # Convert string dates to datetime objects
                    rel_data["created_at"] = self._parse_datetime(rel_data.get("created_at"))
                    
                    # Add source and target info along with related concept info
                    rel_data["source_id"] = record["source_id"]
                    rel_data["target_id"] = record["target_id"]
                    rel_data["related_id"] = record["related_id"]
                    rel_data["related_name"] = record["related_name"]
                    rel_data["related_type"] = record["related_type"]
                    rel_data["direction"] = record["direction"]
                    
                    relationships.append(rel_data)
                
                return relationships
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Neo4j error retrieving relationships for concept {concept_id}: {e}")
            logger.debug(traceback.format_exc())
            return []

    @retry_on_exception(max_retries=1)
    @log_execution_time
    def bulk_create_relationships(self, relationships: List[Relationship]) -> Dict[str, Any]:
        """Create multiple relationships in a single transaction.
        
        Args:
            relationships: List of relationships to create
            
        Returns:
            Dictionary with success status and created IDs
        """
        if not self.driver or not self._is_connected:
            if not self.connect():
                logger.error("Neo4j connection not available")
                return {"success": False, "created": [], "message": "No connection to Neo4j"}
                
        if not relationships:
            return {"success": True, "created": [], "message": "No relationships to create"}
        
        try:
            with self._get_session() as session:
                result = session.execute_write(self._bulk_create_relationships_tx, relationships)
                return result
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Neo4j error bulk creating relationships: {e}")
            logger.debug(traceback.format_exc())
            return {"success": False, "created": [], "message": str(e)}

    def _bulk_create_relationships_tx(self, tx: Transaction, relationships: List[Relationship]) -> Dict[str, Any]:
        """Transaction function for bulk creating relationships.
        
        Args:
            tx: Neo4j transaction
            relationships: List of relationships to create
            
        Returns:
            Dictionary with success status and created IDs
        """
        created_ids = []
        errors = []
        
        # First, verify all concepts exist
        all_concept_ids = set()
        for rel in relationships:
            rel_dict = rel.dict() if hasattr(rel, "dict") else rel
            all_concept_ids.add(rel_dict["source_id"])
            all_concept_ids.add(rel_dict["target_id"])
        
        concept_ids_list = list(all_concept_ids)
        concept_exists_query = """
        MATCH (c:Concept)
        WHERE c.id IN $concept_ids
        RETURN c.id as id
        """
        
        concept_result = tx.run(concept_exists_query, {"concept_ids": concept_ids_list})
        existing_concepts = {record["id"] for record in concept_result}
        
        missing_concepts = all_concept_ids - existing_concepts
        if missing_concepts:
            logger.warning(f"Missing concepts for bulk relationship creation: {missing_concepts}")
        
        # Now create the relationships
        for relationship in relationships:
            try:
                # Convert relationship to dictionary
                rel_dict = relationship.dict() if hasattr(relationship, "dict") else relationship
                
                # Skip if source or target doesn't exist
                if rel_dict["source_id"] not in existing_concepts or rel_dict["target_id"] not in existing_concepts:
                    errors.append({
                        "id": rel_dict.get("id"),
                        "error": f"Source or target concept does not exist: {rel_dict['source_id']} -> {rel_dict['target_id']}"
                    })
                    continue
                
                # Convert datetime objects to strings
                rel_dict["created_at"] = self._format_datetime(rel_dict.get("created_at"))
                
                # Ensure relationship_type is stored as string
                if "relationship_type" in rel_dict and hasattr(rel_dict["relationship_type"], "value"):
                    rel_dict["relationship_type"] = rel_dict["relationship_type"].value
                
                query = """
                MATCH (source:Concept {id: $source_id})
                MATCH (target:Concept {id: $target_id})
                CREATE (source)-[r:RELATES_TO {
                    id: $id,
                    relationship_type: $relationship_type,
                    strength: $strength,
                    description: $description,
                    bidirectional: $bidirectional,
                    created_at: $created_at
                }]->(target)
                RETURN r.id as id
                """
                
                result = tx.run(query, rel_dict).single()
                
                # Create reverse relationship if bidirectional
                if rel_dict.get("bidirectional", False):
                    reverse_id = str(uuid.uuid4())
                    reverse_query = """
                    MATCH (source:Concept {id: $target_id})
                    MATCH (target:Concept {id: $source_id})
                    CREATE (source)-[r:RELATES_TO {
                        id: $reverse_id,
                        relationship_type: $relationship_type,
                        strength: $strength,
                        description: $description,
                        bidirectional: $bidirectional,
                        created_at: $created_at,
                        reverse_of: $id
                    }]->(target)
                    """
                    tx.run(reverse_query, {
                        **rel_dict,
                        "reverse_id": reverse_id
                    })
                
                created_ids.append(rel_dict["id"])
                
            except Exception as e:
                errors.append({"id": rel_dict.get("id"), "error": str(e)})
                logger.error(f"Error creating relationship in bulk operation: {e}")
        
        success = len(errors) == 0
        return {
            "success": success, 
            "created": created_ids,
            "errors": errors,
            "message": f"Created {len(created_ids)} relationships with {len(errors)} errors"
        }

    @retry_on_exception(max_retries=2)
    @log_execution_time
    def get_concept_graph(self, concept_id: str, depth: int = 1, 
                           relationship_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get a subgraph centered on a concept with specified depth.
        
        Args:
            concept_id: ID of the center concept
            depth: Maximum path length from center
            relationship_types: Optional list of relationship types to filter by
            
        Returns:
            Dictionary with nodes and relationships
        """
        if not self.driver or not self._is_connected:
            if not self.connect():
                logger.error("Neo4j connection not available")
                return {"nodes": [], "edges": []}
                
        if not concept_id:
            logger.error("Cannot get concept graph: concept_id is None or empty")
            return {"nodes": [], "edges": []}
        
        try:
            with self._get_session() as session:
                if relationship_types:
                    # Convert enum values to strings if needed
                    types_str = []
                    for rt in relationship_types:
                        if hasattr(rt, "value"):
                            types_str.append(rt.value)
                        else:
                            types_str.append(str(rt))
                    
                    query = f"""
                    MATCH path = (c:Concept {{id: $concept_id}})-[r:RELATES_TO*0..{depth}]-(related)
                    WHERE ALL(rel IN relationships(path) WHERE rel.relationship_type IN $relationship_types)
                    RETURN path
                    LIMIT 1000
                    """
                    params = {"concept_id": concept_id, "relationship_types": types_str}
                else:
                    query = f"""
                    MATCH path = (c:Concept {{id: $concept_id}})-[r:RELATES_TO*0..{depth}]-(related)
                    RETURN path
                    LIMIT 1000
                    """
                    params = {"concept_id": concept_id}
                
                results = session.run(query, params)
                
                nodes = {}
                relationships = {}
                
                for record in results:
                    path = record["path"]
                    
                    # Process nodes
                    for node in path.nodes:
                        node_id = node.get("id")
                        if node_id not in nodes:
                            node_data = dict(node)
                            node_data["created_at"] = self._parse_datetime(node_data.get("created_at"))
                            node_data["updated_at"] = self._parse_datetime(node_data.get("updated_at"))
                            nodes[node_id] = node_data
                    
                    # Process relationships
                    for rel in path.relationships:
                        rel_id = rel.get("id")
                        if rel_id not in relationships:
                            rel_data = dict(rel)
                            rel_data["source_id"] = rel.start_node.get("id")
                            rel_data["target_id"] = rel.end_node.get("id")
                            rel_data["created_at"] = self._parse_datetime(rel_data.get("created_at"))
                            relationships[rel_id] = rel_data
                
                result_data = {
                    "nodes": list(nodes.values()), 
                    "edges": list(relationships.values())
                }
                
                # Provide 'relationships' as an alias for 'edges' for backward compatibility
                result_data["relationships"] = result_data["edges"]
                
                return result_data
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Neo4j error retrieving concept graph for {concept_id}: {e}")
            logger.debug(traceback.format_exc())
            return {"nodes": [], "edges": [], "relationships": []}

    @retry_on_exception(max_retries=1)
    @log_execution_time
    def get_learning_path(self, concept_ids: List[str], goal: str = "") -> List[Dict[str, Any]]:
        """Generate a learning path through specified concepts based on relationships.
        
        Args:
            concept_ids: List of concept IDs to include in path
            goal: Optional learning goal description
            
        Returns:
            List of ordered steps in the learning path
        """
        if not self.driver or not self._is_connected:
            if not self.connect():
                logger.error("Neo4j connection not available")
                return []
                
        if not concept_ids:
            logger.error("Cannot generate learning path: No concept IDs provided")
            return []
        
        try:
            with self._get_session() as session:
                # Verify concepts exist
                check_query = """
                MATCH (c:Concept)
                WHERE c.id IN $concept_ids
                RETURN c.id as id
                """
                existing_results = session.run(check_query, {"concept_ids": concept_ids})
                existing_ids = [record["id"] for record in existing_results]
                
                if len(existing_ids) < len(concept_ids):
                    missing = set(concept_ids) - set(existing_ids)
                    logger.warning(f"Some concepts not found for learning path: {missing}")
                    concept_ids = existing_ids
                
                if not concept_ids:
                    return []
                
                # Get concept details and prerequisites
                query = """
                MATCH (c:Concept)
                WHERE c.id IN $concept_ids
                OPTIONAL MATCH (c)-[r:RELATES_TO]->(prereq:Concept)
                WHERE prereq.id IN $concept_ids AND r.relationship_type = 'prerequisite'
                WITH c, collect(prereq.id) as prerequisites
                RETURN c.id as id, c.name as name, c.difficulty as difficulty, 
                       c.estimated_learning_time_minutes as time, prerequisites
                """
                results = session.run(query, {"concept_ids": concept_ids})
                
                # Build dependency graph
                concepts = {}
                dependencies = {}
                
                for record in results:
                    concept_id = record["id"]
                    concepts[concept_id] = {
                        "id": concept_id,
                        "name": record["name"],
                        "difficulty": record["difficulty"],
                        "time": record["time"] or 30,  # Default 30 minutes if not specified
                        "prerequisites": record["prerequisites"]
                    }
                    dependencies[concept_id] = set(record["prerequisites"])
                
                # Topological sort for ordering
                visited = set()
                temp_mark = set()
                order = []
                
                def visit(node_id):
                    if node_id in temp_mark:
                        # Cycle detected; skip further traversal to break cycle
                        logger.warning(f"Cycle detected in prerequisites involving {node_id}")
                        return
                    if node_id not in visited:
                        temp_mark.add(node_id)
                        for dependency in dependencies.get(node_id, set()):
                            visit(dependency)
                        temp_mark.remove(node_id)
                        visited.add(node_id)
                        order.append(node_id)
                
                for cid in concept_ids:
                    if cid not in visited:
                        visit(cid)
                
                # Create path from order (reverse to get correct dependency order)
                path = []
                current_order = 1
                for cid in reversed(order):
                    if cid in concept_ids:
                        concept = concepts[cid]
                        path.append({
                            "concept_id": cid,
                            "name": concept["name"],
                            "order": current_order,
                            "estimated_time_minutes": concept["time"],
                            "difficulty": concept["difficulty"],
                            "prerequisites": concept["prerequisites"]
                        })
                        current_order += 1
                
                logger.info(f"Generated learning path with {len(path)} steps")
                return path
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Neo4j error generating learning path: {e}")
            logger.debug(traceback.format_exc())
            return []

    @retry_on_exception(max_retries=1)
    @log_execution_time
    def validate_knowledge_graph(self, domain_id: Optional[str] = None) -> ValidationResult:
        """Validate the knowledge graph for consistency issues.
        
        Args:
            domain_id: Optional domain ID to validate
            
        Returns:
            ValidationResult with issues and warnings
        """
        if not self.driver or not self._is_connected:
            if not self.connect():
                logger.error("Neo4j connection not available")
                return ValidationResult(
                    valid=False,
                    issues=[ValidationIssue(
                        issue_type="connection",
                        severity="critical",
                        concepts_involved=[],
                        description="Neo4j connection not available",
                        recommendation="Check Neo4j connection settings"
                    )],
                    timestamp=datetime.now()
                )
        
        issues = []
        warnings = []
        params = {}
        domain_clause = ""
        
        if domain_id:
            domain_clause = "AND (c.parent_id = $domain_id OR c.id = $domain_id)"
            params["domain_id"] = domain_id
            
            # First verify the domain exists
            if not self.get_concept(domain_id):
                return ValidationResult(
                    valid=False,
                    issues=[ValidationIssue(
                        issue_type="missing_domain",
                        severity="critical",
                        concepts_involved=[domain_id],
                        description=f"Domain concept with ID {domain_id} not found",
                        recommendation="Verify the domain ID is correct"
                    )],
                    timestamp=datetime.now()
                )
        
        try:
            with self._get_session() as session:
                # Check for circular prerequisites
                circular_query = f"""
                MATCH path = (c:Concept)-[r:RELATES_TO*2..5]->(c)
                WHERE ALL(rel in relationships(path) WHERE rel.relationship_type = 'prerequisite')
                {domain_clause}
                RETURN path, length(path) as cycle_length
                LIMIT 10
                """
                circular_results = session.run(circular_query, params)
                for record in circular_results:
                    path = record["path"]
                    concepts_in_cycle = [node.get("id") for node in path.nodes]
                    
                    # Get concept names for better messages
                    names_query = """
                    MATCH (c:Concept)
                    WHERE c.id IN $concept_ids
                    RETURN c.id as id, c.name as name
                    """
                    names_result = session.run(names_query, {"concept_ids": concepts_in_cycle})
                    name_map = {r["id"]: r["name"] for r in names_result}
                    
                    # Create issue with concept names
                    cycle_description = " → ".join([name_map.get(cid, cid) for cid in concepts_in_cycle])
                    
                    issues.append(ValidationIssue(
                        issue_type="circular_prerequisite",
                        severity="high",
                        concepts_involved=concepts_in_cycle,
                        description=f"Circular prerequisite chain detected: {cycle_description}",
                        recommendation="Review and remove one of the prerequisite relationships to break the cycle"
                    ))
                
                # Check for orphaned concepts (no relationships)
                orphans_query = f"""
                MATCH (c:Concept)
                WHERE NOT (c)-[:RELATES_TO]-() AND NOT ()-[:RELATES_TO]->(c)
                {domain_clause}
                RETURN c.id as id, c.name as name
                LIMIT 100
                """
                orphans_results = session.run(orphans_query, params)
                for record in orphans_results:
                    warnings.append(ValidationIssue(
                        issue_type="orphaned_concept",
                        severity="medium",
                        concepts_involved=[record["id"]],
                        description=f"Concept '{record['name']}' has no relationships",
                        recommendation="Connect this concept to related concepts or consider removing it"
                    ))
                
                # Check for contradictory prerequisite relationships
                contradictory_query = f"""
                MATCH (a:Concept)-[r1:RELATES_TO]->(b:Concept)-[r2:RELATES_TO]->(a:Concept)
                WHERE r1.relationship_type = 'prerequisite' AND r2.relationship_type = 'prerequisite'
                {domain_clause}
                RETURN a.id as a_id, a.name as a_name, b.id as b_id, b.name as b_name
                LIMIT 50
                """
                contradictory_results = session.run(contradictory_query, params)
                for record in contradictory_results:
                    issues.append(ValidationIssue(
                        issue_type="contradictory_prerequisites",
                        severity="high",
                        concepts_involved=[record["a_id"], record["b_id"]],
                        description=f"Contradictory prerequisites between '{record['a_name']}' and '{record['b_name']}'",
                        recommendation="Review and resolve by keeping only one prerequisite direction"
                    ))
                
                # Check for inconsistent difficulty levels
                difficulty_query = f"""
                MATCH (a:Concept)-[r:RELATES_TO]->(b:Concept)
                WHERE r.relationship_type = 'prerequisite' AND (
                    (a.difficulty = 'advanced' AND b.difficulty = 'beginner') OR
                    (a.difficulty = 'expert' AND (b.difficulty = 'beginner' OR b.difficulty = 'intermediate')) OR
                    (a.difficulty = 'advanced' AND b.difficulty = 'intermediate' AND b.importance > 0.7)
                )
                {domain_clause}
                RETURN a.id as a_id, a.name as a_name, a.difficulty as a_diff, 
                       b.id as b_id, b.name as b_name, b.difficulty as b_diff
                LIMIT 50
                """
                difficulty_results = session.run(difficulty_query, params)
                for record in difficulty_results:
                    warnings.append(ValidationIssue(
                        issue_type="inconsistent_difficulty",
                        severity="low",
                        concepts_involved=[record["a_id"], record["b_id"]],
                        description=f"Prerequisite '{record['a_name']}' ({record['a_diff']}) has higher difficulty than dependent '{record['b_name']}' ({record['b_diff']})",
                        recommendation="Review difficulty levels or prerequisite relationship"
                    ))
                
                # Additional check: concepts without difficulty or description
                incomplete_query = f"""
                MATCH (c:Concept)
                WHERE (c.difficulty IS NULL OR c.description IS NULL OR c.description = '')
                {domain_clause}
                RETURN c.id as id, c.name as name, 
                       c.difficulty as difficulty, 
                       c.description as description
                LIMIT 100
                """
                incomplete_results = session.run(incomplete_query, params)
                for record in incomplete_results:
                    missing = []
                    if not record["difficulty"]:
                        missing.append("difficulty")
                    if not record["description"]:
                        missing.append("description")
                        
                    if missing:
                        warnings.append(ValidationIssue(
                            issue_type="incomplete_concept",
                            severity="low",
                            concepts_involved=[record["id"]],
                            description=f"Concept '{record['name']}' is missing: {', '.join(missing)}",
                            recommendation="Complete the concept information"
                        ))
                
                # Determine if valid based on critical issues
                valid = len(issues) == 0
                
                # Count concepts and relationships in the domain for stats
                stats = {}
                if domain_id:
                    count_query = f"""
                    MATCH (c:Concept)
                    WHERE c.id = $domain_id OR c.parent_id = $domain_id
                    RETURN count(c) as concept_count
                    """
                    concept_count = session.run(count_query, {"domain_id": domain_id}).single()["concept_count"]
                    
                    rel_query = f"""
                    MATCH (a:Concept)-[r:RELATES_TO]->(b:Concept)
                    WHERE (a.id = $domain_id OR a.parent_id = $domain_id) AND
                          (b.id = $domain_id OR b.parent_id = $domain_id)
                    RETURN count(r) as rel_count
                    """
                    rel_count = session.run(rel_query, {"domain_id": domain_id}).single()["rel_count"]
                    
                    stats = {
                        "concepts": concept_count,
                        "relationships": rel_count,
                        "issues": len(issues),
                        "warnings": len(warnings)
                    }
                
                return ValidationResult(
                    valid=valid,
                    issues=issues,
                    warnings=warnings,
                    stats=stats,
                    timestamp=datetime.now()
                )
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Neo4j error validating knowledge graph: {e}")
            logger.debug(traceback.format_exc())
            return ValidationResult(
                valid=False,
                issues=[ValidationIssue(
                    issue_type="validation_error",
                    severity="critical",
                    concepts_involved=[],
                    description=f"Error during validation: {str(e)}",
                    recommendation="Check Neo4j connection and query syntax"
                )],
                timestamp=datetime.now()
            )
    @retry_on_exception(max_retries=1)
    @log_execution_time
    def get_domain_structure(self, domain_id: str, include_relationships: bool = True) -> Dict[str, Any]:
        """Get the structure of a domain including its concepts and their hierarchy.
        
        Args:
            domain_id: ID of the domain concept
            include_relationships: Whether to include relationships
            
        Returns:
            Dictionary with domain, concepts, and relationships
        """
        if not self.driver or not self._is_connected:
            if not self.connect():
                logger.error("Neo4j connection not available")
                return {"domain": None, "concepts": [], "relationships": []}
                
        if not domain_id:
            logger.error("Cannot get domain structure: domain_id is None or empty")
            return {"domain": None, "concepts": [], "relationships": []}
        
        try:
            with self._get_session() as session:
                # Get domain concept
                domain_query = """
                MATCH (d:Concept {id: $domain_id})
                RETURN d
                """
                domain_result = session.run(domain_query, {"domain_id": domain_id}).single()
                
                if not domain_result:
                    logger.error(f"Domain concept not found: {domain_id}")
                    return {"domain": None, "concepts": [], "relationships": []}
                
                domain = dict(domain_result["d"])
                domain["created_at"] = self._parse_datetime(domain.get("created_at"))
                domain["updated_at"] = self._parse_datetime(domain.get("updated_at"))
                
                # Get all concepts in the domain
                concepts_query = """
                MATCH (c:Concept)
                WHERE c.id = $domain_id OR c.parent_id = $domain_id OR 
                      EXISTS((c)-[:RELATES_TO {relationship_type: 'part_of'}]->(:Concept {id: $domain_id}))
                RETURN c
                ORDER BY c.importance DESC
                """
                concepts_result = session.run(concepts_query, {"domain_id": domain_id})
                concepts = []
                concept_ids = []
                
                for record in concepts_result:
                    c = dict(record["c"])
                    c["created_at"] = self._parse_datetime(c.get("created_at"))
                    c["updated_at"] = self._parse_datetime(c.get("updated_at"))
                    concepts.append(c)
                    concept_ids.append(c.get("id"))
                
                relationships = []
                if include_relationships and concept_ids:
                    # Fetch relationships between domain concepts
                    relationships_query = """
                    MATCH (source:Concept)-[r:RELATES_TO]->(target:Concept)
                    WHERE source.id IN $concept_ids AND target.id IN $concept_ids
                    RETURN r, source.id as source_id, target.id as target_id,
                           source.name as source_name, target.name as target_name
                    """
                    rel_results = session.run(relationships_query, {"concept_ids": concept_ids})
                    
                    for record in rel_results:
                        rel = dict(record["r"])
                        rel["source_id"] = record["source_id"] 
                        rel["target_id"] = record["target_id"]
                        rel["source_name"] = record["source_name"]
                        rel["target_name"] = record["target_name"]
                        rel["created_at"] = self._parse_datetime(rel.get("created_at"))
                        relationships.append(rel)
                
                logger.info(f"Retrieved domain structure with {len(concepts)} concepts and {len(relationships)} relationships")
                
                return {
                    "domain": domain,
                    "concepts": concepts,
                    "relationships": relationships
                }
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Neo4j error getting domain structure for {domain_id}: {e}")
            logger.debug(traceback.format_exc())
            return {"domain": None, "concepts": [], "relationships": []}

    @retry_on_exception(max_retries=1)
    @log_execution_time
    def search_concepts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for concepts by text query.
        
        Args:
            query: Text to search for
            limit: Maximum number of results to return
            
        Returns:
            List of matching concept dictionaries
        """
        if not self.driver or not self._is_connected:
            if not self.connect():
                logger.error("Neo4j connection not available")
                return []
                
        if not query:
            logger.error("Cannot search concepts: query is None or empty")
            return []
        
        try:
            with self._get_session() as session:
                # Create case-insensitive pattern
                search_pattern = f"(?i).*{query}.*"
                
                # Search by name, description, and keywords
                search_query = """
                MATCH (c:Concept)
                WHERE c.name =~ $pattern OR c.description =~ $pattern OR 
                      ANY(keyword IN c.keywords WHERE keyword =~ $pattern)
                RETURN c
                ORDER BY 
                    CASE WHEN c.name =~ $pattern THEN 10 ELSE 0 END +
                    CASE WHEN c.description =~ $pattern THEN 5 ELSE 0 END +
                    CASE WHEN ANY(keyword IN c.keywords WHERE keyword =~ $pattern) THEN 3 ELSE 0 END +
                    c.importance DESC
                LIMIT $limit
                """
                
                results = session.run(search_query, {"pattern": search_pattern, "limit": limit})
                
                concepts = []
                for record in results:
                    c = dict(record["c"])
                    c["created_at"] = self._parse_datetime(c.get("created_at"))
                    c["updated_at"] = self._parse_datetime(c.get("updated_at"))
                    concepts.append(c)
                
                logger.info(f"Found {len(concepts)} concepts matching query: '{query}'")
                return concepts
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Neo4j error searching concepts: {e}")
            logger.debug(traceback.format_exc())
            return []

    @retry_on_exception(max_retries=1)
    @log_execution_time
    def find_path_between_concepts(self, source_id: str, target_id: str, 
                                   max_depth: int = 5) -> Dict[str, Any]:
        """Find the shortest path between two concepts.
        
        Args:
            source_id: ID of the source concept
            target_id: ID of the target concept
            max_depth: Maximum path length to search
            
        Returns:
            Dictionary with path information
        """
        if not self.driver or not self._is_connected:
            if not self.connect():
                logger.error("Neo4j connection not available")
                return {"path_exists": False, "path": [], "length": 0}
                
        if not source_id or not target_id:
            logger.error("Cannot find path: source_id or target_id is None or empty")
            return {"path_exists": False, "path": [], "length": 0}
        
        try:
            with self._get_session() as session:
                # Find shortest path
                path_query = f"""
                MATCH path = shortestPath((a:Concept {{id: $source_id}})-[r:RELATES_TO*1..{max_depth}]-(b:Concept {{id: $target_id}}))
                RETURN path
                """
                
                result = session.run(path_query, {"source_id": source_id, "target_id": target_id}).single()
                
                if not result:
                    logger.info(f"No path found between concepts {source_id} and {target_id}")
                    return {"path_exists": False, "path": [], "length": 0}
                
                path = result["path"]
                
                # Process nodes and relationships on the path
                nodes = []
                relationships = []
                
                for node in path.nodes:
                    node_data = dict(node)
                    node_data["created_at"] = self._parse_datetime(node_data.get("created_at"))
                    node_data["updated_at"] = self._parse_datetime(node_data.get("updated_at"))
                    nodes.append(node_data)
                
                for rel in path.relationships:
                    rel_data = dict(rel)
                    rel_data["source_id"] = rel.start_node.get("id")
                    rel_data["target_id"] = rel.end_node.get("id")
                    rel_data["created_at"] = self._parse_datetime(rel_data.get("created_at"))
                    relationships.append(rel_data)
                
                logger.info(f"Found path of length {len(relationships)} between concepts {source_id} and {target_id}")
                
                return {
                    "path_exists": True,
                    "path": {
                        "nodes": nodes,
                        "relationships": relationships
                    },
                    "length": len(relationships)
                }
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Neo4j error finding path between concepts: {e}")
            logger.debug(traceback.format_exc())
            return {"path_exists": False, "path": [], "length": 0}

    @retry_on_exception(max_retries=1)
    @log_execution_time
    def get_concept_hierarchies(self, domain_id: str) -> Dict[str, Any]:
        """Get hierarchical structures within a domain.
        
        Args:
            domain_id: ID of the domain concept
            
        Returns:
            Dictionary with hierarchical structures
        """
        if not self.driver or not self._is_connected:
            if not self.connect():
                logger.error("Neo4j connection not available")
                return {"hierarchies": []}
                
        if not domain_id:
            logger.error("Cannot get concept hierarchies: domain_id is None or empty")
            return {"hierarchies": []}
        
        try:
            with self._get_session() as session:
                # Find hierarchy roots (direct children of domain that have children)
                roots_query = """
                MATCH (domain:Concept {id: $domain_id})<-[:RELATES_TO {relationship_type: 'part_of'}]-(root:Concept)
                WHERE EXISTS((root)<-[:RELATES_TO {relationship_type: 'part_of'}]-(:Concept))
                RETURN root.id AS id, root.name AS name
                ORDER BY root.importance DESC
                """
                
                roots_result = session.run(roots_query, {"domain_id": domain_id})
                hierarchies = []
                
                for root in roots_result:
                    root_id = root["id"]
                    root_name = root["name"]
                    
                    # Get full hierarchy for this root
                    hierarchy_query = """
                    MATCH path = (root:Concept {id: $root_id})<-[:RELATES_TO*0..5 {relationship_type: 'part_of'}]-(descendant:Concept)
                    RETURN path
                    """
                    
                    hierarchy_result = session.run(hierarchy_query, {"root_id": root_id})
                    
                    hierarchy_nodes = {}
                    hierarchy_relationships = {}
                    
                    for record in hierarchy_result:
                        path = record["path"]
                        
                        # Process nodes
                        for node in path.nodes:
                            node_id = node.get("id")
                            if node_id not in hierarchy_nodes:
                                node_data = dict(node)
                                node_data["created_at"] = self._parse_datetime(node_data.get("created_at"))
                                node_data["updated_at"] = self._parse_datetime(node_data.get("updated_at"))
                                hierarchy_nodes[node_id] = node_data
                        
                        # Process relationships
                        for rel in path.relationships:
                            rel_id = rel.get("id")
                            if rel_id not in hierarchy_relationships:
                                rel_data = dict(rel)
                                rel_data["source_id"] = rel.start_node.get("id")
                                rel_data["target_id"] = rel.end_node.get("id")
                                rel_data["created_at"] = self._parse_datetime(rel_data.get("created_at"))
                                hierarchy_relationships[rel_id] = rel_data
                    
                    hierarchies.append({
                        "root_id": root_id,
                        "root_name": root_name,
                        "nodes": list(hierarchy_nodes.values()),
                        "relationships": list(hierarchy_relationships.values())
                    })
                
                logger.info(f"Found {len(hierarchies)} hierarchies in domain {domain_id}")
                return {"hierarchies": hierarchies}
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Neo4j error getting concept hierarchies: {e}")
            logger.debug(traceback.format_exc())
            return {"hierarchies": []}

    @retry_on_exception(max_retries=1)
    @log_execution_time
    def delete_entity(self, entity_id: str) -> bool:
        """Delete any entity (concept or relationship) by ID.
        
        Args:
            entity_id: ID of the entity to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if not self.driver or not self._is_connected:
            if not self.connect():
                logger.error("Neo4j connection not available")
                return False
                
        if not entity_id:
            logger.error("Cannot delete entity: entity_id is None or empty")
            return False
        
        try:
            with self._get_session() as session:
                # Try to delete as concept first
                try:
                    success = self.delete_concept(entity_id)
                    if success:
                        return True
                except Exception as e:
                    logger.debug(f"Entity {entity_id} not a concept or error deleting: {e}")
                
                # Try to delete as relationship
                try:
                    success = self.delete_relationship(entity_id)
                    if success:
                        return True
                except Exception as e:
                    logger.debug(f"Entity {entity_id} not a relationship or error deleting: {e}")
                
                # If we get here, the entity wasn't found or couldn't be deleted
                logger.warning(f"Entity {entity_id} not found or couldn't be deleted")
                return False
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Neo4j error deleting entity {entity_id}: {e}")
            logger.debug(traceback.format_exc())
            return False

    @retry_on_exception(max_retries=1)
    @log_execution_time
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph.
        
        Returns:
            Dictionary with graph statistics
        """
        if not self.driver or not self._is_connected:
            if not self.connect():
                logger.error("Neo4j connection not available")
                return {"status": "error", "message": "Neo4j connection not available"}
        
        try:
            with self._get_session() as session:
                stats = {}
                
                # Count concepts by type
                concept_types_query = """
                MATCH (c:Concept)
                RETURN c.concept_type as type, count(c) as count
                ORDER BY count DESC
                """
                
                concept_types_result = session.run(concept_types_query)
                concept_types = {record["type"]: record["count"] for record in concept_types_result}
                stats["concept_types"] = concept_types
                
                # Count relationships by type
                rel_types_query = """
                MATCH ()-[r:RELATES_TO]->()
                RETURN r.relationship_type as type, count(r) as count
                ORDER BY count DESC
                """
                
                rel_types_result = session.run(rel_types_query)
                rel_types = {record["type"]: record["count"] for record in rel_types_result}
                stats["relationship_types"] = rel_types
                
                # Count concepts by difficulty
                difficulty_query = """
                MATCH (c:Concept)
                RETURN c.difficulty as difficulty, count(c) as count
                ORDER BY count DESC
                """
                
                difficulty_result = session.run(difficulty_query)
                difficulties = {record["difficulty"]: record["count"] for record in difficulty_result}
                stats["difficulties"] = difficulties
                
                # Top level domains
                domains_query = """
                MATCH (d:Concept {concept_type: 'domain'})
                OPTIONAL MATCH (d)<-[:RELATES_TO {relationship_type: 'part_of'}]-(c:Concept)
                RETURN d.id as id, d.name as name, count(c) as children
                ORDER BY children DESC
                LIMIT 10
                """
                
                domains_result = session.run(domains_query)
                domains = [{"id": record["id"], "name": record["name"], "children": record["children"]} 
                           for record in domains_result]
                stats["top_domains"] = domains
                
                # Graph density statistics
                density_query = """
                MATCH (c:Concept)
                OPTIONAL MATCH (c)-[r:RELATES_TO]-()
                RETURN count(DISTINCT c) as concepts, count(DISTINCT r) as relationships
                """
                
                density_result = session.run(density_query).single()
                concepts_count = density_result["concepts"]
                relationships_count = density_result["relationships"]
                
                # Calculate theoretical max relationships (n*(n-1)/2 for undirected)
                max_relationships = concepts_count * (concepts_count - 1)
                density = relationships_count / max_relationships if max_relationships > 0 else 0
                
                stats["graph_metrics"] = {
                    "concepts": concepts_count,
                    "relationships": relationships_count,
                    "density": density,
                    "avg_relationships_per_concept": relationships_count / concepts_count if concepts_count > 0 else 0
                }
                
                # Recently added concepts
                recent_query = """
                MATCH (c:Concept)
                WHERE c.created_at IS NOT NULL
                RETURN c.id as id, c.name as name, c.concept_type as type, c.created_at as created
                ORDER BY c.created_at DESC
                LIMIT 10
                """
                
                recent_result = session.run(recent_query)
                recent_concepts = []
                
                for record in recent_result:
                    created_at = self._parse_datetime(record["created"])
                    if created_at:
                        recent_concepts.append({
                            "id": record["id"],
                            "name": record["name"],
                            "type": record["type"],
                            "created_at": created_at.isoformat() if created_at else None
                        })
                
                stats["recent_concepts"] = recent_concepts
                
                return {
                    "status": "success",
                    "stats": stats,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Neo4j error getting statistics: {e}")
            logger.debug(traceback.format_exc())
            return {"status": "error", "message": str(e)}

    @retry_on_exception(max_retries=1)
    def clear_database(self, confirm_code: str) -> Dict[str, Any]:
        """Clear all data from the database. Requires confirmation code.
        
        Args:
            confirm_code: Confirmation code (must be 'DELETE_ALL_DATA')
            
        Returns:
            Dictionary with operation result
        """
        if confirm_code != 'DELETE_ALL_DATA':
            return {
                "success": False,
                "message": "Invalid confirmation code. Must be 'DELETE_ALL_DATA'"
            }
            
        if not self.driver or not self._is_connected:
            if not self.connect():
                logger.error("Neo4j connection not available")
                return {"success": False, "message": "Neo4j connection not available"}
        
        try:
            with self._get_session() as session:
                # Delete all relationships
                rel_query = """
                MATCH ()-[r]->()
                DELETE r
                """
                session.run(rel_query)
                
                # Delete all nodes
                node_query = """
                MATCH (n)
                DELETE n
                """
                session.run(node_query)
                
                # Re-initialize schema
                self._schema_initialized = False
                self._setup_schema()
                
                return {
                    "success": True,
                    "message": "Database cleared successfully"
                }
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            self._metrics["last_error_time"] = datetime.now().isoformat()
            logger.error(f"Neo4j error clearing database: {e}")
            logger.debug(traceback.format_exc())
            return {"success": False, "message": f"Error clearing database: {str(e)}"}