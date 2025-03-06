"""
Ptolemy Knowledge Map System - Qdrant Service
==========================================
Service for vector similarity search operations using Qdrant.
"""

import logging
import time
import uuid
import os
import traceback
import threading
from typing import Dict, List, Optional, Any, Tuple, Union, Set, TypeVar, Callable
from datetime import datetime
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import Qdrant client with error handling
try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.models import (
        Distance, VectorParams, UpdateStatus, PointStruct, Filter, 
        FieldCondition, PayloadSchemaType, MatchValue, Range, SearchParams,
        ScoredPoint, Batch, RecommendStrategy, RecommendRequest,MatchText
    )
    from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException
    QDRANT_AVAILABLE = True
except ImportError as e:
    QDRANT_AVAILABLE = False
    logging.error(f"Qdrant client not available: {e}. Install with 'pip install qdrant-client'")

# Numpy for vector operations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.error("NumPy not available. Install with 'pip install numpy'")

from config import QdrantConfig
from models import ConceptType, ConceptSimilarityResult

# Configure module-level logger
logger = logging.getLogger("qdrant.service")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Type hint for reusable decorator
F = TypeVar('F', bound=Callable[..., Any])

def retry_on_exception(max_retries: int = 3, initial_delay: float = 1.0,
                      backoff_factor: float = 2.0, 
                      exceptions: Tuple = (Exception,)) -> Callable[[F], F]:
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

class QdrantService:
    """Service for vector database operations using Qdrant."""
    
    def __init__(self, config: QdrantConfig):
        """Initialize the Qdrant service with the given configuration.
        
        Args:
            config: Configuration for Qdrant connection and collection
        """
        self.config = config
        self.client = None
        self._lock = threading.RLock()  # For thread safety
        
        # Service state
        self._is_connected = False
        self._collection_ready = False
        self._last_error = None
        
        # Metrics
        self._metrics = {
            "points_stored": 0,
            "points_deleted": 0,
            "searches_performed": 0,
            "batches_processed": 0,
            "errors": 0,
            "connection_attempts": 0,
            "last_operation_time": None
        }
        
        # Executor for parallel operations
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers or 5)
        
        # Connect to Qdrant
        if QDRANT_AVAILABLE:
            self.connect()
            if self._is_connected:
                self._setup_collection()
        else:
            logger.error("Qdrant service initialized in limited mode - required libraries not available")
    
    @retry_on_exception(max_retries=3, initial_delay=2.0, 
                      exceptions=(ResponseHandlingException, ConnectionError, TimeoutError))
    def connect(self) -> bool:
        """Establish connection to Qdrant with retry logic.
        
        Returns:
            bool: True if connected successfully, False otherwise
        """
        self._metrics["connection_attempts"] += 1
        
        with self._lock:
            try:
                logger.info(f"Connecting to Qdrant at {self.config.url}")
                
                # Configure client options
                client_kwargs = {
                    "url": self.config.url,
                    "timeout": self.config.timeout
                }
                
                # Add API key if specified
                if self.config.api_key:
                    client_kwargs["api_key"] = self.config.api_key
                
                # Connect to Qdrant
                self.client = QdrantClient(**client_kwargs)
                
                # Test connection
                self.client.get_collections()
                
                self._is_connected = True
                self._last_error = None
                logger.info("Connected to Qdrant successfully")
                return True
            except Exception as e:
                self._metrics["errors"] += 1
                self._is_connected = False
                self._last_error = str(e)
                logger.error(f"Failed to connect to Qdrant: {e}")
                self.client = None
                return False
    
    @retry_on_exception(max_retries=2)
    @log_execution_time
    def _setup_collection(self) -> bool:
        """Set up Qdrant collection for storing concept embeddings.
        
        Returns:
            bool: True if collection is ready, False otherwise
        """
        if not self.client or not self._is_connected:
            logger.error("Cannot set up Qdrant collection: No connection")
            return False

        try:
            with self._lock:
                # Get existing collections
                collections = self.client.get_collections().collections
                collection_names = [c.name for c in collections]
                
                # Create collection if it doesn't exist
                if self.config.collection_name not in collection_names:
                    # Determine distance metric
                    distance = Distance.COSINE
                    if self.config.distance.lower() == "euclidean":
                        distance = Distance.EUCLID
                    elif self.config.distance.lower() == "dot":
                        distance = Distance.DOT
                    
                    # Create collection with configured parameters
                    self.client.create_collection(
                        collection_name=self.config.collection_name,
                        vectors_config=VectorParams(
                            size=self.config.vector_size,
                            distance=distance
                        ),
                        optimizers_config=models.OptimizersConfigDiff(
                            indexing_threshold=self.config.indexing_threshold or 20000
                        ),
                        shard_number=self.config.shard_number,
                        replication_factor=self.config.replication_factor,
                        write_consistency_factor=self.config.write_consistency_factor,
                        on_disk_payload=self.config.on_disk_payload
                    )
                    
                    # Create payload indexes for faster filtering
                    self._create_payload_indexes()
                    
                    logger.info(f"Created Qdrant collection: {self.config.collection_name}")
                else:
                    # Check if collection needs updating
                    try:
                        self._update_collection_if_needed()
                    except Exception as e:
                        logger.warning(f"Failed to update collection: {e}")
                    
                    logger.info(f"Qdrant collection already exists: {self.config.collection_name}")
                
                self._collection_ready = True
                return True
        except Exception as e:
            self._metrics["errors"] += 1
            self._collection_ready = False
            self._last_error = str(e)
            logger.error(f"Failed to set up Qdrant collection: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def _create_payload_indexes(self) -> None:
        """Create payload indexes for efficient filtering."""
        if not self.client or not self._is_connected:
            return
            
        try:
            # Create standard indexes
            index_fields = [
                {"name": "concept_type", "type": PayloadSchemaType.KEYWORD},
                {"name": "difficulty", "type": PayloadSchemaType.KEYWORD},
                {"name": "importance", "type": PayloadSchemaType.FLOAT},
                {"name": "name", "type": PayloadSchemaType.TEXT},
                {"name": "created_at", "type": PayloadSchemaType.DATETIME}
            ]
            
            # Add service namespacing index if shared collections are enabled
            if self.config.shared_collections_enabled:
                index_fields.append({"name": self.config.namespacing_field, "type": PayloadSchemaType.KEYWORD})
            
            for field in index_fields:
                try:
                    self.client.create_payload_index(
                        collection_name=self.config.collection_name,
                        field_name=field["name"],
                        field_schema=field["type"]
                    )
                except Exception as e:
                    # Index might already exist, continue
                    logger.debug(f"Error creating index for {field['name']}: {e}")
                    continue
            
            logger.info(f"Created payload indexes for collection {self.config.collection_name}")
        except Exception as e:
            logger.error(f"Failed to create payload indexes: {e}")
    
    def _update_collection_if_needed(self) -> None:
        """Update collection settings if needed."""
        try:
            # Get current collection info
            collection_info = self.client.get_collection(self.config.collection_name)
            
            # Check vector size
            current_size = collection_info.config.params.vectors.size
            if current_size != self.config.vector_size:
                logger.warning(
                    f"Vector size mismatch: configured {self.config.vector_size}, " 
                    f"but collection has {current_size}"
                )
        except Exception as e:
            logger.warning(f"Failed to verify collection settings: {e}")
    
    def close(self) -> None:
        """Close the Qdrant client connection and clean up resources."""
        logger.info("Closing Qdrant service")
        
        with self._lock:
            # Close thread pool
            if hasattr(self, '_executor') and self._executor:
                self._executor.shutdown(wait=False)
            
            # Close client connection
            if self.client:
                self.client = None
                self._is_connected = False
            
            logger.info("Qdrant connection closed")
    
    @retry_on_exception(max_retries=1)
    @log_execution_time
    def health_check(self) -> Dict[str, Any]:
        """Check Qdrant connection health and return metrics.
        
        Returns:
            Dict containing health status and metrics
        """
        if not QDRANT_AVAILABLE:
            return {
                "status": "unavailable", 
                "error": "Qdrant client library not installed",
                "available": False
            }
            
        if not self.client or not self._is_connected:
            self._attempt_reconnect()
            if not self._is_connected:
                return {
                    "status": "disconnected", 
                    "error": self._last_error or "No connection to Qdrant",
                    "available": True
                }
        
        try:
            # Basic connectivity check
            collections = self.client.get_collections().collections
            
            # Collection check
            collection_exists = self.config.collection_name in [c.name for c in collections]
            if not collection_exists:
                return {
                    "status": "degraded",
                    "error": f"Collection '{self.config.collection_name}' not found",
                    "available": True,
                    "metrics": self._metrics,
                    "collection_count": len(collections)
                }
            
            # Get collection stats
            collection_info = self.client.get_collection(self.config.collection_name)
            
            # Count points
            count_result = self.client.count(
                collection_name=self.config.collection_name,
                exact=True
            )
            
            # Get Qdrant version if available
            version = "unknown"
            try:
                telemetry = self.client.get_telemetry()
                version = telemetry.get("version", "unknown")
            except:
                pass
            
            return {
                "status": "connected",
                "version": version,
                "collection_count": len(collections),
                "vector_count": count_result.count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": str(collection_info.config.params.vectors.distance),
                "shard_number": collection_info.config.params.shard_number,
                "available": True,
                "metrics": self._metrics,
                "collection_ready": self._collection_ready
            }
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            logger.error(f"Qdrant health check failed: {e}")
            
            # Attempt to reconnect
            self._attempt_reconnect()
            
            return {
                "status": "error", 
                "error": str(e),
                "available": True,
                "metrics": self._metrics
            }
    
    def _attempt_reconnect(self) -> bool:
        """Attempt to reconnect to Qdrant if disconnected.
        
        Returns:
            bool: True if reconnection successful, False otherwise
        """
        if self._is_connected:
            return True
            
        try:
            logger.info("Attempting to reconnect to Qdrant")
            success = self.connect()
            if success:
                self._setup_collection()
            return success
        except Exception as e:
            logger.error(f"Reconnection attempt failed: {e}")
            return False
    
    @retry_on_exception(max_retries=2)
    @log_execution_time
    def store_embedding(self, concept_id: str, embedding: List[float], 
                        metadata: Dict[str, Any] = None) -> bool:
        """Store a concept embedding in Qdrant.
        
        Args:
            concept_id: ID of the concept
            embedding: Vector embedding of the concept
            metadata: Additional metadata to store
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client or not self._is_connected:
            if not self._attempt_reconnect():
                logger.error("Qdrant connection not available")
                return False
        
        if not concept_id or not embedding:
            logger.error("Missing concept ID or embedding")
            return False
        
        try:
            # Validate embedding dimensions
            if len(embedding) != self.config.vector_size:
                logger.error(
                    f"Embedding size mismatch: expected {self.config.vector_size}, got {len(embedding)}"
                )
                return False
            
            # Prepare metadata
            payload = metadata or {}
            
            # Add timestamp if not present
            if "created_at" not in payload:
                payload["created_at"] = datetime.now().isoformat()
            
            # Add service namespace to payload
            if self.config.shared_collections_enabled:
                payload[self.config.namespacing_field] = self.config.namespacing_value
                
                # Add service prefix to ID if using shared collections
                vector_id = f"{self.config.service_prefix}{concept_id}" if not concept_id.startswith(self.config.service_prefix) else concept_id
            else:
                vector_id = concept_id
                
            # Store in Qdrant
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=[
                    PointStruct(
                        id=vector_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
            
            # Update metrics
            self._metrics["points_stored"] += 1
            self._metrics["last_operation_time"] = datetime.now().isoformat()
            
            return True
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            logger.error(f"Qdrant error storing embedding for {concept_id}: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    @retry_on_exception(max_retries=2)
    @log_execution_time
    def store_batch_embeddings(self, points: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Store multiple concept embeddings in Qdrant.
        
        Args:
            points: List of dicts with 'id', 'vector', and optional 'payload'
            
        Returns:
            Tuple containing:
              - bool: True if all points were stored successfully
              - List[str]: List of successfully stored point IDs
        """
        if not self.client or not self._is_connected:
            if not self._attempt_reconnect():
                logger.error("Qdrant connection not available")
                return False, []
        
        if not points:
            logger.warning("No points provided for batch storage")
            return True, []
        
        try:
            # Process in batches if the input is large
            batch_size = self.config.batch_size or 100
            
            if len(points) > batch_size:
                return self._store_large_batch(points, batch_size)
            
            # Prepare point structs
            now = datetime.now().isoformat()
            point_structs = []
            valid_ids = []
            
            for point in points:
                # Validate required fields
                if 'id' not in point or 'vector' not in point:
                    logger.warning(f"Invalid point data: missing id or vector")
                    continue
                
                # Validate vector dimension
                if len(point['vector']) != self.config.vector_size:
                    logger.warning(
                        f"Vector size mismatch for {point['id']}: expected {self.config.vector_size}, "
                        f"got {len(point['vector'])}"
                    )
                    continue
                
                # Prepare metadata
                payload = point.get('payload', {}) or {}
                
                # Add timestamp if not present
                if "created_at" not in payload:
                    payload["created_at"] = now
                
                # Create point struct
                point_structs.append(
                    PointStruct(
                        id=point['id'],
                        vector=point['vector'],
                        payload=payload
                    )
                )
                valid_ids.append(point['id'])
            
            if not point_structs:
                logger.warning("No valid points to store after validation")
                return False, []
            
            # Store in Qdrant
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=point_structs,
                wait=True
            )
            
            # Update metrics
            self._metrics["points_stored"] += len(point_structs)
            self._metrics["batches_processed"] += 1
            self._metrics["last_operation_time"] = datetime.now().isoformat()
            
            logger.info(f"Successfully stored {len(point_structs)} embeddings in batch")
            return True, valid_ids
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            logger.error(f"Qdrant error storing batch embeddings: {e}")
            logger.debug(traceback.format_exc())
            return False, []
    
    def _store_large_batch(self, points: List[Dict[str, Any]], batch_size: int) -> Tuple[bool, List[str]]:
        """Handle storage of large batches by splitting into smaller ones.
        
        Args:
            points: List of point data
            batch_size: Maximum batch size
            
        Returns:
            Tuple containing:
              - bool: True if all batches were successful
              - List[str]: List of successfully stored point IDs
        """
        logger.info(f"Processing large batch of {len(points)} points with batch size {batch_size}")
        
        # Split into batches
        batches = [points[i:i+batch_size] for i in range(0, len(points), batch_size)]
        
        # Process batches
        all_successful = True
        all_successful_ids = []
        
        for i, batch in enumerate(batches):
            logger.debug(f"Processing batch {i+1}/{len(batches)} with {len(batch)} points")
            success, ids = self.store_batch_embeddings(batch)
            
            if success:
                all_successful_ids.extend(ids)
            else:
                all_successful = False
        
        return all_successful, all_successful_ids
    
    @retry_on_exception(max_retries=2)
    @log_execution_time
    def delete_embedding(self, concept_id: str) -> bool:
        """Delete a concept embedding from Qdrant.
        
        Args:
            concept_id: ID of the concept to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client or not self._is_connected:
            if not self._attempt_reconnect():
                logger.error("Qdrant connection not available")
                return False
        
        if not concept_id:
            logger.error("No concept ID provided for deletion")
            return False
        
        try:
            # Check if point exists
            try:
                result = self.client.retrieve(
                    collection_name=self.config.collection_name,
                    ids=[concept_id],
                    with_vectors=False
                )
                if not result:
                    logger.warning(f"Concept {concept_id} not found in Qdrant, nothing to delete")
                    return True  # Consider it success since the point is already gone
            except:
                # If retrieve fails, proceed with deletion anyway
                pass
            
            # Delete from Qdrant
            self.client.delete(
                collection_name=self.config.collection_name,
                points_selector=[concept_id],
                wait=True
            )
            
            # Update metrics
            self._metrics["points_deleted"] += 1
            self._metrics["last_operation_time"] = datetime.now().isoformat()
            
            return True
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            logger.error(f"Qdrant error deleting embedding {concept_id}: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    @retry_on_exception(max_retries=2)
    @log_execution_time
    def delete_batch_embeddings(self, concept_ids: List[str]) -> Tuple[bool, List[str]]:
        """Delete multiple concept embeddings from Qdrant.
        
        Args:
            concept_ids: List of concept IDs to delete
            
        Returns:
            Tuple containing:
              - bool: True if all deletions were successful
              - List[str]: List of successfully deleted concept IDs
        """
        if not self.client or not self._is_connected:
            if not self._attempt_reconnect():
                logger.error("Qdrant connection not available")
                return False, []
        
        if not concept_ids:
            logger.warning("No concept IDs provided for batch deletion")
            return True, []
        
        try:
            # Process in batches for large deletions
            batch_size = self.config.batch_size or 100
            
            if len(concept_ids) > batch_size:
                return self._delete_large_batch(concept_ids, batch_size)
            
            # Delete from Qdrant
            self.client.delete(
                collection_name=self.config.collection_name,
                points_selector=concept_ids,
                wait=True
            )
            
            # Update metrics
            self._metrics["points_deleted"] += len(concept_ids)
            self._metrics["last_operation_time"] = datetime.now().isoformat()
            
            logger.info(f"Successfully deleted {len(concept_ids)} embeddings in batch")
            return True, concept_ids
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            logger.error(f"Qdrant error deleting batch embeddings: {e}")
            logger.debug(traceback.format_exc())
            return False, []
    
    def _delete_large_batch(self, concept_ids: List[str], batch_size: int) -> Tuple[bool, List[str]]:
        """Handle deletion of large batches by splitting into smaller ones.
        
        Args:
            concept_ids: List of concept IDs to delete
            batch_size: Maximum batch size
            
        Returns:
            Tuple containing:
              - bool: True if all batches were successful
              - List[str]: List of successfully deleted concept IDs
        """
        logger.info(f"Processing large deletion of {len(concept_ids)} points with batch size {batch_size}")
        
        # Split into batches
        batches = [concept_ids[i:i+batch_size] for i in range(0, len(concept_ids), batch_size)]
        
        # Process batches
        all_successful = True
        all_successful_ids = []
        
        for i, batch in enumerate(batches):
            logger.debug(f"Processing deletion batch {i+1}/{len(batches)} with {len(batch)} points")
            success, ids = self.delete_batch_embeddings(batch)
            
            if success:
                all_successful_ids.extend(ids)
            else:
                all_successful = False
        
        return all_successful, all_successful_ids
    
    @retry_on_exception(max_retries=1)
    @log_execution_time
    def search_similar(self, embedding: List[float], limit: int = 10, 
                      concept_types: Optional[List[ConceptType]] = None,
                      min_importance: Optional[float] = None,
                      search_params: Optional[Dict[str, Any]] = None) -> List[ConceptSimilarityResult]:
        """Search for concepts similar to a given vector embedding.
        
        Args:
            embedding: Vector embedding to search with
            limit: Maximum number of results to return
            concept_types: Optional filter for concept types
            min_importance: Optional minimum importance score
            search_params: Optional additional search parameters
            
        Returns:
            List of concept similarity results
        """
        if not self.client or not self._is_connected:
            if not self._attempt_reconnect():
                logger.error("Qdrant connection not available")
                return []
        
        if not embedding:
            logger.error("No embedding provided for search")
            return []
        
        # Validate embedding dimension
        if len(embedding) != self.config.vector_size:
            logger.error(
                f"Embedding size mismatch: expected {self.config.vector_size}, got {len(embedding)}"
            )
            return []
        
        try:
            # Build filter conditions
            filter_conditions = []
            
            # Add service namespace filter if shared collections are enabled
            if self.config.shared_collections_enabled:
                filter_conditions.append(
                    FieldCondition(
                        key=self.config.namespacing_field,
                        match=MatchValue(value=self.config.namespacing_value)
                    )
                )
            
            # Filter by concept type
            if concept_types:
                # Convert enum values to strings if needed
                concept_types_str = []
                for ct in concept_types:
                    if hasattr(ct, "value"):
                        concept_types_str.append(ct.value)
                    else:
                        concept_types_str.append(str(ct))
                
                if concept_types_str:
                    filter_conditions.append(
                        FieldCondition(
                            key="concept_type",
                            match=MatchValue(value=concept_types_str)
                        )
                    )
            
            # Filter by importance
            if min_importance is not None:
                filter_conditions.append(
                    FieldCondition(
                        key="importance",
                        range=Range(
                            gte=float(min_importance)
                        )
                    )
                )
            
            # Set up query filter
            search_filter = Filter(must=filter_conditions) if filter_conditions else None
            
            # Set up search parameters
            params = SearchParams(
                hnsw_ef=search_params.get("hnsw_ef", 128) if search_params else 128,
                exact=search_params.get("exact", False) if search_params else False
            )
            
            # Execute search
            results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=embedding,
                limit=limit,
                query_filter=search_filter,
                search_params=params,
                with_payload=True,
                score_threshold=search_params.get("score_threshold") if search_params else None
            )
            
            # Update metrics
            self._metrics["searches_performed"] += 1
            self._metrics["last_operation_time"] = datetime.now().isoformat()
            
            # Process results
            similarity_results = self._process_search_results(results)
            
            logger.debug(f"Found {len(similarity_results)} similar concepts")
            return similarity_results
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            logger.error(f"Qdrant error searching similar concepts: {e}")
            logger.debug(traceback.format_exc())
            return []
    
    def _process_search_results(self, results: List[ScoredPoint]) -> List[ConceptSimilarityResult]:
        """Process search results into standardized format.
        
        Args:
            results: Raw search results from Qdrant
            
        Returns:
            List of formatted concept similarity results
        """
        similarity_results = []
        
        for res in results:
            payload = res.payload or {}
            
            # Get concept type - handle both string and enum values
            concept_type_str = payload.get("concept_type", "topic")
            try:
                concept_type = ConceptType(concept_type_str)
            except (ValueError, TypeError):
                concept_type = ConceptType.TOPIC  # Default if invalid
            
            # Create result object
            similarity_results.append(
                ConceptSimilarityResult(
                    concept_id=str(res.id),
                    concept_name=payload.get("name", "Unknown"),
                    similarity=float(res.score),
                    concept_type=concept_type
                )
            )
        
        return similarity_results
    
    @retry_on_exception(max_retries=1)
    @log_execution_time
    def concept_similarity(self, concept_id1: str, concept_id2: str) -> Optional[float]:
        """Calculate similarity between two concepts using their embeddings.
        
        Args:
            concept_id1: ID of the first concept
            concept_id2: ID of the second concept
            
        Returns:
            Similarity score between 0 and 1, or None if calculation fails
        """
        if not self.client or not self._is_connected:
            if not self._attempt_reconnect():
                logger.error("Qdrant connection not available")
                return None
        
        if not concept_id1 or not concept_id2:
            logger.error("Both concept IDs are required")
            return None
        
        try:
            # Retrieve vectors for both concepts
            vectors = self.client.retrieve(
                collection_name=self.config.collection_name,
                ids=[concept_id1, concept_id2],
                with_vectors=True
            )
            
            if len(vectors) != 2:
                missing_ids = []
                found_ids = [point.id for point in vectors]
                if concept_id1 not in found_ids:
                    missing_ids.append(concept_id1)
                if concept_id2 not in found_ids:
                    missing_ids.append(concept_id2)
                    
                logger.warning(f"Concept embeddings not found for: {', '.join(missing_ids)}")
                return None
            
            # Extract vectors
            vec1 = None
            vec2 = None
            for vec in vectors:
                if vec.id == concept_id1:
                    vec1 = vec.vector
                elif vec.id == concept_id2:
                    vec2 = vec.vector
            
            if vec1 and vec2:
                # Option 1: Use Qdrant search for accurate similarity calculation
                result = self.client.search(
                    collection_name=self.config.collection_name,
                    query_vector=vec1,
                    limit=1,
                    query_filter=Filter(
                        must=[
                            FieldCondition(
                                key="id",
                                match=MatchValue(value=concept_id2)
                            )
                        ]
                    )
                )
                
                if result:
                    similarity = result[0].score
                    # Ensure similarity is between 0 and 1
                    similarity = max(0.0, min(1.0, float(similarity)))
                    return similarity
                
                # Option 2: Calculate ourselves if search fails
                if NUMPY_AVAILABLE:
                    v1 = np.array(vec1)
                    v2 = np.array(vec2)
                    
                    # Cosine similarity formula: dot(v1, v2) / (|v1| * |v2|)
                    dot_product = np.dot(v1, v2)
                    norm_v1 = np.linalg.norm(v1)
                    norm_v2 = np.linalg.norm(v2)
                    
                    if norm_v1 > 0 and norm_v2 > 0:
                        similarity = dot_product / (norm_v1 * norm_v2)
                        # Ensure similarity is between 0 and 1
                        similarity = max(0.0, min(1.0, float(similarity)))
                        return similarity
            
            return None
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            logger.error(f"Qdrant error calculating concept similarity: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    @retry_on_exception(max_retries=1)
    @log_execution_time
    def get_nearest_neighbors(self, concept_id: str, limit: int = 10, 
                             min_similarity: float = 0.0) -> List[ConceptSimilarityResult]:
        """Get nearest neighboring concepts for a given concept ID.
        
        Args:
            concept_id: ID of the concept
            limit: Maximum number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of similar concepts
        """
        if not self.client or not self._is_connected:
            if not self._attempt_reconnect():
                logger.error("Qdrant connection not available")
                return []
        
        if not concept_id:
            logger.error("No concept ID provided")
            return []
        
        try:
            # First check if concept exists
            concept_vectors = self.client.retrieve(
                collection_name=self.config.collection_name,
                ids=[concept_id],
                with_vectors=True
            )
            
            if not concept_vectors:
                logger.warning(f"Concept embedding not found: {concept_id}")
                return []
            
            # Extract embedding
            embedding = concept_vectors[0].vector
            
            # Search for similar concepts
            results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=embedding,
                limit=limit + 1,  # Account for the concept itself
                with_payload=True,
                score_threshold=min_similarity
            )
            
            # Update metrics
            self._metrics["searches_performed"] += 1
            self._metrics["last_operation_time"] = datetime.now().isoformat()
            
            # Process results, skipping the original concept
            similarity_results = []
            for res in results:
                if str(res.id) != concept_id:  # Skip the original concept
                    payload = res.payload or {}
                    
                    # Get concept type
                    concept_type_str = payload.get("concept_type", "topic")
                    try:
                        concept_type = ConceptType(concept_type_str)
                    except (ValueError, TypeError):
                        concept_type = ConceptType.TOPIC  # Default if invalid
                    
                    # Create result object
                    similarity_results.append(
                        ConceptSimilarityResult(
                            concept_id=str(res.id),
                            concept_name=payload.get("name", "Unknown"),
                            similarity=float(res.score),
                            concept_type=concept_type
                        )
                    )
                    
                    # Stop once we have enough results
                    if len(similarity_results) >= limit:
                        break
            
            logger.debug(f"Found {len(similarity_results)} nearest neighbors for concept {concept_id}")
            return similarity_results
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            logger.error(f"Qdrant error getting nearest neighbors: {e}")
            logger.debug(traceback.format_exc())
            return []
    
    @retry_on_exception(max_retries=1)
    @log_execution_time
    def find_clusters(self, threshold: float = 0.85, 
                      min_cluster_size: int = 3, 
                      max_points: int = 10000) -> List[List[str]]:
        """Find clusters of similar concepts based on embeddings.
        
        Args:
            threshold: Similarity threshold for clustering
            min_cluster_size: Minimum number of concepts in a cluster
            max_points: Maximum number of points to process
            
        Returns:
            List of clusters, where each cluster is a list of concept IDs
        """
        if not self.client or not self._is_connected:
            if not self._attempt_reconnect():
                logger.error("Qdrant connection not available")
                return []
                
        if not NUMPY_AVAILABLE:
            logger.error("NumPy is required for clustering but not available")
            return []
        
        try:
            # Get all concept IDs (up to max_points)
            all_points = []
            offset = None
            
            while len(all_points) < max_points:
                scroll_result = self.client.scroll(
                    collection_name=self.config.collection_name,
                    limit=1000,  # Batch size for scrolling
                    offset=offset,
                    with_payload=False,
                    with_vectors=False
                )
                
                if not scroll_result or not scroll_result[0]:
                    break
                    
                points_batch, offset = scroll_result
                all_points.extend(points_batch)
                
                if offset is None:
                    break
            
            if not all_points:
                logger.warning("No concepts found in collection")
                return []
                
            logger.info(f"Processing {len(all_points)} points for clustering")
            
            # Extract concept IDs
            concept_ids = [str(point.id) for point in all_points]
            
            # If too many points, use a different approach
            if len(concept_ids) > 1000:
                return self._find_clusters_sampling(concept_ids, threshold, min_cluster_size)
            
            # Standard clustering approach for smaller sets
            clusters = []
            processed_ids = set()
            
            # Process concepts in parallel
            futures = []
            
            with ThreadPoolExecutor(max_workers=min(10, len(concept_ids) // 20 + 1)) as executor:
                # Process concepts in chunks
                chunk_size = 50
                for i in range(0, len(concept_ids), chunk_size):
                    chunk = concept_ids[i:i+chunk_size]
                    future = executor.submit(
                        self._process_cluster_chunk, 
                        chunk, 
                        threshold, 
                        processed_ids
                    )
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        chunk_clusters = future.result()
                        for cluster in chunk_clusters:
                            if len(cluster) >= min_cluster_size:
                                clusters.append(cluster)
                                processed_ids.update(cluster)
                    except Exception as e:
                        logger.error(f"Error processing cluster chunk: {e}")
            
            # Add any remaining clusters from individual concepts
            for concept_id in concept_ids:
                if concept_id in processed_ids:
                    continue
                    
                neighbors = self.get_nearest_neighbors(concept_id, limit=100, min_similarity=threshold)
                neighbor_ids = [neighbor.concept_id for neighbor in neighbors 
                              if neighbor.similarity >= threshold]
                
                cluster = [concept_id] + neighbor_ids
                cluster = [cid for cid in cluster if cid not in processed_ids]
                
                if len(cluster) >= min_cluster_size:
                    clusters.append(cluster)
                    processed_ids.update(cluster)
            
            logger.info(f"Found {len(clusters)} clusters with threshold {threshold}")
            return clusters
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            logger.error(f"Qdrant error finding clusters: {e}")
            logger.debug(traceback.format_exc())
            return []
    
    def _process_cluster_chunk(self, concept_ids: List[str], threshold: float, 
                             processed_ids: Set[str]) -> List[List[str]]:
        """Process a chunk of concepts for clustering.
        
        Args:
            concept_ids: List of concept IDs to process
            threshold: Similarity threshold
            processed_ids: Set of already processed IDs
            
        Returns:
            List of clusters found in this chunk
        """
        clusters = []
        local_processed = set()
        
        # Check if IDs have been processed by other threads
        concept_ids = [cid for cid in concept_ids if cid not in processed_ids]
        
        for concept_id in concept_ids:
            if concept_id in local_processed:
                continue
                
            neighbors = self.get_nearest_neighbors(concept_id, limit=100, min_similarity=threshold)
            neighbor_ids = [neighbor.concept_id for neighbor in neighbors 
                          if neighbor.similarity >= threshold]
            
            cluster = [concept_id] + neighbor_ids
            
            # Remove already processed IDs
            cluster = [cid for cid in cluster if cid not in processed_ids and cid not in local_processed]
            
            if cluster:
                clusters.append(cluster)
                local_processed.update(cluster)
        
        return clusters
    
    def _find_clusters_sampling(self, concept_ids: List[str], threshold: float, 
                              min_cluster_size: int) -> List[List[str]]:
        """Find clusters using a sampling approach for large collections.
        
        Args:
            concept_ids: List of all concept IDs
            threshold: Similarity threshold
            min_cluster_size: Minimum cluster size
            
        Returns:
            List of clusters
        """
        logger.info(f"Using sampling approach for clustering {len(concept_ids)} concepts")
        
        # Sample a subset of concepts
        sample_size = min(500, len(concept_ids) // 10)
        
        # Use numpy for random sampling without replacement
        import numpy as np
        sample_indices = np.random.choice(len(concept_ids), size=sample_size, replace=False)
        sample_ids = [concept_ids[i] for i in sample_indices]
        
        # Use sampled concepts as potential cluster centers
        clusters = []
        processed_ids = set()
        
        for concept_id in sample_ids:
            if concept_id in processed_ids:
                continue
                
            try:
                neighbors = self.get_nearest_neighbors(
                    concept_id, 
                    limit=min_cluster_size * 3,  # Get more than needed
                    min_similarity=threshold
                )
                
                neighbor_ids = [n.concept_id for n in neighbors if n.similarity >= threshold]
                cluster = [concept_id] + neighbor_ids
                
                # Remove already processed
                cluster = [cid for cid in cluster if cid not in processed_ids]
                
                if len(cluster) >= min_cluster_size:
                    clusters.append(cluster)
                    processed_ids.update(cluster)
            except Exception as e:
                logger.warning(f"Error processing concept {concept_id} for clustering: {e}")
        
        logger.info(f"Found {len(clusters)} clusters using sampling approach")
        return clusters
    
    @retry_on_exception(max_retries=1)
    @log_execution_time
    def get_concept_vector(self, concept_id: str) -> Optional[List[float]]:
        """Retrieve the vector embedding for a concept.
        
        Args:
            concept_id: ID of the concept
            
        Returns:
            Vector embedding or None if not found
        """
        if not self.client or not self._is_connected:
            if not self._attempt_reconnect():
                logger.error("Qdrant connection not available")
                return None
        
        if not concept_id:
            logger.error("No concept ID provided")
            return None
        
        try:
            result = self.client.retrieve(
                collection_name=self.config.collection_name,
                ids=[concept_id],
                with_vectors=True
            )
            
            if not result:
                logger.warning(f"Concept embedding not found: {concept_id}")
                return None
                
            if not result[0].vector:
                logger.warning(f"Concept exists but has no vector: {concept_id}")
                return None
                
            return result[0].vector
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            logger.error(f"Qdrant error retrieving concept vector: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    @retry_on_exception(max_retries=1)
    @log_execution_time
    def update_metadata(self, concept_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a concept embedding.
        
        Args:
            concept_id: ID of the concept
            metadata: New metadata to set
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client or not self._is_connected:
            if not self._attempt_reconnect():
                logger.error("Qdrant connection not available")
                return False
        
        if not concept_id or not metadata:
            logger.error("Concept ID and metadata are required")
            return False
        
        try:
            # Check if concept exists
            exists = self.client.retrieve(
                collection_name=self.config.collection_name,
                ids=[concept_id],
                with_vectors=False,
                with_payload=False
            )
            
            if not exists:
                logger.warning(f"Cannot update metadata: Concept {concept_id} not found")
                return False
            
            # Update metadata
            self.client.set_payload(
                collection_name=self.config.collection_name,
                payload=metadata,
                points=[concept_id],
                wait=True
            )
            
            # Update metrics
            self._metrics["last_operation_time"] = datetime.now().isoformat()
            
            return True
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            logger.error(f"Qdrant error updating metadata for {concept_id}: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    @retry_on_exception(max_retries=1)
    @log_execution_time
    def scroll_concepts(self, offset: int = 0, limit: int = 100,
                        filter_conditions: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], int]:
        """Scroll through concepts with pagination.
        
        Args:
            offset: Starting offset for pagination
            limit: Maximum number of results to return
            filter_conditions: Optional filtering conditions
            
        Returns:
            Tuple containing:
              - List of concept data
              - Next offset for pagination or None if no more results
        """
        if not self.client or not self._is_connected:
            if not self._attempt_reconnect():
                logger.error("Qdrant connection not available")
                return [], 0
        
        try:
            # Build filter if provided
            search_filter = None
            if filter_conditions:
                filter_must = []
                
                # Handle concept types filter
                if 'concept_types' in filter_conditions:
                    concept_types = filter_conditions['concept_types']
                    # Convert provided concept types to strings
                    concept_types_str = []
                    for ct in concept_types:
                        if hasattr(ct, "value"):
                            concept_types_str.append(ct.value)
                        else:
                            concept_types_str.append(str(ct))
                    
                    if concept_types_str:
                        filter_must.append(
                            FieldCondition(
                                key="concept_type",
                                match=MatchValue(value=concept_types_str)
                            )
                        )
                
                # Handle importance filter
                if 'min_importance' in filter_conditions:
                    min_importance = float(filter_conditions['min_importance'])
                    filter_must.append(
                        FieldCondition(
                            key="importance",
                            range=Range(
                                gte=min_importance
                            )
                        )
                    )
                
                # Handle difficulty filter
                if 'difficulty' in filter_conditions:
                    difficulty = filter_conditions['difficulty']
                    if hasattr(difficulty, "value"):
                        difficulty = difficulty.value
                    filter_must.append(
                        FieldCondition(
                            key="difficulty",
                            match=MatchValue(value=difficulty)
                        )
                    )
                
                # Create filter if we have conditions
                if filter_must:
                    search_filter = Filter(must=filter_must)
            
            # Execute scroll query
            result, next_offset = self.client.scroll(
                collection_name=self.config.collection_name,
                limit=limit,
                offset=offset,
                filter=search_filter,
                with_payload=True,
                with_vectors=False
            )
            
            # Convert results to dictionaries
            concepts = []
            for point in result:
                # Create concept data
                concept = point.payload.copy() if point.payload else {}
                concept['id'] = str(point.id)
                concepts.append(concept)
            
            logger.debug(f"Scrolled {len(concepts)} concepts with offset {offset}")
            return concepts, next_offset
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            logger.error(f"Qdrant error scrolling concepts: {e}")
            logger.debug(traceback.format_exc())
            return [], 0
    
    @retry_on_exception(max_retries=1)
    @log_execution_time
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the concepts collection.
        
        Returns:
            Dictionary with collection information
        """
        if not self.client or not self._is_connected:
            if not self._attempt_reconnect():
                logger.error("Qdrant connection not available")
                return {}
        
        try:
            # Get collection info
            collection_info = self.client.get_collection(self.config.collection_name)
            
            # Count points
            count_result = self.client.count(
                collection_name=self.config.collection_name,
                exact=True
            )
            
            # Get collection statistics if available
            try:
                telemetry = self.client.get_telemetry()
            except:
                telemetry = {}
            
            # Build result
            return {
                "name": self.config.collection_name,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": str(collection_info.config.params.vectors.distance),
                "vector_count": count_result.count,
                "shard_number": collection_info.config.params.shard_number,
                "replication_factor": collection_info.config.params.replication_factor,
                "on_disk": collection_info.config.params.on_disk_payload,
                "version": telemetry.get("version", "unknown"),
                "indexing_threshold": getattr(collection_info.config.optimizers_config, "indexing_threshold", None)
            }
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            logger.error(f"Qdrant error getting collection info: {e}")
            logger.debug(traceback.format_exc())
            return {}
    
    @retry_on_exception(max_retries=1)
    @log_execution_time
    def search_by_text(self, query_embedding: List[float], query_text: str, limit: int = 10) -> List[ConceptSimilarityResult]:
        """Enhanced semantic search that combines vector similarity with text matching.
        
        Args:
            query_embedding: Vector embedding for semantic search
            query_text: Text query for keyword matching
            limit: Maximum number of results to return
            
        Returns:
            List of concept similarity results
        """
        if not self.client or not self._is_connected:
            if not self._attempt_reconnect():
                logger.error("Qdrant connection not available")
                return []
        
        if not query_embedding or not query_text:
            logger.error("Both query embedding and text are required")
            return []
        
        try:
            # First try combined search with text filter
            try:
                text_condition = FieldCondition(
                    key="name",
                    match=MatchText(text=query_text)
                )
                
                # Try with explicit text filter
                results = self.client.search(
                    collection_name=self.config.collection_name,
                    query_vector=query_embedding,
                    query_filter=Filter(must=[text_condition]),
                    limit=limit,
                    with_payload=True
                )
                
                if results:
                    # If we get results, process them
                    similarity_results = self._process_search_results(results)
                    
                    # Update metrics
                    self._metrics["searches_performed"] += 1
                    
                    return similarity_results
            except Exception as text_error:
                # If text matching fails, log and continue with fallback
                logger.warning(f"Text search failed, falling back to vector-only: {text_error}")
            
            # Fallback: Perform vector search only
            results = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_embedding,
                limit=limit * 2,  # Get more results for post-filtering
                with_payload=True
            )
            
            # Post-process to filter by text
            filtered_results = []
            for res in results:
                # Check if text appears in name or description
                name = res.payload.get("name", "").lower()
                description = res.payload.get("description", "").lower()
                query_lower = query_text.lower()
                
                if query_lower in name or query_lower in description:
                    filtered_results.append(res)
                    if len(filtered_results) >= limit:
                        break
            
            # If we have enough results after filtering, use them
            if filtered_results:
                similarity_results = self._process_search_results(filtered_results)
            else:
                # Otherwise, just use the top vector results
                similarity_results = self._process_search_results(results[:limit])
            
            # Update metrics
            self._metrics["searches_performed"] += 1
            self._metrics["last_operation_time"] = datetime.now().isoformat()
            
            return similarity_results
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            logger.error(f"Qdrant error performing text search: {e}")
            logger.debug(traceback.format_exc())
            return []
    
    @retry_on_exception(max_retries=2)
    @log_execution_time
    def clear_collection(self) -> bool:
        """Clear all vectors from the collection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client or not self._is_connected:
            if not self._attempt_reconnect():
                logger.error("Qdrant connection not available")
                return False
        
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.config.collection_name in collection_names:
                # Option 1: Delete and recreate
                logger.info(f"Deleting collection {self.config.collection_name}")
                self.client.delete_collection(self.config.collection_name)
                
                # Recreate the collection
                self._setup_collection()
                
                # Update metrics
                self._metrics["points_deleted"] += 1000  # Arbitrary large number
                self._metrics["last_operation_time"] = datetime.now().isoformat()
                
                return True
            else:
                # Collection doesn't exist, create it
                self._setup_collection()
                return True
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            logger.error(f"Qdrant error clearing collection: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    @retry_on_exception(max_retries=1)
    @log_execution_time
    def recommend_concepts(self, positive_examples: List[str], 
                          negative_examples: Optional[List[str]] = None,
                          limit: int = 10) -> List[ConceptSimilarityResult]:
        """Get concept recommendations based on positive and negative examples.
        
        Args:
            positive_examples: List of concept IDs to use as positive examples
            negative_examples: Optional list of concept IDs to avoid
            limit: Maximum number of recommendations to return
            
        Returns:
            List of recommended concepts
        """
        if not self.client or not self._is_connected:
            if not self._attempt_reconnect():
                logger.error("Qdrant connection not available")
                return []
        
        if not positive_examples:
            logger.error("No positive examples provided for recommendations")
            return []
        
        try:
            # Prepare recommendation query
            if negative_examples:
                request = RecommendRequest(
                    positive=positive_examples,
                    negative=negative_examples,
                    limit=limit,
                    with_payload=True,
                    strategy=RecommendStrategy.AVERAGE_VECTOR
                )
            else:
                request = RecommendRequest(
                    positive=positive_examples,
                    limit=limit,
                    with_payload=True,
                    strategy=RecommendStrategy.AVERAGE_VECTOR
                )
            
            # Get recommendations
            results = self.client.recommend(
                collection_name=self.config.collection_name,
                **request.__dict__
            )
            
            # Update metrics
            self._metrics["searches_performed"] += 1
            self._metrics["last_operation_time"] = datetime.now().isoformat()
            
            # Process results
            similarity_results = self._process_search_results(results)
            
            logger.debug(f"Found {len(similarity_results)} recommended concepts")
            return similarity_results
        except Exception as e:
            self._metrics["errors"] += 1
            self._last_error = str(e)
            logger.error(f"Qdrant error getting concept recommendations: {e}")
            logger.debug(traceback.format_exc())
            
            # Fallback to traditional search if recommendation fails
            try:
                logger.info("Falling back to traditional search for recommendations")
                # Get embeddings for positive examples
                embeddings = []
                for concept_id in positive_examples:
                    embedding = self.get_concept_vector(concept_id)
                    if embedding:
                        embeddings.append(embedding)
                
                if not embeddings:
                    return []
                
                # Calculate average embedding
                if NUMPY_AVAILABLE:
                    avg_embedding = np.mean(embeddings, axis=0).tolist()
                    
                    # Search similar concepts
                    results = self.search_similar(avg_embedding, limit=limit)
                    
                    # Filter out the positive examples
                    results = [
                        r for r in results 
                        if r.concept_id not in positive_examples and 
                        (not negative_examples or r.concept_id not in negative_examples)
                    ]
                    
                    return results[:limit]
                else:
                    return []
            except Exception as fallback_error:
                logger.error(f"Fallback recommendation also failed: {fallback_error}")
                return []